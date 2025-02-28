from pymongo import MongoClient
import io, json, os
import faiss
import numpy as np
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from dotenv import load_dotenv
import sys
import logging
from datetime import datetime
from bson import ObjectId
import time
import tiktoken
from sentence_transformers import SentenceTransformer
import requests
from bson.json_util import dumps
import streamlit as st

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

if not OPENAI_API_KEY:
    st.error("Missingimport streamlit as st OpenAI API Key. Set OPENAI_API_KEY in environment variables.")
    st.stop()

if not GROQ_API_KEY:
    logging.error("GROQ_API_KEY is missing. Please set it in the environment variables.")

# Initialize OpenAI LLM
llm = ChatOpenAI(model="gpt-4", temperature=0.0, openai_api_key=OPENAI_API_KEY)

# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize FAISS Index
query_dim = 384
index = faiss.IndexFlatL2(query_dim)
stored_queries = []
query_mappings = {}

def add_query_to_faiss(question, mongo_query):
    vector = embedding_model.encode([question])
    index.add(np.array(vector, dtype=np.float32))
    stored_queries.append(question)
    query_mappings[question] = mongo_query

def find_similar_query(user_query):
    vector = embedding_model.encode([user_query])
    distances, indices = index.search(np.array(vector, dtype=np.float32), 1)
    if distances[0][0] < 0.5:
        return stored_queries[indices[0][0]], query_mappings[stored_queries[indices[0][0]]]
    return None, None

# MongoDB Local Connection
MONGO_URI = "mongodb://localhost:27017"
client = MongoClient(MONGO_URI)
db = client["task_demo"]
tasks_collection = db["tasks"]
users_collection = db["users"]
projects_collection = db["projects"]

st.title("MongoDB AI Agent with LangChain-OpenAI and FAISS")
st.write("Ask anything and get an answer")
input_text = st.text_area("Enter your question here")

# Load sample file
with io.open("sample.txt", "r", encoding="utf-8") as f1:
    sample = f1.read()

prompt = """
    You are an expert in converting English questions into MongoDB queries for the 'task_demo' database. This database contains three collections: 'tasks', 'users', and 'projects', with nested and embedded data.

    **Collections Structure:**
    - 'tasks' collection: fields like AssigneeUserId, TaskName, Status (embedded with text, key, type), ProjectID, Task_Priority, createdAt, SprintArray, Task_Leader.
    - 'users' collection: fields like _id, Employee_FName, Employee_LName, Employee_Name, createdAt.
    - 'projects' collection: fields like _id, AssigneeUserid, LeaderUserId, ProjectCategory, ProjectName.

    **Important Rules:**
    1. Always reference `status.text` for task status filtering (e.g., `status.text: "Done"`).
    2. Always use `sprintArray.folderName` for sprint folder filtering (e.g., `sprintArray.folderName: "Development"`).
    3. For simple field filtering, use `filter`. For aggregation, only include the aggregation pipeline—no `filter` field.

    **Relationships:** Each task (tasks.AssigneeUserId) can have multiple assignees (users._id), and each task (tasks.ProjectID) belongs to one project (projects._id), with a project potentially having multiple assignees.

    **Task:** Use the provided sample question and corresponding query as a reference. Only return the MongoDB query, no additional details.

    sample_question: {sample}
    As an expert, use them as needed.

    input: {question}
    output:
"""

def convert_dates(query):
    if isinstance(query, dict):
        for key, value in query.items():
            if isinstance(value, dict) and "$date" in value:
                try:
                    query[key] = datetime.fromisoformat(value["$date"].replace("Z", "+00:00"))
                except ValueError:
                    pass
            else:
                convert_dates(value)
    elif isinstance(query, list):
        for i in range(len(query)):
            convert_dates(query[i])
    return query

def convert_to_objectid(query):
    if isinstance(query, dict):
        for key, value in query.items():
            if isinstance(value, str) and len(value) == 24:
                try:
                    query[key] = ObjectId(value)
                except Exception as e:
                    pass
            elif isinstance(value, dict):
                convert_to_objectid(value)
    elif isinstance(query, list):
        for i in range(len(query)):
            convert_to_objectid(query[i])
    return query


def execute_mongo_query(query):
    try:
        query = convert_dates(query)
        query = convert_to_objectid(query)
        if "aggregate" in query:
            logging.info(f"Executing MongoDB aggregation query: {query}")
            collection = db[query["collection"]]
            results = collection.aggregate(query["aggregate"])
        elif "filter" in query:
            logging.info(f"Executing MongoDB filter query: {query}")
            collection = db[query["collection"]]
            results = collection.find(query["filter"], query.get("projection", {}))
        else:
            raise ValueError("Query must contain either 'aggregate' or 'filter' field")
        return list(results)
        # return dumps(results)  # This will serialize the ObjectId as a string
    except Exception as e:
        logging.error(f"Error executing query: {str(e)}")
        return [f"Error: {str(e)}"]


def count_tokens(text, model_name="gpt-3.5-turbo"):
    encoding = tiktoken.encoding_for_model(model_name)
    tokens = encoding.encode(text)
    return len(tokens)

def get_groq_response(question, query_results):
    try:
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        prompt_groq = """
            You are the translator for the json formatted results from the MongoDB database to human-like language.
            Example:
            question: How many distinct tasks are there?
            generated query: {
                "collection": "tasks",
                "aggregate": [
                    {
                        "$count": "total_tasks"
                    }
                ]
            }
            Query results: {
            "total_tasks": 66543
            }

            Now you have to convert the query results into human-readable sentences.

            Final result: There are a total of 66543 distinct tasks in the tasks collection.
        """
        data = {
            "model": GROQ_MODEL,
            "messages": [
                {"role": "system", "content": prompt_groq},
                {"role": "user", "content": question},
                {"role": "user", "content": f"Query results: {query_results}"}
            ]
        }
        logging.info("Sending request to Groq API...")
        response = requests.post(GROQ_API_URL, json=data, headers=headers)

        if response.status_code == 200:
            result = response.json()
            logging.info("Received response from Groq API successfully.")
            return result["choices"][0]["message"]["content"].strip()
        else:
            logging.error(f"Groq API Error {response.status_code}: {response.text}")
            return f"Error: {response.json()}"
    except Exception as e:
        logging.error(f"Error in Groq API request: {e}")
        return f"API Error: {str(e)}"

# Prompt template and chain setup
query_with_prompt = PromptTemplate(
    template=prompt,
    input_variables=["question", "sample"]
)
llmchain = LLMChain(llm=llm, prompt=query_with_prompt, verbose=True)

if input_text:
    button = st.button("Submit")
    if button:
        start_time = time.time()
        try:
            question = input_text
            similar_question, stored_query = find_similar_query(question)

            # Count input tokens
            input_prompt = query_with_prompt.format(question=question, sample=sample)
            input_tokens_used = count_tokens(input_prompt)
            st.write(f"Tokens used for input prompt: {input_tokens_used}")
            
            if stored_query:
                query = stored_query
                st.write(f"Using cached query for: {similar_question}")
            else:
                response = llmchain.invoke({"question": question, "sample": sample})
                query_generation_time = time.time() - start_time
                query_text = response["text"]
                query_tokens_used = count_tokens(query_text)
                query = json.loads(response["text"])
                add_query_to_faiss(question, query) # add to faiss for future use.

                st.write(f"Time taken to generate the query: {query_generation_time:.2f} seconds")
                st.write(f"Tokens used for query generation: {query_tokens_used}")

            st.subheader("Generated MongoDB Query:")
            st.text(json.dumps(query, indent=4))

            start_time = time.time()
            query_results = execute_mongo_query(query)
            query_results_time = time.time() - start_time

            result_text = json.dumps(query_results)
            result_tokens_used = count_tokens(result_text)

            st.subheader("Query Results:")
            for row in query_results:
                st.write(row)

            st.write(f"Time taken to fetch results: {query_results_time:.2f} seconds")
            st.write(f"Tokens used for results: {result_tokens_used}")

            if query_results:
                human_like_gen_time = time.time() - start_time
                human_readable_output = get_groq_response(question, query_results)

                st.subheader("Final result:")
                st.write(human_readable_output)

                st.write(f"Time taken to generate human like sentence: {query_results_time:.2f} seconds")

            else:
                st.error("No query results returned from MongoDB.")

        except json.JSONDecodeError as e:
            st.error(f"Error parsing JSON: {e}. LLM output was not valid JSON.")
        except Exception as e:
            st.error(f"Error executing query: {e}")