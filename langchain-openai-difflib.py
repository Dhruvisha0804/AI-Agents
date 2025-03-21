from pymongo import MongoClient
import json
import os
import faiss
import numpy as np
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from dotenv import load_dotenv
import logging
from datetime import datetime
from bson import ObjectId
import time
import tiktoken
from sentence_transformers import SentenceTransformer
import requests
import streamlit as st
import difflib

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

if not OPENAI_API_KEY:
    st.error("Missing OpenAI API Key. Set OPENAI_API_KEY in environment variables.")
    st.stop()

if not GROQ_API_KEY:
    logging.warning("GROQ_API_KEY is missing. Groq functionality will be disabled.")
    st.warning("GROQ API key not provided. Skipping human-like result generation.")

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
    print("****add_query_to_faiss called")
    vector = embedding_model.encode([question])
    index.add(np.array(vector, dtype=np.float32))
    stored_queries.append(question)
    query_mappings[question] = mongo_query

# MongoDB Local Connection
MONGO_URI = "mongodb://localhost:27017"
client = MongoClient(MONGO_URI)
db = client["task_demo"]
tasks_collection = db["tasks"]
users_collection = db["users"]
projects_collection = db["projects"]

filename = 'sample.txt'

st.title("MongoDB AI Agent with LangChain-OpenAI and FAISS")
st.write("Ask anything and get an answer")
input_text = st.text_area("Enter your question here")

def pretty_print_json(query):
    try:
        query_dict = json.loads(query)
        return json.dumps(query_dict, indent=4)
    except json.JSONDecodeError:
        return query

def read_samples(filename):
    samples = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        question = None
        query = None
        is_query = False
        for line in lines:
            line = line.strip()
            if line.startswith('Question'):
                if question and query:
                    samples.append((question, query))
                question = line[len('Question'):].strip()
                query = None
                is_query = False
            elif line.startswith('Query'):
                is_query = True
            elif is_query:
                if line.startswith('json'):
                    line = line[4:].strip()
                query = (query + " " + line) if query else line
        if question and query:
            samples.append((question, query))
    return samples

def find_similar_question(user_query, samples, threshold=0.6):
    questions = [sample[0] for sample in samples]
    queries = {sample[0]: sample[1] for sample in samples}
    matches = difflib.get_close_matches(user_query, questions, n=len(questions), cutoff=threshold)

    if not matches:
        return None, None, []

    results = []
    for match in matches:
        score = difflib.SequenceMatcher(None, user_query, match).ratio()
        results.append((match, score, queries[match]))

    best_match = results[0]
    return best_match[0], best_match[2], results

prompt = """
    You are an expert in converting English questions into MongoDB queries!
    The database is named 'task_demo' and contains three collections: 'tasks', 'users', and 'projects'.
    including nested and embedded data structures that add depth and detail to the document.
    The 'tasks' collection has fields: AssigneeUserId, TaskName, Status (embedded object with text, key, and type), ProjectID, Task_Priority, createdAt, SprintArray, Task_Leader etc.
    The 'users' collection has fields: _id, Employee_FName, Employee_LName, Employee_Name, createdAt, etc.
    The 'projects' collection has fields: _id, AssigneeUserid, LeaderUserId, ProjectCategory, ProjectName, etc.

    **Important Rule**: Always reference the `status.text` field when filtering by task status (e.g., `status.text: "Done"`).
    Always reference the `sprintArray.folderName` field when filtering by sprint folders (e.g., `sprintArray.folderName: "Development"`).

    **Additional Rule**:
    - When you need to filter tasks or query simple fields, you can use the `filter` field.
    - If the query requires aggregation, **do not include a `filter` field**. Instead, only include the aggregation pipeline.

    ***Relationship Between users, tasks, and projects***

    Each task (tasks.AssigneeUserId) can have multiple assignees (users._id), and users can be assigned to multiple tasks.
    Each task (tasks.ProjectID) belongs to a single project (projects._id), one project may have multiple assignees.

    Below are several sample user questions related to the MongoDB document provided,
    and the corresponding MongoDB aggregation pipeline queries that can be used to fetch the desired data.
    Use them wisely.

    sample_question: {similar_question}
    corresponding_query: {corresponding_query}

    As an expert you must use them whenever required. And note that us sample question you have to modify query as per the users question. 
    Note: You have to just return the query nothing else. Don't return any additional detail with the query. Please follow this strictly.
    input: {user_query}
    output:

    Please return only the MongoDB query for the user's question. The output should be a valid aggregation pipeline query.
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
                except Exception:
                    pass
            elif isinstance(value, dict):
                convert_to_objectid(value)
    elif isinstance(query, list):
        for i in range(len(query)):
            convert_to_objectid(query[i])
    return query

def execute_mongo_query(query):
    if not query:
        st.error("Query could not be generated. Please try a different question.")
        return []

    try:
        query = convert_dates(query)
        query = convert_to_objectid(query)

        # Ensure "collection" key exists
        if "collection" not in query:
            raise ValueError("Missing 'collection' key in query.")

        collection = db[query["collection"]]

        if "aggregate" in query:
            logging.info(f"Executing MongoDB aggregation query: {json.dumps(query, indent=4)}")
            results = list(collection.aggregate(query["aggregate"]))  # Convert cursor to list
        elif "filter" in query:
            logging.info(f"Executing MongoDB filter query: {json.dumps(query, indent=4)}")
            results = list(collection.find(query["filter"], query.get("projection", {})))  # Convert cursor to list
        else:
            raise ValueError("Query must contain either 'aggregate' or 'filter' field.")

        return results

    except Exception as e:
        logging.error(f"Error executing query: {str(e)}")
        return [f"Error: {str(e)}"]



def count_tokens(text, model_name="gpt-3.5-turbo"):
    encoding = tiktoken.encoding_for_model(model_name)
    tokens = encoding.encode(text)
    return len(tokens)

def get_groq_response(question, query_results):
    if not GROQ_API_KEY:
        return "GROQ API key not configured."
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
    input_variables=["similar_question","corresponding_query", "user_query"]
)
llmchain = LLMChain(llm=llm, prompt=query_with_prompt, verbose=True)

if input_text:
    button = st.button("Submit")
    if button:
        start_time = time.time()
        try:
            question = input_text
            samples = read_samples(filename)
            similar_question, corresponding_query, results = find_similar_question(question, samples, threshold=0.6)

            if similar_question and corresponding_query:
                stored_query = corresponding_query
            else:
                stored_query = None

            input_prompt = query_with_prompt.format(similar_question=similar_question, corresponding_query=stored_query, user_query=question)

            input_tokens_used = count_tokens(input_prompt)
            st.write(f"Tokens used for input prompt: {input_tokens_used}")

            if stored_query:
                query = json.loads(stored_query)
                st.write(f"Using cached query for: {similar_question}")
            else:
                st.write("No exact match found, generating a new query.")

                response = llmchain.invoke({
                    "similar_question": similar_question,
                    "corresponding_query": stored_query,
                    "user_query": question
                })
                query_text = response["text"]
                query_tokens_used = count_tokens(query_text)

                # Validate AI response before adding to FAISS
                try:
                    query = json.loads(query_text)
                    add_query_to_faiss(question, query)  # Only add if valid
                except json.JSONDecodeError:
                    logging.error(f"Invalid JSON received: {query_text}")
                    st.error("AI-generated query is not valid JSON. Please rephrase your question.")

                query_generation_time = time.time() - start_time
                st.write(f"Time taken to generate the query: {query_generation_time:.2f} seconds")
                st.write(f"Tokens used for query generation: {query_tokens_used}")

            st.subheader("Generated MongoDB Query:")
            st.text(json.dumps(query, indent=4))

            # Execute query
            query_start_time = time.time()
            query_results = execute_mongo_query(query)
            query_results_time = time.time() - query_start_time

            # Ensure results are iterable
            if isinstance(query_results, list):
                st.subheader("Query Results:")
                for row in query_results:
                    st.write(row)

                st.write(f"Time taken to fetch results: {query_results_time:.2f} seconds")
                result_text = json.dumps(query_results)
                result_tokens_used = count_tokens(result_text)
                st.write(f"Tokens used for results: {result_tokens_used}")

                # Generate human-readable output
                human_start_time = time.time()
                human_readable_output = get_groq_response(question, query_results)
                human_like_gen_time = time.time() - human_start_time

                st.subheader("Final Result:")
                st.write(human_readable_output)
                st.write(f"Time taken to generate human-like sentence: {human_like_gen_time:.2f} seconds")
            else:
                st.error(f"Query execution failed: {query_results}")

        except json.JSONDecodeError as e:
            st.error(f"Error parsing JSON: {e}. LLM output was not valid JSON.")
        except Exception as e:
            st.error(f"Error executing query: {e}")


user_query = input_text
samples = read_samples(filename)

similar_question, corresponding_query, results = find_similar_question(user_query, samples, threshold=0.6)

if similar_question:
    print(f"Similar Question: {similar_question}")
    print("Corresponding Query:")
    print(pretty_print_json(corresponding_query))
    print("\nAll Matches and Similarity Scores:")
    for question, score, query in results:
        print(f"Question: {question} \nScore: {score}\n")
else:
    print("No similar question found.")


