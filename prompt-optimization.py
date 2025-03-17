import os
import json
import logging
import numpy as np
import faiss
from pymongo import MongoClient
from dotenv import load_dotenv
import time
import tiktoken
from sentence_transformers import SentenceTransformer
import requests
import streamlit as st
from datetime import datetime
from bson import ObjectId

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

if not GROQ_API_KEY:
    logging.error("GROQ_API_KEY is missing. Please set it in the environment variables.")

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

st.title("MongoDB AI Agent with Groq and FAISS")
st.write("Ask anything and get an answer")
input_text = st.text_area("Enter your question here")

# Load sample file (assuming sample.txt is still needed)
with open("sample.txt", "r", encoding="utf-8") as f1:
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

    **Task:** Use the provided sample question and corresponding query as a reference. Only return the MongoDB query in json format, no additional details.

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
    except Exception as e:
        logging.error(f"Error executing query: {str(e)}")
        return [f"Error: {str(e)}"]

def get_groq_response(question, query_results, prompt):
    try:
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        data = {
            "model": GROQ_MODEL,
            "messages": [
                {"role": "system", "content": prompt.format(question=question, sample=sample)},
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

if input_text:
    button = st.button("Submit")
    if button:
        start_time = time.time()
        error_occurred = False
        query = {}
        try:
            question = input_text
            similar_question, stored_query = find_similar_query(question)

            if stored_query:
                query = stored_query
                st.write(f"Using cached query for: {similar_question}")
            else:
                response_text = get_groq_response(question, "Generate the MongoDB query", prompt)
                query_generation_time = time.time() - start_time
                if not response_text.strip():
                    st.error("Groq API returned an empty response.")
                    error_occurred = True
                else:
                    if response_text.startswith("```json"):
                        response_text = response_text[7:]
                        if response_text.endswith("```"):
                            response_text = response_text[:-3]
                    try:
                        query = json.loads(response_text)
                        add_query_to_faiss(question, query)
                    except json.JSONDecodeError as e:
                        st.error(f"Error parsing JSON: {e}. Groq output was not valid JSON: {response_text}")
                        error_occurred = True
                if error_occurred:
                    pass
                else:
                    st.write(f"Time taken to generate the query: {query_generation_time:.2f} seconds")

                    st.subheader("Generated MongoDB Query:")
                    st.text(json.dumps(query, indent=4))

                    start_time = time.time()
                    query_results = execute_mongo_query(query)
                    query_results_time = time.time() - start_time

                    st.subheader("Query Results:")
                    for row in query_results:
                        st.write(row)

                    st.write(f"Time taken to fetch results: {query_results_time:.2f} seconds")

                    if query_results:
                        try:
                            # query_results_json = json.dumps(query_results) #convert to json string.
                            human_like_gen_time = time.time() - start_time
                            human_readable_output = get_groq_response(question, query_results, """
                                You are a translator for MongoDB query results in JSON format. Your task is to convert the results into a concise, human-readable sentence.
                                
                            """)

                            st.subheader("Final result:")
                            st.write(human_readable_output)

                            st.write(f"Time taken to generate human like sentence: {query_results_time:.2f} seconds")
                        except TypeError as e:
                            st.error(f"Error serializing query results to JSON: {e}")

                    else:
                        st.error("No query results returned from MongoDB.")

        except json.JSONDecodeError as e:
            st.error(f"Error parsing JSON: {e}. LLM output was not valid JSON.")
        except Exception as e:
            st.error(f"Error executing query: {e}")