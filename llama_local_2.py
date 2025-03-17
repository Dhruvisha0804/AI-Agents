import streamlit as st
from pymongo import MongoClient
import io, json, os
import logging
from datetime import datetime
from bson import ObjectId
import time
import transformers
import torch
import tiktoken
import re

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load local LLaMA model
model_id = "Qwen/Qwen2.5-3B"
pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

# MongoDB Local Connection
MONGO_URI = "mongodb://localhost:27017"
client = MongoClient(MONGO_URI)
db = client["task_demo"]
tasks_collection = db["tasks"]
users_collection = db["users"]

st.title("MongoDB AI Agent with LLaMA")
st.write("Ask anything and get an answer")
input_text = st.text_area("Enter your question here")

# Load sample file
with io.open("sample-text.txt", "r", encoding="utf-8") as f1:
    sample = f1.read()

def clean_json_response(response_text):
    response_text = response_text.strip()
    response_text = re.sub(r"```json|```", "", response_text)  # Remove any markdown code block markers
    response_text = response_text.strip("`")  # Remove accidental surrounding backticks
    return response_text

# Prompt Template for Query Generation
query_prompt = """
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

    Output **only** the JSON query and nothing else.

    sample_question: {sample}
    As an expert, use them as needed.

    **Most Importsant:** Do not return with any additional text like ```json, json, etc, just return MongoDB json query only.

    **Input:** {question}
    **Output JSON only:**
"""


# Prompt Template for Human-Readable Response
response_prompt = """
Convert the MongoDB query results into a natural language response.
Example:
Question: {question}
Query Results: {query_results}
Response:
"""

def count_tokens(text, model_name="Qwen/Qwen2.5-3B"):
    encoding = tiktoken.encoding_for_model(model_name)
    tokens = encoding.encode(text)
    return len(tokens)

def generate_response(prompt_text):
    terminators = [pipeline.tokenizer.eos_token_id]
    if "<|eot_id|>" in pipeline.tokenizer.get_vocab():
        terminators.append(pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>"))

    outputs = pipeline(
        prompt_text,
        max_new_tokens=256,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )
    return outputs[0]["generated_text"]

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
    try:
        query = convert_dates(query)
        query = convert_to_objectid(query)

        if "aggregate" in query:
            collection = db[query["collection"]]
            results = collection.aggregate(query["aggregate"])
        elif "filter" in query:
            collection = db[query["collection"]]
            results = collection.find(query["filter"], query.get("projection", {}))
        else:
            raise ValueError("Query must contain either 'aggregate' or 'filter' field")

        return list(results)
    except Exception as e:
        logging.error(f"Error executing query: {str(e)}")
        return [f"Error: {str(e)}"]

if input_text is not None:
    button = st.button("Submit")
    if button:
        start_time = time.time()
        
        question = input_text
        query_prompt_text = query_prompt.format(question=question, sample=sample)
        # query_response = generate_response(query_prompt_text)
        query_response = generate_response(query_prompt_text)
        cleaned_response = clean_json_response(query_response)
        query_generation_time = time.time() - start_time

        print("*****", query_prompt_text)

        print("#####", query_response)

        print("@@@@@", cleaned_response)
        
        try:
            query = json.loads(query_response)
        except json.JSONDecodeError as e:
            st.error(f"Query Parsing Error: {e}")
            query = None
        
        if query:
            st.write(f"Time taken to generate query: {query_generation_time:.2f} seconds")
            st.subheader("Generated MongoDB Query:")
            st.text(json.dumps(query, indent=4))

            start_time = time.time()
            query_results = execute_mongo_query(query)
            query_results_time = time.time() - start_time
            st.write(f"Time taken to fetch results: {query_results_time:.2f} seconds")
            
            st.subheader("Query Results:")
            for row in query_results:
                st.write(row)
            
            response_prompt_text = response_prompt.format(question=question, query_results=query_results)
            human_readable_output = generate_response(response_prompt_text)
            
            st.subheader("Final result:")
            st.write(human_readable_output)
        else:
            st.error("Failed to generate a valid MongoDB query.")