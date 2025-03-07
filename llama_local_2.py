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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load local LLaMA model
model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
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
with io.open("sample.txt", "r", encoding="utf-8") as f1:
    sample = f1.read()

# Prompt Template for Query Generation
query_prompt = """
You are an expert in converting English questions into MongoDB queries!
Database: 'task_demo'
Collections: 'tasks', 'users', 'projects'
Ensure correct field references and return only the query.
Sample questions & queries:
{sample}

User Question: {question}
MongoDB Query:
"""

# Prompt Template for Human-Readable Response
response_prompt = """
Convert the MongoDB query results into a natural language response.
Example:
Question: {question}
Query Results: {query_results}
Response:
"""

def count_tokens(text, model_name="meta-llama/Meta-Llama-3-8B-Instruct"):
    encoding = tiktoken.encoding_for_model(model_name)
    tokens = encoding.encode(text)
    return len(tokens)

def generate_response(prompt_text):
    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
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
        query_response = generate_response(query_prompt_text)
        query_generation_time = time.time() - start_time
        
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