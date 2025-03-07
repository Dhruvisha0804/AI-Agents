import os
import json
import logging
import torch
import streamlit as st
from dotenv import load_dotenv
from pymongo import MongoClient
from langchain.tools import StructuredTool
from langchain.agents import initialize_agent, AgentType
from langchain import hub
from langchain.llms.base import LLM
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from typing import List, Optional

# Load environment variables
load_dotenv()
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# MongoDB Local Connection
MONGO_URI = "mongodb://localhost:27017"
client = MongoClient(MONGO_URI)
db = client["task_demo"]

# Function to Extract Data from MongoDB
def perform_extraction(query_str, collection_name):
    try:
        query = json.loads(query_str)
        collection = db[collection_name]
        documents = list(collection.find(query))
        return json.dumps(documents, default=str)
    except Exception as e:
        logging.error(f"Error in perform_extraction: {e}")
        return json.dumps({"error": str(e)})

# Function to Retrieve MongoDB Schema
def retrieve_schema(database_name):
    try:
        db = client[database_name]
        collection_names = db.list_collection_names()
        all_schemas = {}
        for collection_name in collection_names:
            collection = db[collection_name]
            document = collection.find_one()
            schema = {key: type(value).__name__ for key, value in document.items()} if document else {}
            all_schemas[collection_name] = schema
        return json.dumps(all_schemas)
    except Exception as e:
        logging.error(f"Error in retrieve_schema: {e}")
        return json.dumps({"error": str(e)})

# Load Hugging Face Model Locally
model_id = "google/flan-t5-large"
try:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    logging.info(f"Loaded model {model_id} successfully.")
except Exception as e:
    logging.error(f"Error loading model {model_id}: {e}")
    raise

# Function to Run Local Model
def local_llm(prompt):
    try:
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=100)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        logging.error(f"Error in local_llm: {e}")
        return "Error processing request."

# Custom LLM Wrapper for LangChain
class CustomLLM(LLM):
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        return local_llm(prompt)

    @property
    def _identifying_params(self):
        return {}

    @property
    def _llm_type(self):
        return "custom"

# Create LangChain-compatible LLM
llm = CustomLLM()

# Define Tools for Agent
tool_extract = StructuredTool.from_function(
    perform_extraction,
    description="Extract documents from a MongoDB collection based on a JSON query."
)
tool_schema = StructuredTool.from_function(
    retrieve_schema,
    description="Retrieve the schema of a MongoDB database."
)

# Initialize Agent
try:
    prompt = hub.pull("hwchase17/react")
    agent_executor = initialize_agent(
        tools=[tool_extract, tool_schema],
        llm=llm,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        agent_kwargs={"prompt": prompt}
    )
    logging.info("Agent initialized successfully.")
except Exception as e:
    logging.error(f"Error initializing agent: {e}")
    raise

# Streamlit App
st.title("LangChain HuggingFace MongoDB Read Agent")

user_query = st.text_input("Enter your query:")

if st.button("Submit"):
    if user_query:
        try:
            sample_text = """
                Question 1. How many tasks are there by each different status?
                Query
                json
                {
                    "collection": "tasks", 
                    "aggregate": [
                        { "$group": { "_id": "$status.text", "count": { "$sum": 1 } } }
                    ]
                }

                Question 2. How many tasks exist?
                Query
                json
                {
                    "collection": "tasks",
                    "aggregate": [
                        { "$count": "total_tasks" }
                    ]
                }
            """

            system_prompt = (
                f"""
                You are an expert in converting English questions into MongoDB queries for the 'task_demo' database. 
                Use the following sample questions and their corresponding MongoDB queries as references:
                
                {sample_text}
                
                Follow these rules:
                1. Always reference `status.text` for task status filtering (e.g., `status.text: "Done"`).
                2. Always use `sprintArray.folderName` for sprint folder filtering (e.g., `sprintArray.folderName: "Development"`).
                3. For simple field filtering, use `filter`. For aggregation, only include the aggregation pipeline—no `filter` field.
                
                Given this information, convert the following user query into a MongoDB query:
                
                input: {user_query}
                output:
                """
            )
            response = agent_executor.run(system_prompt)
            st.write("Response:", response)
        except Exception as e:
            logging.error(f"Error processing query: {e}")
            st.error(f"Error: {e}")
    else:
        st.warning("Please enter a query.")