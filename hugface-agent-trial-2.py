import os
from dotenv import load_dotenv
from pymongo import MongoClient
import logging
from langchain.tools import StructuredTool
from langchain.agents import initialize_agent, AgentType
from langchain import hub
import streamlit as st
import json
from langchain_community.llms import HuggingFaceHub

# Load environment variables
load_dotenv()
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Configure logging
logging.basicConfig(level=logging.DEBUG)

if not HUGGINGFACEHUB_API_TOKEN:
    logging.error("Missing HUGGINGFACEHUB_API_TOKEN. Set HUGGINGFACEHUB_API_TOKEN in environment variables.")
    raise Exception("Missing HUGGINGFACEHUB_API_TOKEN. Set HUGGINGFACEHUB_API_TOKEN in environment variables.")

# MongoDB Local Connection
MONGO_URI = "mongodb://localhost:27017"
client = MongoClient(MONGO_URI)
db = client["task_demo"]

def perform_extraction(query_str, collection_name):
    """Extract data from MongoDB collection."""
    try:
        query = json.loads(query_str)
        collection = db[collection_name]
        documents = collection.find(query)
        return json.dumps(list(documents), default=str)  # Serialize to JSON
    except Exception as e:
        logging.error(f"Error in perform_extraction: {e}")
        return f"Error: {e}"

def retrieve_schema(database_name):
    try:
        db = client[database_name]
        collection_names = db.list_collection_names()
        all_schemas = {}
        for collection_name in collection_names:
            collection = db[collection_name]
            document = collection.find_one()
            schema = {}
            if document:
                for key, value in document.items():
                    schema[key] = type(value).__name__
            all_schemas[collection_name] = schema
        return json.dumps(all_schemas)
    except Exception as e:
        logging.error(f"Error in retrieve_schema: {e}")
        return json.dumps({})

# Initializing the HuggingFaceHub model
model_id = "google/flan-t5-small"
llm = HuggingFaceHub(repo_id=model_id, huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN)

# Tools for the agent
tool_extract = StructuredTool.from_function(
    perform_extraction,
    description="Extract documents from a specified MongoDB collection based on a query. Input must be a valid JSON string representing the query, and the collection name."
)
tool_schema = StructuredTool.from_function(
    retrieve_schema,
    description="Retrieve the schema of a specified MongoDB database. Input is the database name."
)

# Initialize the agent
try:
    prompt = hub.pull("hwchase17/react")
    agent_executor = initialize_agent(
        [tool_extract, tool_schema],
        llm,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        agent_kwargs={"prompt": prompt}
    )
    logging.debug("Agent initialized successfully.")
except Exception as e:
    logging.error(f"Error initializing agent: {e}")

# Streamlit app
st.title("LangChain HuggingFace MongoDB Read Agent")

user_query = st.text_input("Enter your query:")

if st.button("Submit"):
    if user_query:
        try:
            schema = retrieve_schema(db.name)
            response = agent_executor.run(f"Given the database schema {schema}, your task is to perform read operations on a NoSQL database. Before proceeding with the query '{user_query}', consider the data size involved. If user asks for schema, respond with the schema. If user asks to extract data, convert the user's question into a valid JSON query, and use the tools provided. If user asks a question that is not related to the database, respond with a message that you can only answer database related questions.")
            st.write("Response:", response)
        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.warning("Please enter a query.")