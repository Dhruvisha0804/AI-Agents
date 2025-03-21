import os
import logging
import json
import io
from dotenv import load_dotenv
from pymongo import MongoClient
from langchain.tools import StructuredTool
from langchain.agents import initialize_agent, AgentType
from langchain_groq import ChatGroq
from langchain import hub
import streamlit as st
import httpx
import time

MAX_RETRIES = 5
RETRY_DELAY = 5  # seconds

# Load environment variables
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama-3.3-70b-versatile"  # Updated model name

# Load sample queries
with io.open("sample.txt", "r", encoding="utf-8") as f1:
    sample = f1.read()

# Configure logging
logging.basicConfig(level=logging.DEBUG)

if not GROQ_API_KEY:
    logging.error("GROQ_API_KEY is missing. Groq functionality will be disabled.")
    raise Exception("Missing Groq API Key. Set GROQ_API_KEY in environment variables.")

# MongoDB Local Connection
MONGO_URI = "mongodb://localhost:27017"
client = MongoClient(MONGO_URI)
db = client["task_demo"]

def perform_extraction(query_str, collection_name):
    """Extract data from MongoDB collection."""
    try:
        if not collection_name:
            raise ValueError("collection_name is required")
        
        query = json.loads(query_str)  # Deserialize the query
        collection = db[collection_name]
        documents = collection.find(query)
        return json.dumps(list(documents), default=str)  # Serialize to JSON
    except Exception as e:
        logging.error(f"Error in perform_extraction: {e}")
        return f"Error: {e}"

def retrieve_schema(database_name):
    """Retrieve the schema of the MongoDB database."""
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
    
def fetch_with_retry():
    retries = 0
    while retries < MAX_RETRIES:
        try:
            response = llm()  # Replace with your API call function
            return response
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:  # Too Many Requests
                retries += 1
                wait_time = RETRY_DELAY * (2 ** retries)  # Exponential backoff
                logging.warning(f"Rate limit exceeded, retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                raise e  # Re-raise if it's not a rate limit error
    raise Exception("Max retries exceeded, rate limit still in effect.")


# Initialize the ChatGroq model
llm = ChatGroq(temperature=0, model_name=GROQ_MODEL, groq_api_key=GROQ_API_KEY)

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
    raise

# Streamlit app
st.title("LangChain Groq MongoDB Read Agent")

user_query = st.text_input("Enter your query:")

# In the main section, update the response to ensure collection_name is passed correctly
if st.button("Submit"):
    if user_query:
        try:
            schema = retrieve_schema(db.name)
            logging.debug(f"Schema sent to agent: {schema}")
            
            # Construct the query prompt for the agent
            prompt_input = f"""
            You are an expert in converting English questions into MongoDB queries!
            The database is named 'task_demo' and contains two collections: 'tasks' and 'users', with nested data structures.
            The 'tasks' collection includes fields like AssigneeUserId, TaskName, Status, ProjectID, Task_Priority, createdAt, etc.
            The 'users' collection has fields like _id, Employee_FName, Employee_LName, Employee_Name, createdAt, etc.
            The 'projects' collection contains _id, AssigneeUserid, LeaderUserId, ProjectCategory, ProjectName, etc.
            
            **Important Rule**: Always use `status.text` for task status filtering (e.g., `status.text: "Done"`).
            **Additional Rule**: Filter tasks using `sprintArray.folderName` (e.g., `sprintArray.folderName: "Development"`).
            
            **Relationship**:
            - A task may have multiple assignees (tasks.AssigneeUserId and users._id) and belongs to a project (tasks.ProjectID).
            - Projects can have multiple assignees.

            Below are sample user questions and MongoDB aggregation pipelines:

            sample_question: {sample}

            As an expert, use these samples as guidelines. Don't return extra details, only the MongoDB query.

            input:{user_query}
            output:
            Please return a valid MongoDB aggregation query based on the user's input.
            """
            
            # Run the agent and retrieve the query
            response = agent_executor.run(prompt_input)
            
            # Ensure the response contains the correct structure for collection name
            action_input = json.loads(response.get('action_input', '{}'))
            collection_name = action_input.get('collection_name')

            if collection_name:
                # Proceed with the extraction
                query_str = action_input.get('query_str', '{}')
                result = perform_extraction(query_str, collection_name)
                st.write("MongoDB Query Result:", result)
            else:
                st.error("Invalid action: Missing collection name.")
            
        except Exception as e:
            st.error(f"Error occurred while processing your query: {e}")
    else:
        st.warning("Please enter a query.")
