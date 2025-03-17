import json
import numpy as np
import faiss
import torch
import pymongo
import logging
import streamlit as st
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import re

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Model loading function
@st.cache_resource
def load_llama_model():
    try:
        MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
        
        # Initialize pipeline with device map set to 'auto'
        pipe = pipeline(
            "text-generation",
            model=MODEL_ID,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        logging.debug("Llama model loaded successfully.")
        return pipe
    
    except Exception as e:
        logging.error(f"Error loading Llama model: {e}")
        st.error("Failed to load Llama model. Check the model name and ensure it is downloaded.")
        return None


@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


pipe = load_llama_model()
embedding_model = load_embedding_model()
logging.debug("Embedding model loaded.")

# Load FAISS index
index = faiss.IndexFlatL2(384)  # 384 is the embedding size for 'all-MiniLM-L6-v2'

# Load MongoDB
client = pymongo.MongoClient("mongodb://localhost:27017")
db = client["task_demo"]

def load_sample_queries():
    """Load sample queries from file."""
    try:
        with open("sample.txt", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        logging.warning("sample.txt not found. Using empty context.")
        return ""

def get_most_similar_question(question):
    """Find the most similar question from FAISS index."""
    vector = np.array(embedding_model.encode([question]), dtype=np.float32)
    if index.ntotal > 0:  # Check if FAISS has stored vectors
        distances, indices = index.search(vector, 1)
        if distances[0][0] < 0.7:  # Adjust similarity threshold
            return indices[0][0]
    return None

def generate_query(question, sample_context):
    """Generate MongoDB query using Llama model."""

    if not pipe:
        logging.error("Model pipeline is not initialized.")
        return None
    
    similar_index = get_most_similar_question(question)
    
    # Retrieve most similar query (if exists)
    similar_query = "" if similar_index is None else sample_context.split('\n')[similar_index]
    
    # Prepare modified prompt
    # modified_prompt = f"""
    # You are an expert in converting English questions into MongoDB queries for the 'task_demo' database. This database contains three collections: 'tasks', 'users', and 'projects', with nested and embedded data.

    # **Collections Structure:**
    # - 'tasks' collection: fields like AssigneeUserId, TaskName, Status (embedded with text, key, type), ProjectID, Task_Priority, createdAt, SprintArray, Task_Leader.
    # - 'users' collection: fields like _id, Employee_FName, Employee_LName, Employee_Name, createdAt.
    # - 'projects' collection: fields like _id, AssigneeUserid, LeaderUserId, ProjectCategory, ProjectName.

    # **Important Rules:**
    # 1. Always reference `status.text` for task status filtering (e.g., `status.text: "Done"`).
    # 2. Always use `sprintArray.folderName` for sprint folder filtering (e.g., `sprintArray.folderName: "Development"`).
    # 3. For simple field filtering, use `filter`. For aggregation, only include the aggregation pipeline—no `filter` field.

    # **Relationships:** Each task (tasks.AssigneeUserId) can have multiple assignees (users._id), and each task (tasks.ProjectID) belongs to one project (projects._id), with a project potentially having multiple assignees.

    # **Task:** Use the provided sample question and corresponding query as a reference. Only return the MongoDB query in json format, no additional details.

    # Sample Queries: {sample_context}

    # If the user's question matches an existing sample, use it. Otherwise, create a new query.

    # As an expert, use them as needed.

    # **Most Importsant:** Do not return with any additional text like ```json, json, etc, just return MongoDB json query only.

    # User Question: {question}
    # """

    modified_prompt = f"""
        You are an expert in converting English questions into MongoDB queries for the 'task_demo' database. This database contains three collections: 'tasks', 'users', and 'projects'. 

        **Important Rules:**
        - For task status, use `status.text` (e.g., `status.text: "Done"`).
        - For sprint folders, use `sprintArray.folderName` (e.g., `sprintArray.folderName: "Development"`).
        - Use `filter` for simple field filtering and the aggregation pipeline for aggregation queries (no `filter` field).
        - Each task may have multiple assignees (users._id) and belongs to one project (projects._id).

        **Context:**
        {f"Similar query found: {similar_query}" if similar_query else "No similar query found."}

        **Task:** Convert the following user question into a MongoDB query in JSON format. ONLY return the query, no additional text.

        **User Question:** {question}
        """

    # Get response from Llama model
    try:
        response = pipe(modified_prompt, max_new_tokens=256)
        response_text = response[0]['generated_text'].strip()
    except Exception as e:
        logging.error("Error generating query: %s", e)
        return None

    # Debug log model response
    logging.debug("Raw Model Response: %s", response_text)
    
    # Extract valid JSON using regex
    try:
        match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if match:
            extracted_json = match.group(0)
            
            # If multiple JSON objects are found, handle this case (e.g., choose the first one)
            if extracted_json.count('{') > 1:
                extracted_json = "{" + extracted_json.split('{', 1)[1]  # Remove extra parts if multiple JSON objects
                logging.warning("Multiple JSON objects found, selecting first valid one.")

            # Try parsing the extracted JSON
            return json.loads(extracted_json)
        else:
            logging.error("No valid JSON found in the response.")
            return None
    except json.JSONDecodeError as e:
        logging.error(f"JSON Parsing Error: {e}\nRaw Extracted JSON: {extracted_json if extracted_json else 'None'}")
        return None

def execute_mongo_query(query):
    """Execute the generated MongoDB query and return results."""
    try:
        collection_name = query.get("collection", "tasks")  # Default to 'tasks'
        collection = db[collection_name]
        
        if "aggregate" in query:
            results = list(collection.aggregate(query["aggregate"]))
        else:
            results = list(collection.find(query.get("filter", {})))
        
        return results
    except Exception as e:
        logging.exception("MongoDB Query Execution Failed")
        return None

# Streamlit UI
st.title("MongoDB Query Generator & Executor")

with st.form("query_form"):
    input_text = st.text_area("Enter your question here")
    submit_button = st.form_submit_button("Submit")

# Initialize generated_query before checking its value
generated_query = None

if submit_button:
    sample_context = load_sample_queries()
    if pipe:
        generated_query = generate_query(input_text, sample_context)
        logging.debug("Generated query: %s", generated_query)
    else:
        st.error("Failed to initialize model. Please check logs.")
    
    if generated_query:
        st.json(generated_query)  # Display generated query
        query_results = execute_mongo_query(generated_query)
        if query_results:
            st.json(query_results)  # Display query results
        else:
            st.error("Error executing MongoDB query.")
    else:
        st.error("Failed to generate MongoDB query.")
