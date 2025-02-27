import streamlit as st
import streamlit as st
from pymongo import MongoClient
import io, json, os
import faiss
import numpy as np
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from dotenv import load_dotenv
import logging
from datetime import datetime
from bson import ObjectId
from sentence_transformers import SentenceTransformer  # Importing SentenceTransformer
import requests

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

if not OPENAI_API_KEY:
    st.error("Missing OpenAI API Key. Set OPENAI_API_KEY in environment variables.")
    st.stop()

# Initialize OpenAI LLM & Embeddings (Note: We are not using OpenAI embeddings anymore)
llm = ChatOpenAI(model="gpt-4", temperature=0.0, openai_api_key=OPENAI_API_KEY)

# Initialize the SentenceTransformer model (local embeddings)
model = SentenceTransformer('all-MiniLM-L6-v2')  # Using a pre-trained model for embeddings

# MongoDB Local Connection
MONGO_URI = "mongodb://localhost:27017"
client = MongoClient(MONGO_URI)
db = client["task_demo"]
tasks_collection = db["tasks"]
users_collection = db["users"]

# Load sample-text.txt and preprocess
sample_text = io.open("sample-text.txt", "r", encoding="utf-8").read()
sample_lines = sample_text.strip().split("\n")
sample_questions = []
sample_queries = []

for i in range(0, len(sample_lines), 2):  # Assuming alternate lines contain question and query
    sample_questions.append(sample_lines[i])
    sample_queries.append(sample_lines[i + 1] if i + 1 < len(sample_lines) else "")

# Function to get embeddings using SentenceTransformer (local)
def get_embedding(text):
    return model.encode(text)  # Use SentenceTransformer to get embedding

# Generate embeddings and create FAISS index
embedding_dim = 384  # Dimensionality of the all-MiniLM-L6-v2 embeddings (384)
faiss_index = faiss.IndexFlatL2(embedding_dim)  # Using FAISS with L2 distance metric
embeddings = np.array([get_embedding(q) for q in sample_questions])

# Adding embeddings to the FAISS index
faiss_index.add(embeddings)

def get_relevant_examples(user_query, top_k=3):
    user_embedding = get_embedding(user_query).reshape(1, -1)
    distances, indices = faiss_index.search(user_embedding, top_k)
    relevant_samples = [sample_questions[i] + "\n" + sample_queries[i] for i in indices[0] if i < len(sample_queries)]
    return "\n".join(relevant_samples)

st.title("MongoDB AI Agent with LangChain-OpenAI & FAISS")
st.write("Ask anything and get an answer")
input_text = st.text_area("Enter your question here")

prompt_template = """
    You are an expert in converting English questions into MongoDB queries!
    The database is named 'task_demo' and contains two collections: 'tasks' and 'users'.
    including nested and embedded data structures that add depth and detail to the document.
    The 'tasks' collection has fields: AssigneeUserId, TaskName, Status (embedded object with text, key, and type), ProjectID, Task_Priority, createdAt, SprintArray, Task_Leader etc.
    The 'users' collection has fields: _id, Employee_FName, Employee_LName, Employee_Name, createdAt, etc.

    **Important Rule**: Always reference the `status.text` field when filtering by task status (e.g., `status.text: "Done"`).
    Always reference the `sprintArray.folderName` field when filtering by sprint folders (e.g., `sprintArray.folderName: "Development"`).
    
    **Additional Rule**: 
    - When you need to filter tasks or query simple fields, you can use the `filter` field.
    - If the query requires aggregation, **do not include a `filter` field**. Instead, only include the aggregation pipeline. 

    ***Relationship Between users and tasks***

    Each task can have one or more assignees, represented by AssigneeUserId in the tasks collection. This field stores an array of references to the _id of users in the users collection.
    Users can be assigned multiple tasks, as seen in the AssigneeUserId field of the tasks collection.

    Below are several sample user questions related to the MongoDB document provided, 
    and the corresponding MongoDB aggregation pipeline queries that can be used to fetch the desired data.
    Use them wisely.

    Note: You have to just return the query nothing else. Don't return any additional detail with the query.Please follow this strictly

    Relevant examples:
    {examples}

    Question: {question}
    MongoDB Query:

    """

query_with_prompt = PromptTemplate(template=prompt_template, input_variables=["question", "examples"])
llmchain = LLMChain(llm=llm, prompt=query_with_prompt, verbose=True)

def convert_dates(query):
    """Recursively converts '$date' strings into datetime objects in a given query."""
    if isinstance(query, dict):
        for key, value in query.items():
            if isinstance(value, dict) and "$date" in value:
                try:
                    query[key] = datetime.fromisoformat(value["$date"].replace("Z", "+00:00"))  # Convert to datetime
                except ValueError:
                    pass  # Ignore if conversion fails
            else:
                convert_dates(value)  # Recursively process nested dictionaries
    elif isinstance(query, list):
        for i in range(len(query)):
            convert_dates(query[i])  # Recursively process lists
    return query


def convert_to_objectid(query):
    """Recursively converts string representations of ObjectIds to actual ObjectId objects."""
    if isinstance(query, dict):
        for key, value in query.items():
            if isinstance(value, str) and len(value) == 24:  # Length of ObjectId string
                try:
                    # Convert to ObjectId if the field is _id or any other field
                    query[key] = ObjectId(value)
                except Exception as e:
                    pass  # Ignore if conversion fails
            elif isinstance(value, dict):
                convert_to_objectid(value)  # Recurse if it's a nested dictionary
    elif isinstance(query, list):
        for i in range(len(query)):
            convert_to_objectid(query[i])  # Recurse if it's a list of dictionaries
    return query

def execute_mongo_query(query):
    """Executes the MongoDB query safely."""
    try:
        query = convert_to_objectid(query)
        if "aggregate" in query:
            results = db[query["collection"]].aggregate(query["aggregate"])
        elif "filter" in query:
            results = db[query["collection"]].find(query["filter"], query.get("projection", {}))
        else:
            raise ValueError("Invalid query format")
        return list(results)
    except Exception as e:
        logging.error(f"Error executing query: {e}")
        return [f"Error: {str(e)}"]

def get_groq_response(question, query_results):
    """Converts JSON query results into human-readable text using Groq."""
    data = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {"role": "system", "content": "Translate MongoDB query results into human-readable language."},
            {"role": "user", "content": f"Question: {question}\nQuery Results: {query_results}"}
        ]
    }
    response = requests.post("https://api.groq.com/openai/v1/chat/completions", json=data,
                             headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"})
    return response.json().get("choices", [{}])[0].get("message", {}).get("content", "Error processing response")

# Retrieve relevant examples based on input text
def retrieve_relevant_examples(input_text):
    # Convert the input text into an embedding
    input_embedding = get_embedding(input_text).reshape(1, -1)  # Reshaping for FAISS query
    # Perform a search in the FAISS index
    distances, indices = faiss_index.search(input_embedding, k=3)  # Retrieving top 3 most relevant
    relevant_examples = []
    
    # Collect the relevant question-query pairs
    for idx in indices[0]:
        if idx >= 0:  # Ensuring valid index
            relevant_examples.append({
                "question": sample_questions[idx],
                "query": sample_queries[idx]
            })
    
    return relevant_examples


if input_text:
    button = st.button("Submit")
    if button:
        relevant_examples = retrieve_relevant_examples(input_text)
        print("******", relevant_examples)
        formatted_examples = "\n".join([f"Example Q: {ex['question']}\nExample Query: {ex['query']}" for ex in relevant_examples])
        response = llmchain.invoke({"question": input_text, "examples": formatted_examples})
        query = json.loads(response["text"])
        st.subheader("Generated MongoDB Query:")
        st.text(json.dumps(query, indent=4))
        query_results = execute_mongo_query(query)
        st.subheader("Query Results:")
        for row in query_results:
            st.write(row)
        human_readable_output = get_groq_response(input_text, query_results)
        st.subheader("Final Result:")
        st.write(human_readable_output)
