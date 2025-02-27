from dotenv import load_dotenv
import streamlit as st
import os
import requests
import json
import logging
import openai
from pymongo import MongoClient

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

# OpenAI API Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    logging.error("OPENAI_API_KEY is missing. Please set it in the environment variables.")

# MongoDB Connection
MONGO_URI = "mongodb://localhost:27017"  # Update if using a remote MongoDB instance
try:
    client = MongoClient(MONGO_URI)
    db = client["task_demo"]  # Database name
    logging.info("Connected to MongoDB successfully.")
except Exception as e:
    logging.error(f"MongoDB Connection Error: {e}")


# Function to send query to OpenAI's GPT model
def get_openai_response(question, prompt):
    try:
        logging.info("Sending request to OpenAI API...")

        openai.api_key = OPENAI_API_KEY  # Set API key

        response = openai.ChatCompletion.create(
            model="gpt-4-turbo",  # Use GPT-4 turbo or "gpt-3.5-turbo" if you prefer
            messages=[
                {"role": "system", "content": prompt[0]},
                {"role": "user", "content": question}
            ]
        )

        logging.info("Received response from OpenAI API successfully.")
        return response["choices"][0]["message"]["content"].strip()
    
    except Exception as e:
        logging.error(f"Error in OpenAI API request: {e}")
        return f"API Error: {str(e)}"


# Function to execute MongoDB query
def execute_mongo_query(query):
    try:
        logging.info(f"Executing MongoDB query: {query}")

        if "count_documents" in query:
            count = db[query["collection"]].count_documents(query["filter"])
            return [f"Count: {count}"]
        elif "pipeline" in query:
            results = list(db[query["collection"]].aggregate(query["pipeline"]))
        else:
            results = list(db[query["collection"]].find(query["filter"], query.get("projection", {})))

        logging.info("MongoDB query executed successfully.")
        return results if results else ["No records found."]

    except Exception as e:
        logging.error(f"Database Error: {e}")
        return [f"Database Error: {str(e)}"]


# Define System Prompt for MongoDB Query Generation
prompt = [
    """
    You are an expert in converting English questions into MongoDB queries!
    The database is named 'task_demo' and contains two collections: 'tasks' and 'users'.
    The 'tasks' collection has fields: AssigneeUserId, TaskName, Status (embedded object with text, key, and type), ProjectID, Task_Priority, CreatedAt, etc.
    The 'users' collection has fields: _id, Employee_FName, Employee_LName, Employee_Name, createdAt, etc.
    In both collections, the '_id' from 'users' and 'AssigneeUserId' from 'tasks' are the same, and they need to be joined for certain queries.

    **Important Rule**: Always reference the `status.text` field when filtering by task status (e.g., `status.text: "Done"`).
    
    **Additional Rule**: 
    - When you need to filter tasks or query simple fields, you can use the `filter` field.
    - If the query includes dates, use the **"$date"** format (e.g., `{"$date": "YYYY-MM-DDTHH:MM:SS.sssZ"}`) to represent date fields in queries.
    - If the query requires aggregation, **do not include a `filter` field**. Instead, only include the aggregation pipeline. For example:

    - "How many tasks are there by each different status?" → 
    {
        "collection": "tasks", 
        "aggregate": [
            { "$group": { "_id": "$status.text", "count": { "$sum": 1 } } }
        ]
    }

    - "How many tasks exist?" → { "collection": "tasks", "filter": {}, "count_documents": true }
    
    - "List users who have task status 'done'" → {
        "collection": "tasks", 
        "filter": { "status.text": "Done" }, 
        "lookup": {
            "from": "users", 
            "localField": "AssigneeUserId", 
            "foreignField": "_id", 
            "as": "user_info"
        },
        "projection": { "user_info.Employee_FName": 1, "user_info.Employee_LName": 1, "_id": 0 }
    }

    - "Show all tasks assigned to '6571e715e5d066d39dd01b39'" → 
    { "collection": "tasks", "filter": { 
        "AssigneeUserId": "6571e715e5d066d39dd01b39" }, 
        "projection": { "TaskName": 1, "_id": 0 } 
    }

    - "List all tasks for employee 'Shiv Joshi'" → {
        "collection": "tasks", 
        "aggregate": [
            {
                "$lookup": {
                    "from": "users",
                    "localField": "AssigneeUserId",
                    "foreignField": "_id",
                    "as": "user_info"
                }
            },
            {
                "$unwind": "$user_info"
            },
            {
                "$match": {
                    "user_info.Employee_Name": "Shiv Joshi"
                }
            },
            {
                "$project": {
                    "TaskName": 1,
                    "_id": 0
                }
            }
        ]
    }

    - "List tasks created after a specific day" → 
        {
            "collection": "tasks",
            "filter": { "createdAt": { "$gt": ISODate("2025-02-20T00:00:00.000Z") } },
            "projection": { "TaskName": 1, "_id": 0 }
        }



    **Return only valid JSON queries without extra formatting.**
    """
]


# Streamlit App
st.set_page_config(page_title="MongoDB AI Agent with OpenAI")
st.header("OpenAI MongoDB Query Generator")

question = st.text_input("Ask your database question: ", key="input")
submit = st.button("Generate Query & Fetch Data")

# If submit is clicked
if submit:
    logging.info(f"Received question: {question}")
    generated_query = get_openai_response(question, prompt)
    st.subheader("Generated MongoDB Query:")
    st.code(generated_query, language="json")

    try:
        mongo_query = json.loads(generated_query)  # Convert string to dict
        query_results = execute_mongo_query(mongo_query)
    except json.JSONDecodeError as e:
        logging.error(f"Query Parsing Error: {e}")
        query_results = [f"Query Parsing Error: {str(e)}"]
    except Exception as e:
        logging.error(f"Unexpected Error: {e}")
        query_results = [f"Unexpected Error: {str(e)}"]

    st.subheader("Query Results:")
    for row in query_results:
        st.write(row)
