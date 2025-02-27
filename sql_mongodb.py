# from dotenv import load_dotenv
# import streamlit as st
# import os
# import requests
# import json
# from pymongo import MongoClient

# # Load environment variables
# load_dotenv()

# # Groq API Configuration
# GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# GROQ_MODEL = "llama-3.3-70b-versatile"
# GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# # MongoDB Connection
# MONGO_URI = "mongodb://localhost:27017"  # Update if using a remote MongoDB instance
# client = MongoClient(MONGO_URI)
# db = client["task_demo"]  # Database name


# # Function to send query to Groq Llama model
# def get_groq_response(question, prompt):
#     headers = {
#         "Authorization": f"Bearer {GROQ_API_KEY}",
#         "Content-Type": "application/json"
#     }

#     data = {
#         "model": GROQ_MODEL,
#         "messages": [
#             {"role": "system", "content": prompt[0]},
#             {"role": "user", "content": question}
#         ]
#     }

#     response = requests.post(GROQ_API_URL, json=data, headers=headers)

#     if response.status_code == 200:
#         result = response.json()
#         return result["choices"][0]["message"]["content"].strip()
#     else:
#         return f"Error: {response.json()}"


# # Function to execute query on MongoDB
# def execute_mongo_query(query):
#     try:
#         if "count_documents" in query:
#             count = db[query["collection"]].count_documents(query["filter"])
#             return [f"Count: {count}"]
#         elif "aggregate" in query:
#             results = list(db[query["collection"]].aggregate(query["aggregate"]))
#         else:
#             results = list(db[query["collection"]].find(query["filter"], query.get("projection", {})))

#         return results if results else ["No records found."]
#     except Exception as e:
#         return [f"Database Error: {str(e)}"]


# # Define Your Prompt for MongoDB
# prompt = [
#     """
#     You are an expert in converting English questions into MongoDB queries!
#     The database is named 'task_demo' and contains two collections: 'tasks' and 'users'.
#     The 'tasks' collection has fields: AssigneUserid, TaskName, Status, ProjectID, Task_Priority, CreatedAt etc.
#     The 'users' collection has fields: _id, Employee_FName, Employee_LName, Employee_Name, createdAt etc.

#     Generate **valid JSON-based MongoDB queries** to handle various database-related requests.
#     Ensure that queries are formatted correctly without additional text or explanations.

#     Examples:
#     - "How many tasks exist?" → { "collection": "tasks", "filter": {}, "count_documents": true }
#     - "Show all tasks assigned to '6571e715e5d066d39dd01b39'" → { "collection": "tasks", "filter": { "AssigneeUserId": "6571e715e5d066d39dd01b39" }, "projection": { "TaskName": 1, "_id": 0 } }
#     - "List users who have tasks in progress" → { "collection": "tasks", "filter": { "Status": "inprogress" }, "projection": { "AssigneeUserId": 1, "_id": 0 } }

#     **Return only valid JSON queries without extra formatting.**
#     """
# ]

# # Streamlit App
# st.set_page_config(page_title="MongoDB AI Agent with Groq")
# st.header("Groq AI MongoDB Query Generator")

# question = st.text_input("Ask your database question: ", key="input")
# submit = st.button("Generate Query & Fetch Data")

# # If submit is clicked
# if submit:
#     generated_query = get_groq_response(question, prompt)
#     st.subheader("Generated MongoDB Query:")
#     st.code(generated_query, language="json")

#     try:
#         mongo_query = json.loads(generated_query)  # Convert string to dict
#         query_results = execute_mongo_query(mongo_query)
#     except Exception as e:
#         query_results = [f"Query Parsing Error: {str(e)}"]

#     st.subheader("Query Results:")
#     for row in query_results:
#         st.write(row)




from dotenv import load_dotenv
import streamlit as st
import os
import requests
import json
import logging
from pymongo import MongoClient

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

# Groq API Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

if not GROQ_API_KEY:
    logging.error("GROQ_API_KEY is missing. Please set it in the environment variables.")

# MongoDB Connection
MONGO_URI = "mongodb://localhost:27017"  # Update if using a remote MongoDB instance
try:
    client = MongoClient(MONGO_URI)
    db = client["task_demo"]  # Database name
    logging.info("Connected to MongoDB successfully.")
except Exception as e:
    logging.error(f"MongoDB Connection Error: {e}")


# Function to send query to Groq Llama model
def get_groq_response(question, prompt):
    try:
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }

        data = {
            "model": GROQ_MODEL,
            "messages": [
                {"role": "system", "content": prompt[0]},
                {"role": "user", "content": question}
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


def execute_mongo_query(query):
    try:
        logging.info(f"Executing MongoDB query: {query}")
        
        # Handling count_documents queries
        if "count_documents" in query:
            count = db[query["collection"]].count_documents(query["filter"])
            return [f"Count: {count}"]

        # Handling aggregation queries (Fix for 'pipeline' issue)
        elif "pipeline" in query:
            results = list(db[query["collection"]].aggregate(query["pipeline"]))
        
        # Handling standard find queries
        else:
            results = list(db[query["collection"]].find(query["filter"], query.get("projection", {})))
        
        logging.info("MongoDB query executed successfully.")
        return results if results else ["No records found."]

    except Exception as e:
        logging.error(f"Database Error: {e}")
        return [f"Database Error: {str(e)}"]



# Define Your Prompt for MongoDB
# prompt = [
#     """
#     You are an expert in converting English questions into MongoDB queries!
#     The database is named 'task_demo' and contains two collections: 'tasks' and 'users'.
#     The 'tasks' collection has fields: AssigneUserid, TaskName, Status, ProjectID, Task_Priority, CreatedAt etc.
#     The 'users' collection has fields: _id, Employee_FName, Employee_LName, Employee_Name, createdAt etc.

#     Generate **valid JSON-based MongoDB queries** to handle various database-related requests.
#     Ensure that queries are formatted correctly without additional text or explanations.

#     Examples:
#     - "How many tasks exist?" → { "collection": "tasks", "filter": {}, "count_documents": true }
#     - "Show all tasks assigned to '6571e715e5d066d39dd01b39'" → { "collection": "tasks", "filter": { "AssigneeUserId": "6571e715e5d066d39dd01b39" }, "projection": { "TaskName": 1, "_id": 0 } }
#     - "List users who have tasks in progress" → { "collection": "tasks", "filter": { "Status": "inprogress" }, "projection": { "AssigneeUserId": 1, "_id": 0 } }

#     **Return only valid JSON queries without extra formatting.**
#     """
# ]

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
st.set_page_config(page_title="MongoDB AI Agent with Groq")
st.header("Groq AI MongoDB Query Generator")

question = st.text_input("Ask your database question: ", key="input")
submit = st.button("Generate Query & Fetch Data")

# If submit is clicked
if submit:
    logging.info(f"Received question: {question}")
    generated_query = get_groq_response(question, prompt)
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
