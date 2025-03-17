import os
import json
import logging
import numpy as np
from pymongo import MongoClient
from dotenv import load_dotenv
import time
from datetime import datetime
from bson import ObjectId

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

# MongoDB Local Connection
MONGO_URI = "mongodb://localhost:27017"
client = MongoClient(MONGO_URI)
db = client["task_demo"]
tasks_collection = db["tasks"]
users_collection = db["users"]
projects_collection = db["projects"]

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

def main():
    start_time = time.time()
    try:
        hardcoded_query = {
            "collection": "tasks",
            "aggregate": [
                {
                    "$match": {
                        "status.text": "To Do"
                    }
                },
                {
                    "$count": "total_tasks"
                }
            ]
        }
        print("Hardcoded MongoDB Query:")
        print(json.dumps(hardcoded_query, indent=4))

        start_time = time.time()
        query_results = execute_mongo_query(hardcoded_query)
        query_results_time = time.time() - start_time

        print("Query Results:")
        for row in query_results:
            print(row)

        print(f"Time taken to fetch results: {query_results_time:.2f} seconds")

        if not query_results:
            print("No query results returned from MongoDB.")
    except Exception as e:
        print(f"Error executing query: {e}")

if __name__ == "__main__":
    main()