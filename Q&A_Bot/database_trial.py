# import pyodbc
#
# DB_CONNECTION_STRING = (
#     "DRIVER={ODBC Driver 17 for SQL Server};"
#     "SERVER=DESKTOP-EM3I6GO;"  # Change this if needed
#     "DATABASE=QAagent;"  # Replace with actual DB name
#     "Trusted_Connection=yes;"
# )
#
# try:
#     conn = pyodbc.connect(DB_CONNECTION_STRING)
#     cursor = conn.cursor()
#     cursor.execute("SELECT name FROM sys.databases")
#
#     print("✅ Connection Successful! Databases available:")
#     for row in cursor.fetchall():
#         print(row[0])
#
#     conn.close()
# except Exception as e:
#     print(f"❌ Error connecting to SQL Server: {str(e)}")








# from dotenv import load_dotenv
# from groq import Groq
# import os
#
# load_dotenv()  # Load environment variables from a .env file
#
# client = Groq(
#     api_key=os.getenv('GROQ_API_KEY'),  # Get the API key from environment variables
# )
#
# chat_completion = client.chat.completions.create(
#     messages=[
#         {
#             "role": "user",
#             "content": "WAP to generate a star with triangle ",
#         }
#     ],
#     model="llama3-70b-8192",
# )
#
# print(chat_completion.choices[0].message.content)



import json
from pymongo import MongoClient

# MongoDB Connection
MONGO_URI = "mongodb://localhost:27017"
client = MongoClient(MONGO_URI)
db = client["task_demo"]  # Database Name
tasks_collection = db["tasks"]  # Collection Name

# Function to Execute MongoDB Query
def read_mongo_query(query_str):
    try:
        # Ensure the AI-generated string is properly formatted as JSON
        parsed_query = json.loads(query_str)

        # Validate JSON structure
        if not isinstance(parsed_query, dict) or "filter" not in parsed_query:
            return ["Error: Generated query is not in the expected format"]

        # Extract query parameters
        filter_query = parsed_query.get("filter", {})
        projection = parsed_query.get("projection", {})

        # Execute MongoDB query
        results = list(tasks_collection.find(filter_query, projection))

        return results if results else ["No matching data found."]

    except json.JSONDecodeError as e:
        return [f"Database Error: Invalid JSON format - {str(e)}"]
    except Exception as e:
        return [f"Database Error: {str(e)}"]

# Example Usage
query = '{"filter": {"AssigneeUserId": "6571e715e5d066d39dd01b39"}, "projection": {"TaskName": 1, "_id": 0}}'
print(read_mongo_query(query))  # Run the query execution function


