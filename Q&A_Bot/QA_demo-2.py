import pyodbc
import requests
from flask import Flask, request, jsonify
import dotenv
import os
import logging

dotenv.load_dotenv()

# Configure Logging
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)

# SQL Server Database Connection
DB_CONNECTION_STRING = (
    "DRIVER={ODBC Driver 17 for SQL Server};"
    "SERVER=DESKTOP-EM3I6GO;"  # Change this if needed
    "DATABASE=SQL_SERVER_DB;"  # Replace with actual DB name
    "Trusted_Connection=yes;"
)


# Function to fetch tasks from SQL Server
def get_tasks():
    try:
        logging.debug("🔹 Connecting to SQL Server...")
        conn = pyodbc.connect(DB_CONNECTION_STRING)
        logging.debug("✅ SQL Server Connection Successful!")

        cursor = conn.cursor()
        logging.debug("🔹 Fetching tasks from database...")

        cursor.execute("SELECT id, user_name, task_name, status FROM tasks")
        tasks = cursor.fetchall()
        conn.close()

        if not tasks:
            logging.debug("⚠️ No tasks found in the database.")
            return "No tasks found."

        task_list = "\n".join([f"{t[0]}. {t[1]} - {t[2]}" for t in tasks])
        logging.debug(f"✅ Retrieved Tasks:\n{task_list}")
        return task_list

    except Exception as e:
        logging.error(f"❌ Error fetching tasks: {str(e)}")
        return f"Error fetching tasks: {str(e)}"


# Function to send a query to Groq AI
def ask_groq(query):
    try:
        logging.debug("🔹 Connecting to Groq API...")
        headers = {
            "Authorization": f"Bearer {os.environ.get('GROQ_API_KEY')}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "llama-3.3-70b-versatile",
            "messages": [{"role": "user", "content": query}]
        }

        response = requests.post("https://api.groq.com/v1/chat/completions", json=data, headers=headers)
        response_json = response.json()
        logging.debug(f"✅ Groq API Response: {response_json}")

        return response_json["choices"][0]["message"]["content"]

    except Exception as e:
        logging.error(f"❌ Error connecting to Groq API: {str(e)}")
        return f"Error connecting to Groq API: {str(e)}"


@app.route("/chat", methods=["POST"])
def chat():
    try:
        logging.debug("🔹 Received chat request")
        # user_message = request.json.get("message", "").lower()
        user_message = request.json["message"].lower()
        logging.debug(f"📩 User Message: {user_message}")

        if "list my tasks" in user_message:
            logging.debug("🔍 User requested task list.")
            task_list = get_tasks()
            response_text = f"Here are your tasks:\n{task_list}"
        else:
            logging.debug("🔍 User query sent to Groq.")
            response_text = ask_groq(user_message)

        logging.debug(f"📤 Response: {response_text}")
        return jsonify({"response": response_text})

    except Exception as e:
        logging.error(f"❌ Error in /chat endpoint: {str(e)}")
        return jsonify({"response": f"Error: {str(e)}"})


if __name__ == "__main__":
    logging.debug("🚀 Starting Flask Server...")
    app.run(debug=True)




