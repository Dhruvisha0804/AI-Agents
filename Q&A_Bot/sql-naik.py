from dotenv import load_dotenv

load_dotenv()  # Load environment variables

import streamlit as st
import os
import pyodbc  # SQL Server connection
import requests
import dotenv

dotenv.load_dotenv()

# Groq API Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# SQL Server Connection String
DB_CONNECTION_STRING = (
    "DRIVER={ODBC Driver 17 for SQL Server};"
    "SERVER=DESKTOP-EM3I6GO;"  # Change this if needed
    "DATABASE=QAagent;"  # Replace with actual DB name
    "Trusted_Connection=yes;"
)


# Function to send query to Groq Llama model
def get_groq_response(question, prompt):
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

    response = requests.post(GROQ_API_URL, json=data, headers=headers)

    if response.status_code == 200:
        result = response.json()
        return result["choices"][0]["message"]["content"].strip()
    else:
        return f"Error: {response.json()}"


# Function to execute query on SQL Server
def read_sql_query(sql):
    try:
        conn = pyodbc.connect(DB_CONNECTION_STRING)
        cursor = conn.cursor()
        cursor.execute(sql)

        # Fetch results if SELECT query
        if sql.strip().lower().startswith("select"):
            rows = cursor.fetchall()
        else:
            conn.commit()
            rows = ["Query executed successfully."]

        conn.close()
        return rows

    except Exception as e:
        return [f"Database Error: {str(e)}"]


# Define Your Prompt
prompt = [
    """
    You are an expert in converting English questions to SQL queries!
    The SQL server database is named 'QAagent' and it contains a table 'tasks' with columns like 'id', 'user_name', 'task_name', 'task_created_date', 'work_hours', 'status'.
    Make sure query format must support MS SQL server query format.
    
    Examples:
    - "How many tasks are there?" → SQL: SELECT COUNT(*) FROM tasks;
    - "Show all tasks with status 'Completed'" → SQL: SELECT * FROM tasks WHERE status = 'Completed';

    The response should ONLY contain the SQL query, without '```' or the word 'sql'.
    """
]


# Streamlit App
st.set_page_config(page_title="SQL Agent with Groq")
st.header("Groq AI SQL Query Generator for SQL Server")

question = st.text_input("Ask your SQL question: ", key="input")
submit = st.button("Generate Query & Fetch Data")

# If submit is clicked
if submit:
    generated_sql = get_groq_response(question, prompt)
    st.subheader("Generated SQL Query:")
    st.code(generated_sql, language="sql")

    print(f"Generated SQL Query: {generated_sql}")  # Log the generated SQL query

    # Execute the SQL query
    query_results = read_sql_query(generated_sql)

    st.subheader("Query Results:")
    for row in query_results:
        st.write(row)
