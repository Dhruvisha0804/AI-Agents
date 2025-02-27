import streamlit as st
from pymongo import MongoClient
import io, json, os
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationSummaryMemory
from dotenv import load_dotenv
import logging
from datetime import datetime
from bson import ObjectId
import time
import tiktoken
import requests

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    st.error("Missing OpenAI API Key. Set OPENAI_API_KEY in environment variables.")
    st.stop()

# Initialize memory
memory = ConversationSummaryMemory(llm=ChatOpenAI())

# Initialize OpenAI LLM
llm = ChatOpenAI(model="gpt-4", temperature=0.0, openai_api_key=OPENAI_API_KEY)

# MongoDB Local Connection
MONGO_URI = "mongodb://localhost:27017"
client = MongoClient(MONGO_URI)
db = client["task_demo"]
tasks_collection = db["tasks"]
users_collection = db["users"]

st.title("MongoDB AI Agent with LangChain-OpenAI")
st.write("Ask anything and get an answer")
input_text = st.text_area("Enter your question here")

# Initial system prompt (schema and instructions)
system_prompt = """
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

    sample_question: {sample}
    As an expert you must use them whenever required.
    Note: You have to just return the query nothing else. Don't return any additional detail with the query.Please follow this strictly.
    input:{question}
    output:

    Please return only the MongoDB query for the user's question. The output should be a valid aggregation pipeline query.
"""



# Load sample file
with io.open("sample.txt", "r", encoding="utf-8") as f1:
    sample = f1.read()

# Groq API Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

if not GROQ_API_KEY:
    logging.error("GROQ_API_KEY is missing. Please set it in the environment variables.")

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
    try:
        query = convert_dates(query)  # Convert date strings before execution

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


def count_tokens(text, model_name="gpt-3.5-turbo"): 
    encoding = tiktoken.encoding_for_model(model_name) 
    tokens = encoding.encode(text) 
    return len(tokens)

# Function to send query to Groq Llama model
def get_groq_response(question, query_results):
    try:
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }

        prompt_groq = """
        You are the translator for the json formatted results from the MongoDB database to human-like language.
        Example:
        question: How many distinct tasks are there?
        generated query: {
            "collection": "tasks",
            "aggregate": [
                {
                    "$count": "total_tasks"
                }
            ]
        }
        Query results: {
        "total_tasks": 66543
        }

        Now you have to convert the query results into human-readable sentences.

        Final result: There are a total of 66543 distinct tasks in the tasks collection.
        """

        # Adding the prompt, question, and query results to the request data
        data = {
            "model": GROQ_MODEL,
            "messages": [
                {"role": "system", "content": prompt_groq},
                {"role": "user", "content": question},
                {"role": "user", "content": f"Query results: {query_results}"}
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

# Prompt template and chain setup
prompt = PromptTemplate(
    input_variables=["question"],
    template="Question: {question}"
)

# LLM Chain with memory
llm_chain = LLMChain(llm=llm, prompt=prompt, memory=memory, verbose=True)


if input_text:
    button = st.button("Submit")
    if button:
        start_time = time.time()
        try:
            # First, send the system prompt to establish context
            if not memory.chat_memory.messages:
                llm_chain.predict(question=system_prompt)

            # Then, send the user's question
            prompt_with_instruction = f"""
                Always generate **only** a valid MongoDB query. Do **not** respond with explanations, conversational text, or non-query answers.

                {input_text}
            """
            
            # response = llm_chain.predict(question=input_text)
            response = llm_chain.predict(question=prompt_with_instruction)
            query_generation_time = time.time() - start_time
            query_tokens_used = count_tokens(response)

            print("Raw LLM Output:", response)
            # print("Memory:", memory.chat_memory.messages) #Print memory content.
            
            logging.info(f"Raw LLM Output: {response}")

            try:
                query = json.loads(response)
            except json.JSONDecodeError:
                logging.error(f"Invalid query response: {response}")
                st.error("The model's response is not a valid MongoDB query.")


            st.write(f"Time taken to generate the query: {query_generation_time:.2f} seconds")
            st.write(f"Tokens used for query generation: {query_tokens_used}")

            st.subheader("Generated MongoDB Query:")
            st.text(json.dumps(query, indent=4))

            start_time = time.time()
            query_results = execute_mongo_query(query)
            query_results_time = time.time() - start_time
            result_text = json.dumps(query_results)
            result_tokens_used = count_tokens(result_text)

            st.subheader("Query Results:")
            for row in query_results:
                st.write(row)

            st.write(f"Time taken to fetch results: {query_results_time:.2f} seconds")
            st.write(f"Tokens used for results: {result_tokens_used}")
            if query_results:
                groq_start_time = time.time()
                human_readable_output = get_groq_response(input_text, query_results)
                groq_response_time = time.time() - groq_start_time

                st.subheader("Final result:")
                st.write(human_readable_output)
                st.write(f"Time taken to generate human-like sentence: {groq_response_time:.2f} seconds")

        except Exception as e:
            st.error(f"Error: {e}")

