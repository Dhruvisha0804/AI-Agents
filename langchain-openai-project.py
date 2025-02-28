import streamlit as st
from pymongo import MongoClient
import io, json, os
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from dotenv import load_dotenv
import sys
import logging
from datetime import datetime
from bson import ObjectId
import time  
import tiktoken

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    st.error("Missing OpenAI API Key. Set OPENAI_API_KEY in environment variables.")
    st.stop()


import requests

# Groq API Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

if not GROQ_API_KEY:
    logging.error("GROQ_API_KEY is missing. Please set it in the environment variables.")

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

# Load sample file
with io.open("sample.txt", "r", encoding="utf-8") as f1:
    sample = f1.read()



prompt ="""

    You are an expert in converting English questions into MongoDB queries!
    The database is named 'task_demo' and contains two collections: 'tasks' and 'users'.
    including nested and embedded data structures that add depth and detail to the document.
    The 'tasks' collection has fields: AssigneeUserId, TaskName, Status (embedded object with text, key, and type), ProjectID, Task_Priority, createdAt, SprintArray, Task_Leader etc.
    The 'users' collection has fields: _id, Employee_FName, Employee_LName, Employee_Name, createdAt, etc.
    The 'projects' collection has fields: _id, AssigneeUserid, LeaderUserId, ProjectCategory, ProjectName, etc.

    **Important Rule**: Always reference the `status.text` field when filtering by task status (e.g., `status.text: "Done"`).
    Always reference the `sprintArray.folderName` field when filtering by sprint folders (e.g., `sprintArray.folderName: "Development"`).
    
    **Additional Rule**: 
    - When you need to filter tasks or query simple fields, you can use the `filter` field.
    - If the query requires aggregation, **do not include a `filter` field**. Instead, only include the aggregation pipeline. 

    ***Relationship Between users, tasks, and projects***

    Each task (tasks.AssigneeUserId) can have multiple assignees (users._id), and users can be assigned to multiple tasks.
    Each task (tasks.ProjectID) belongs to a single project (projects._id), one project may have multiple assignees.

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


from datetime import datetime

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
    
    
# Prompt template and chain setup (no changes needed here)
query_with_prompt = PromptTemplate(
    template=prompt,
    input_variables=["question", "sample"]
)
llmchain = LLMChain(llm=llm, prompt=query_with_prompt, verbose=True)

if input_text is not None:
    button = st.button("Submit")
    if button:

        start_time = time.time()

        try:
            print(input_text)  # Debugging the user input
            question = input_text  # Ensure 'question' is initialized here
            response = llmchain.invoke({"question": question, "sample": sample})

            # Record time and tokens for query generation
            query_generation_time = time.time() - start_time

            # Count tokens for the generated query
            query_text = response["text"]
            query_tokens_used = count_tokens(query_text)  # Count tokens in the response text

            # print("*********************", response)
            print("Raw LLM Output:", response["text"])  # Debugging raw LLM output
            query = json.loads(response["text"])  # Parse the generated query
            
            st.write(f"Time taken to generate the query: {query_generation_time:.2f} seconds")
            st.write(f"Tokens used for query generation: {query_tokens_used}")
            
            # Display query on Streamlit
            st.subheader("Generated MongoDB Query:")
            st.text(json.dumps(query, indent=4))  # Print the query as a formatted string

            start_time = time.time()

            query_results = execute_mongo_query(query)
            # Record time for result fetching
            query_results_time = time.time() - start_time

            try:
                query_results = execute_mongo_query(query)
            except json.JSONDecodeError as e:
                logging.error(f"Query Parsing Error: {e}")
                query_results = [f"Query Parsing Error: {str(e)}"]
            except Exception as e:
                logging.error(f"Unexpected Error: {e}")
                query_results = [f"Unexpected Error: {str(e)}"]

            result_text = json.dumps(query_results)  # Assuming query results are in JSON format
            result_tokens_used = count_tokens(result_text)

            # Display query results
            st.subheader("Query Results:")
            
            for row in query_results:
                st.write(row)

            st.write(f"Time taken to fetch results: {query_results_time:.2f} seconds")
            st.write(f"Tokens used for results: {result_tokens_used}")

            # If the query results are valid, proceed
            if query_results:

                # Record time and tokens for query generation
                human_like_gen_time = time.time() - start_time

                # Get the human-readable response after executing the query
                human_readable_output = get_groq_response(question, query_results)

                # Output the response
                print(human_readable_output)
                st.subheader("Final result:")
                st.write(human_readable_output)

                st.write(f"Time taken to generate human like sentence: {query_results_time:.2f} seconds")

            else:
                st.error("No query results returned from MongoDB.")

        except Exception as e:
            st.error(f"Error: {e}")

        except json.JSONDecodeError as e:
            st.error(f"Error parsing JSON: {e}. LLM output was not valid JSON.")

        except Exception as e:
            st.error(f"Error executing query: {e}")
    