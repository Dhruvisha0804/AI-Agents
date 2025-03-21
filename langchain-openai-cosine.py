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
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import requests
import openai

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

if not OPENAI_API_KEY:
    st.error("Missing OpenAI API Key. Set OPENAI_API_KEY in environment variables.")
    st.stop()

if not GROQ_API_KEY:
    logging.error("GROQ_API_KEY is missing. Please set it in the environment variables.")

# Initialize OpenAI LLM
llm = ChatOpenAI(model="gpt-4", temperature=0.0, openai_api_key=OPENAI_API_KEY)

# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Load sample queries from JSON file
def load_sample_queries(filepath):
    with open(filepath, 'r') as file:
        return json.load(file)

sample_queries = load_sample_queries('sample-json.txt')
stored_queries = [item['question'] for item in sample_queries]
query_mappings = {item['question']: item['query'] for item in sample_queries}

def find_similar_sentence_transformer(user_query, stored_queries):
    if not stored_queries:
        return None
    user_embedding = embedding_model.encode([user_query])
    stored_embeddings = embedding_model.encode(stored_queries)
    cosine_similarities = cosine_similarity(user_embedding, stored_embeddings)
    if cosine_similarities.max() > 0.5:
        most_similar_index = cosine_similarities.argmax()
        return stored_queries[most_similar_index]
    return None


# MongoDB Local Connection
MONGO_URI = "mongodb://localhost:27017"
client = MongoClient(MONGO_URI)
db = client["task_demo"]
tasks_collection = db["tasks"]
users_collection = db["users"]
projects_collection = db["projects"]

st.title("MongoDB AI Agent with LangChain-OpenAI and Sentence Transformers")
st.write("Ask anything and get an answer")
input_text = st.text_area("Enter your question here")

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
        print("**********", query)
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
    
def handle_query(question):
    try:
        # Preprocess question
        similar_question = find_similar_sentence_transformer(question, stored_queries)
        if similar_question:
            similar_query = query_mappings[similar_question]
            query_text = execute_mongo_query(question, similar_query)
        else:
            query_text = execute_mongo_query(question)

        # Validate and execute the query
        query = json.loads(query_text)
        if "collection" not in query:
            raise ValueError("Query does not contain the 'collection' key.")
        
        query_results = execute_mongo_query(query)
        if query_results:
            human_readable_output = get_groq_response(question, query_results)
            return human_readable_output
        else:
            return "No results returned from MongoDB."

    except Exception as e:
        logging.error(f"Error: {e}")
        return f"An error occurred: {str(e)}"


def count_tokens(text, model_name="gpt-3.5-turbo"):
    encoding = tiktoken.encoding_for_model(model_name)
    tokens = encoding.encode(text)
    return len(tokens)

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

if input_text:
    button = st.button("Submit")
    if button:
        start_time = time.time()
        try:
            question = input_text

            input_tokens_used = count_tokens(question)
            st.write(f"Tokens used for input prompt: {input_tokens_used}")

            similar_question = find_similar_sentence_transformer(question, stored_queries)
            if similar_question:
                similar_query = query_mappings[similar_question]
                prompt = f"""
                    You are an expert in converting English questions into MongoDB queries!
                    The database is named 'task_demo' and contains three collections: 'tasks', 'users', and 'projects',
                    including nested and embedded data structures that add depth and detail to the document.
                    The 'tasks' collection has fields: AssigneeUserId, TaskName, Status (embedded object with text, key, and type), 
                    ProjectID, Task_Priority, createdAt, SprintArray, Task_Leader etc.
                    The 'users' collection has fields: _id, Employee_FName, Employee_LName, Employee_Name, createdAt, etc.
                    The 'projects' collection has fields: _id, AssigneeUserid, LeaderUserId, ProjectCategory, ProjectName, etc.

                    **Important Rules:**

                    1. **Always include the `"collection"` key in your output JSON object.** This is essential for the query to be executed.
                    2. Always reference the `status.text` field when filtering by task status (e.g., `status.text: "Done"`).
                    3. Always reference the `sprintArray.folderName` field when filtering by sprint folders (e.g., `sprintArray.folderName: "Development"`).
                    4. When you need to filter tasks or query simple fields, you can use the `filter` field.
                    5. If the query requires aggregation, **do not include a `filter` field**. Instead, only include the aggregation pipeline.
                    6. If you are unsure of the collection, use `"collection": "tasks"`.

                    ***Relationship Between users, tasks, and projects***

                    Each task (tasks.AssigneeUserId) can have multiple assignees (users._id), and users can be assigned to multiple tasks.
                    Each task (tasks.ProjectID) belongs to a single project (projects._id), and one project may have multiple assignees.

                    Here is an example to follow:
                    question: "List out all the projects that are led by the leader id '6571e715e5d066d39dd01ae2'"
                    query: {{
                        "collection": "projects",
                        "filter": {{
                            "LeaderUserId": "6571e715e5d066d39dd01ae2"
                        }}
                    }}

                    Note: You have to just return the query, nothing else. Don't return any additional detail with the query. Please follow this strictly.
                    input: {question}
                    output:

                    Please return only the MongoDB query for the user's question. The output **must** be a valid JSON object with the `"collection"` key.
                """


                # Simplified OpenAI API call
                try:
                    response = openai.chat.completions.create(
                        model="gpt-4",
                        messages=[
                            {"role": "system", "content": "You are an expert..."},
                            {"role": "user", "content": prompt.format(question=question, similar_question = similar_question, similar_query = json.dumps(similar_query))}
                        ]
                    )
                    st.write(f"Raw OpenAI Response: {response.choices[0].message.content}") #print the response.
                    query_text = response.choices[0].message.content
                except Exception as e:
                    st.error(f"OpenAI API Error: {e}")
                
                
                query_with_prompt = PromptTemplate(
                    template=prompt,
                    input_variables=["question"]
                )
                llmchain = LLMChain(llm=llm, prompt=query_with_prompt, verbose=True)

                # Debugging: Print the value of the question variable
                print(f"Question Variable: {question}")

                # Debugging: Print the input to llmchain.invoke()
                print(f"Input to llmchain.invoke(): {{\"question\": question}}")

                response = llmchain.invoke({"question": question})
                print("*********************", response)
                query_generation_time = time.time() - start_time
                query_text = response["text"]
                query_tokens_used = count_tokens(query_text)

                # Print the raw output from OpenAI
                st.write(f"Raw OpenAI Output: {response['text']}")

                try:
                    query = json.loads(response["text"])
                    if "collection" not in query:
                        raise ValueError("LLM output is missing the 'collection' key.")
                except json.JSONDecodeError as e:
                    st.error(f"Error parsing JSON: {e}. LLM output was not valid JSON.")
                    query = None #set query to none to prevent further execution.
                except ValueError as ve:
                    st.error(f"Error: {ve}")
                    query = None #set query to none to prevent further execution.

                st.write(f"Time taken to generate the query: {query_generation_time:.2f} seconds")
                st.write(f"Tokens used for query generation: {query_tokens_used}")
            else:
                prompt = f"""
                    You are an expert in converting English questions into MongoDB queries!
                    The database is named 'task_demo' and contains three collections: 'tasks', 'users', and 'projects'.
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

                    Note: You have to just return the query nothing else. Don't return any additional detail with the query.Please follow this strictly.
                    input:{question}
                    output:

                    Please return only the MongoDB query for the user's question. The output should be a valid aggregation pipeline query.
                """
                query_with_prompt = PromptTemplate(
                    template=prompt,
                    input_variables=["question"]
                )
                llmchain = LLMChain(llm=llm, prompt=query_with_prompt, verbose=True)

                # Debugging: Print the value of the question variable
                print(f"Question Variable: {question}")

                # Debugging: Print the input to llmchain.invoke()
                print(f"Input to llmchain.invoke(): {{\"question\": question}}")

                response = llmchain.invoke({"question": question})
                print("*********************", response)
                query_generation_time = time.time() - start_time
                query_text = response["text"]
                query_tokens_used = count_tokens(query_text)
                query = json.loads(response["text"])

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

                if query is not None: #only execute if query is not None.
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
                    human_like_gen_time = time.time() - start_time
                    human_readable_output = get_groq_response(question, query_results)

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