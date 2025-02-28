from pymongo import MongoClient
import json
import os
import numpy as np
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from dotenv import load_dotenv
import logging
from datetime import datetime
from bson import ObjectId
import time
import tiktoken
from sentence_transformers import SentenceTransformer
import requests
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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
    logging.warning("GROQ_API_KEY is missing. Groq functionality will be disabled.")
    st.warning("GROQ API key not provided. Skipping human-like result generation.")

# Initialize OpenAI LLM
llm = ChatOpenAI(model="gpt-4", temperature=0.0, openai_api_key=OPENAI_API_KEY)

# Load embedding model (this is now not used directly)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# MongoDB Local Connection
MONGO_URI = "mongodb://localhost:27017"
client = MongoClient(MONGO_URI)
db = client["task_demo"]
tasks_collection = db["tasks"]
users_collection = db["users"]
projects_collection = db["projects"]

filename = 'sample.txt'

st.title("MongoDB AI Agent with LangChain-OpenAI and TF-IDF Cosine Similarity")
st.write("Ask anything and get an answer")
input_text = st.text_area("Enter your question here")

def pretty_print_json(query):
    try:
        query_dict = json.loads(query)
        return json.dumps(query_dict, indent=4)
    except json.JSONDecodeError:
        return query

def read_samples(filename):
    samples = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        question = None
        query = None
        is_query = False
        for line in lines:
            line = line.strip()
            if line.startswith('Question'):
                if question and query:
                    samples.append((question, query))
                question = line[len('Question'):].strip()
                query = None
                is_query = False
            elif line.startswith('Query'):
                is_query = True
            elif is_query:
                if line.startswith('json'):
                    line = line[4:].strip()
                query = (query + " " + line) if query else line
        if question and query:
            samples.append((question, query))
    return samples

def find_similar_question(user_query, samples, threshold=0.6):
    questions = [sample[0] for sample in samples]
    queries = {sample[0]: sample[1] for sample in samples}
    
    # Initialize the TfidfVectorizer
    vectorizer = TfidfVectorizer()
    
    # Fit and transform the questions into TF-IDF vectors
    tfidf_matrix = vectorizer.fit_transform(questions)
    
    # Transform the user's query into the same vector space
    user_query_vector = vectorizer.transform([user_query])
    
    # Compute cosine similarity between the user's query and all stored questions
    cosine_similarities = cosine_similarity(user_query_vector, tfidf_matrix).flatten()
    
    # Find the top matches based on cosine similarity
    top_indices = cosine_similarities.argsort()[-5:][::-1]  # Get top 5 most similar questions
    
    # Filter results based on the threshold
    results = []
    for idx in top_indices:
        score = cosine_similarities[idx]
        if score >= threshold:
            results.append((questions[idx], score, queries[questions[idx]]))
    
    # If no matches are found above the threshold, return None
    if results:
        best_match = results[0]  # Best match
        return best_match[0], best_match[2], results
    else:
        return None, None, []

prompt = """
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

    Below are several sample user questions related to the MongoDB document provided,
    and the corresponding MongoDB aggregation pipeline queries that can be used to fetch the desired data.
    Use them wisely.

    sample_question: {similar_question}
    corresponding_query: {corresponding_query}

    As an expert you must use them whenever required. And note that us sample question you have to modify query as per the users question. 
    Note: You have to just return the query nothing else. Don't return any additional detail with the query. Please follow this strictly.
    input: {user_query}
    output:

    Please return only the MongoDB query for the user's question. The output should be a valid aggregation pipeline query.
"""

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
                except Exception:
                    pass
            elif isinstance(value, dict):
                convert_to_objectid(value)
    elif isinstance(query, list):
        for i in range(len(query)):
            convert_to_objectid(query[i])
    return query

def execute_mongo_query(query):
    if not query:
        st.error("Query could not be generated. Please try a different question.")
        return []

    try:
        query = convert_dates(query)
        query = convert_to_objectid(query)

        # Ensure "collection" key exists
        if "collection" not in query:
            raise ValueError("Missing 'collection' key in query.")

        collection = db[query["collection"]]

        if "aggregate" in query:
            logging.info(f"Executing MongoDB aggregation query: {json.dumps(query, indent=4)}")
            results = list(collection.aggregate(query["aggregate"]))  # Convert cursor to list
        elif "filter" in query:
            logging.info(f"Executing MongoDB filter query: {json.dumps(query, indent=4)}")
            results = list(collection.find(query["filter"], query.get("projection", {})))  # Convert cursor to list
        else:
            raise ValueError("Query must contain either 'aggregate' or 'filter' field.")

        return results

    except Exception as e:
        logging.error(f"Error executing query: {str(e)}")
        return [f"Error: {str(e)}"]

def count_tokens(text, model_name="gpt-3.5-turbo"):
    encoding = tiktoken.encoding_for_model(model_name)
    tokens = encoding.encode(text)
    return len(tokens)

def get_groq_response(question, query_results):
    if not GROQ_API_KEY:
        return "GROQ API key not configured."
    try:
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        prompt_groq = """ ... """
        data = {
            "model": GROQ_MODEL,
            "messages": [
                {"role": "system", "content": "You are a highly advanced language model."},
                {"role": "user", "content": question},
                {"role": "assistant", "content": query_results}
            ]
        }
        response = requests.post(GROQ_API_URL, headers=headers, json=data)
        response.raise_for_status()
        return response.json().get("choices", [{}])[0].get("message", {}).get("content", "No response content.")
    except Exception as e:
        logging.error(f"Error fetching Groq response: {str(e)}")
        return "Error in fetching Groq response."


# Combine PromptTemplate and LLMChain setup in one definition
query_with_prompt = PromptTemplate(
    template=prompt,
    input_variables=["similar_question", "user_query"]
)

llmchain = LLMChain(llm=llm, prompt=query_with_prompt, verbose=True)

# Code for reading samples and finding similar questions remains unchanged
samples = read_samples(filename)

def process_query(user_query):
    samples = read_samples(filename)
    similar_question, corresponding_query, results = find_similar_question(user_query, samples)

    if corresponding_query:
        try:
            # Ensure the query is passed as a dictionary, not a string
            if isinstance(corresponding_query, str):
                corresponding_query = json.loads(corresponding_query)  # Only load if it's a string

            mongo_results = execute_mongo_query(corresponding_query)  # Pass the query directly

            token_count = count_tokens(user_query)
            if mongo_results:
                response = json.dumps(mongo_results, indent=4)
                if GROQ_API_KEY:
                    response = get_groq_response(user_query, response)
                st.write(f"Query Result:\n{pretty_print_json(response)}")
                st.write(f"Token count for the query: {token_count} tokens.")
            else:
                st.write("No matching results found in the database.")
        except Exception as e:
            st.write(f"Error executing query: {e}")
    else:
        st.write("No corresponding query found.")




if input_text:
    button = st.button("Submit")
    if button:
        start_time = time.time()
        try:
            question = input_text
            similar_question, _, results = find_similar_question(question, samples, threshold=0.6)

            # Handle the case when a similar question is found
            if similar_question:
                st.write(f"Most similar question found: {similar_question}")
                input_prompt = query_with_prompt.format(similar_question=similar_question, user_query=question)

                # Print the input text and the formatted prompt
                print("Input Text:")
                print(question)  # User's input text

                print("Formatted Prompt that will be sent to OpenAI:")
                print(input_prompt)  # The prompt that will be sent to OpenAI

                input_tokens_used = count_tokens(input_prompt)
                st.write(f"Tokens used for input prompt: {input_tokens_used}")

                process_query(question)

                # Generate query using OpenAI
                response = llmchain.invoke({
                    "similar_question": similar_question,
                    "user_query": question
                })
                query_text = response["text"]
                query_tokens_used = count_tokens(query_text)

                try:
                    query = json.loads(query_text)
                    query_generation_time = time.time() - start_time
                    st.write(f"Time taken to generate the query: {query_generation_time:.2f} seconds")
                    st.write(f"Tokens used for query generation: {query_tokens_used}")
                    st.subheader("Generated MongoDB Query:")
                    st.text(json.dumps(query, indent=4))

                    # Execute query
                    query_start_time = time.time()
                    query_results = execute_mongo_query(query)
                    query_results_time = time.time() - query_start_time

                    if isinstance(query_results, list):
                        st.subheader("Query Results:")
                        for row in query_results:
                            st.write(row)

                        st.write(f"Time taken to fetch results: {query_results_time:.2f} seconds")
                        result_text = json.dumps(query_results)
                        result_tokens_used = count_tokens(result_text)
                        st.write(f"Tokens used for results: {result_tokens_used}")

                        # Generate human-readable output
                        human_start_time = time.time()
                        human_readable_output = get_groq_response(question, query_results)
                        human_like_gen_time = time.time() - human_start_time

                        st.subheader("Final Result:")
                        st.write(human_readable_output)
                        st.write(f"Time taken to generate human-like sentence: {human_like_gen_time:.2f} seconds")
                    else:
                        st.error(f"Query execution failed: {query_results}")

                except json.JSONDecodeError:
                    logging.error(f"Invalid JSON received: {query_text}")
                    st.error("AI-generated query is not valid JSON. Please rephrase your question.")
            else:
                st.write("No similar question found.")

        except Exception as e:
            st.error(f"Error executing query: {e}")

# Process for non-Streamlit version (console output)
user_query = input_text
samples = read_samples(filename)

similar_question, corresponding_query, results = find_similar_question(user_query, samples, threshold=0.6)


if similar_question:
    print(f"Similar Question: {similar_question}")
    print("Corresponding Query:")
    print(pretty_print_json(corresponding_query))
    print("\nAll Matches and Similarity Scores:")
    for question, score, query in results:
        print(f"Question: {question} \nScore: {score}\n")
else:
    print("No similar question found.")
