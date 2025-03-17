# import openai
# import os

# openai.api_key = os.getenv("OPENAI_API_KEY")

# try:
#     response = openai.chat.completions.create(
#         model="gpt-4",
#         messages=[{"role": "user", "content": "Test"}]
#     )
#     print(response.choices[0].message.content)
# except Exception as e:
#     print(f"OpenAI Test Error: {e}")


# import streamlit as st
# import openai
# import os

# openai.api_key = os.getenv("OPENAI_API_KEY")

# if st.button("Test OpenAI"):
#     try:
#         response = openai.chat.completions.create(
#             model="gpt-4",
#             messages=[{"role": "user", "content": "Test"}]
#         )
#         st.write(response.choices[0].message.content)
#     except Exception as e:
#         st.error(f"Streamlit OpenAI Test Error: {e}")


# import difflib
# import json

# # Function to read the sample.txt and extract questions and queries
# def read_samples(filename):
#     samples = []
#     with open(filename, 'r') as f:
#         lines = f.readlines()
#         question = None
#         query = None
#         is_query = False  # Flag to detect if we're reading the query
#         for line in lines:
#             # Parse the Question and Query sections
#             if line.startswith('Question'):
#                 if question and query:
#                     samples.append((question.strip(), query.strip()))
#                 question = line[len('Question '):].strip()
#                 query = None  # Reset query for each new question
#                 is_query = False
#             elif line.startswith('Query'):
#                 is_query = True  # Flag that we're starting to read the query
#             elif is_query:
#                 # Skip the 'json' word and capture the query properly
#                 query = query + line.strip() if query else line.strip()
#                 if query.startswith('json'):
#                     query = query[4:].strip()  # Remove the 'json' prefix
#         if question and query:
#             samples.append((question.strip(), query.strip()))  # Add the last question-query pair
#     return samples

# # Function to find the most similar question with score threshold
# def find_similar_question(user_query, samples, threshold=0.5):
#     questions = [sample[0] for sample in samples]
#     queries = {sample[0]: sample[1] for sample in samples}
    
#     # Find all matches and their similarity scores
#     matches = difflib.get_close_matches(user_query, questions, n=len(questions), cutoff=threshold)
    
#     if not matches:
#         return None, None, []

#     # Find similarity scores for each match
#     results = []
#     for match in matches:
#         score = difflib.SequenceMatcher(None, user_query, match).ratio()
#         results.append((match, score, queries[match]))

#     # Return the best match and the score
#     best_match = results[0]
#     return best_match[0], best_match[2], results

# # Function to pretty-print the JSON query
# def pretty_print_json(query):
#     try:
#         query_dict = json.loads(query)  # Convert the query string to a dictionary
#         return json.dumps(query_dict, indent=4)  # Return the pretty-printed JSON
#     except json.JSONDecodeError:
#         return query  # If invalid JSON, return as is

# # Example usage
# filename = 'sample.txt'
# user_query = "all tasks group by in status"
# samples = read_samples(filename)

# similar_question, corresponding_query, results = find_similar_question(user_query, samples, threshold=0.5)

# if similar_question:
#     print(f"Similar Question: {similar_question}")
#     print("Corresponding Query:")
#     print(pretty_print_json(corresponding_query))  # Print the query nicely
#     print("\nAll Matches and Similarity Scores:")
#     for question, score, query in results:
#         print(f"Question: {question} \nScore: {score}\n")
# else:
#     print("No similar question found.")



# from pymongo import MongoClient
# import logging

# # MongoDB Local Connection
# MONGO_URI = "mongodb://localhost:27017"
# client = MongoClient(MONGO_URI)

# def test_mongo_connection():
#     try:
#         # Check if we can connect to the MongoDB server
#         server_info = client.server_info()  # This will raise an exception if the connection fails
#         print("MongoDB Connection Successful!")
#         # Check if the collections exist in the 'task_demo' database
#         db = client["task_demo"]
#         collections = db.list_collection_names()
#         print("Collections in 'task_demo' database:", collections)

#         # Optionally, check if each collection is accessible by trying to find a document in each
#         for collection_name in collections:
#             collection = db[collection_name]
#             # Just checking if we can fetch a single document from each collection
#             sample_doc = collection.find_one()
#             print(f"Sample document from {collection_name}: {sample_doc}")
#     except Exception as e:
#         print(f"Error connecting to MongoDB: {e}")
#         logging.error(f"Error connecting to MongoDB: {e}")

# # Run the test function
# test_mongo_connection()


# import transformers
# import torch

# model_id = "meta-llama/Llama-3.3-70B-Instruct"

# pipeline = transformers.pipeline(
#     "text-generation",
#     model=model_id,
#     model_kwargs={"torch_dtype": torch.bfloat16},
#     device_map="auto",
# )

# messages = [
#     {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
#     {"role": "user", "content": "Who are you?"},
# ]

# outputs = pipeline(
#     messages,
#     max_new_tokens=256,
# )
# print(outputs[0]["generated_text"][-1])


# from huggingface_hub import snapshot_download
# from llama_cpp import Llama

# model_name_or_path = "meta-llama/Llama-3.2-3B-Instruct" #or 3B version
# local_model_path = "./models"  # Directory to save the model

# # Download the model from Hugging Face Hub
# snapshot_download(repo_id=model_name_or_path, local_dir=local_model_path)

# # Path to the original model files
# original_model_path = local_model_path

# # Path to save the quantized model
# quantized_model_path = "./quantized_models/llama-3.2-3b-instruct.Q4_0.gguf"

# # Quantize the model (example: Q4_0 quantization)
# llm = Llama(model_path=f"{original_model_path}/model.safetensors", n_gpu_layers=-1, n_ctx=2048, logits_all=True, verbose=True)
# llm.convert_to_gguf(quantized_model_path, quantize="Q4_0")

# # Load the quantized model
# llm = Llama(model_path=quantized_model_path)

# print(llm("The best programming language is "))



# import torch
# from transformers import pipeline

# model_id = "meta-llama/Llama-3.2-1B"

# pipe = pipeline(
#     "text-generation", 
#     model=model_id, 
#     torch_dtype=torch.bfloat16, 
#     device_map="auto"
# )

# pipe("The key to life is")







# from transformers import AutoModelForCausalLM

# model_id = "meta-llama/Llama-3.2-3B-Instruct"  # Replace with correct ID
# try:
#     model = AutoModelForCausalLM.from_pretrained(model_id)
#     print("Model loaded successfully!")
# except Exception as e:
#     print(f"Error: {e}")


# import torch

# if torch.cuda.is_available():
#     print("GPU is available!")
#     print(f"GPU device name: {torch.cuda.get_device_name(0)}") #prints the name of the first available gpu
#     device = torch.device("cuda")
# else:
#     print("GPU is NOT available.")
#     device = torch.device("cpu")


# from transformers import AutoModelForCausalLM, AutoTokenizer


# model_name = "Qwen/Qwen2.5-3B"
# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     torch_dtype="auto",
#     device_map="auto"
# )
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# prompt = "Give me a short introduction to large language model."
# messages = [
#     {"role": "user", "content": prompt}
# ]
# text = tokenizer.apply_chat_template(
#     messages,
#     tokenize=False,
#     add_generation_prompt=True
# )
# model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
# generated_ids = model.generate(
#     **model_inputs,
#     max_new_tokens=512
# )
# generated_ids = [
#     output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
# ]
# response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]


# import json

# # Try loading the JSON file
# try:
#     with open('val.json', 'r') as file:
#         data = json.load(file)
#     print("The file is a valid JSON.")
# except json.JSONDecodeError as e:
#     print(f"Invalid JSON. Error: {e}")


# import pandas as pd

# # Load the JSON file into a DataFrame
# # df = pd.read_json('train.json')
# df = pd.read_json('test.json')
# # df = pd.read_json('val.json')

# # Drop rows with missing values in any column
# df = df.dropna()

# # Check if the query field is properly structured
# print(df['query'].head())



# import pandas as pd
# from datasets import Dataset, DatasetDict

# def load_and_validate_json(filename):
#     try:
#         df = pd.read_json(filename, lines=True)
#         return Dataset.from_pandas(df)
#     except ValueError as e:
#         print(f"Error loading {filename}: {e}")
#         return None

# train_dataset = load_and_validate_json("train.json")
# val_dataset = load_and_validate_json("val.json")
# test_dataset = load_and_validate_json("test.json")

# if train_dataset and val_dataset and test_dataset:
#     dataset_dict = DatasetDict({
#         "train": train_dataset,
#         "validation": val_dataset,
#         "test": test_dataset
#     })
#     dataset_dict.save_to_disk("./my_dataset")


# import json

# def is_valid_json_file(filepath):
#     """Checks if a JSON file is valid."""
#     try:
#         with open(filepath, 'r', encoding='utf-8') as f:
#             for line in f:
#                 json.loads(line)
#         return True
#     except json.JSONDecodeError as e:
#         print(f"Error in {filepath}: {e}")
#         return False
#     except FileNotFoundError:
#         print(f"File not found: {filepath}")
#         return False
#     except Exception as e:
#         print(f"An unexpected error occured with {filepath}: {e}")
#         return False

# # Check each file
# is_valid_json_file("train.json")
# is_valid_json_file("val.json")
# is_valid_json_file("test.json")


# import pandas as pd

# try:
#     # Read the JSON data
#     df = pd.read_json("data.json")

#     # Fill NaN values with empty strings
#     df = df.fillna("")

#     # Select the desired columns
#     df = df[["question", "query", "response"]]

#     # Convert the 'response' column to string representation of lists if it's a list
#     df['response'] = df['response'].apply(lambda x: str(x) if isinstance(x, list) else x)

#     # Save the DataFrame to a CSV file
#     df.to_csv("data.csv", index=False)

#     print("data.json has been successfully converted to data.csv.")

# except FileNotFoundError:
#     print("Error: data.json not found.")
# except Exception as e:
#     print(f"An error occurred: {e}")


# from datasets import load_dataset 

# # Load the dataset
# dataset = load_dataset("tatsu-lab/alpaca") 
# train = dataset['train']


# from pymongo import MongoClient
# from transformers import pipeline

# # MongoDB connection
# client = MongoClient("mongodb://localhost:27017")
# db = client['task_demo']
# collection = db['tasks']

# # Load fine-tuned model
# pipe = pipeline("text-generation", model="./llama-finetuned", tokenizer="./llama-finetuned")

# # Query execution logic
# def fetch_data(query):
#     try:
#         mongo_query = eval(query)  # Convert query string to dict
#         result = list(collection.aggregate(mongo_query['aggregate'])) if 'aggregate' in mongo_query else collection.find(mongo_query)
#         return result
#     except Exception as e:
#         return {"error": str(e)}

# # Example usage
# user_input = "How many tasks are marked as 'Done'?"
# query_response = pipe(f"### Instruction: {user_input}", max_length=200)[0]['generated_text']
# mongo_query = query_response.split("### MongoDB Query: ")[1].split("### Response:")[0].strip()

# # Fetch data
# result = fetch_data(mongo_query)
# print(f"Result: {result}")


# import torch
# print(f"Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")



# import re

# text = """Generated text: You are an expert in converting English questions into MongoDB queries!
# sample_question: {"prompt": "### Instruction: How many tasks exist?\n### MongoDB Query: {'collection': 'tasks', 'aggregate': [{'$count': 'total_tasks'}]}", "completion": "### Response: {'total_tasks': 66543}"}
# sample_question: {"prompt": "### Instruction: list all the task of 'Shiv Joshi' which status marked as 'In Progress'\n### MongoDB Query: {'collection': 'tasks', 'aggregate': [{'$lookup': {'from': 'users', 'localField': 'AssigneeUserId', 'foreignField': '_id', 'as': 'user_info'}}, {'$unwind': '$user_info'}, {'$match': {'user_info.Employee_Name': 'Shiv Joshi','status.text': 'In Progress'}}, {'$project': {'TaskName': 1, '_id': 0}]}}", "completion": "### Response: [{'TaskName': 'Audio record limit'}, {'TaskName': 'Snapshot optimization'}, {'TaskName': 'Additional 2: Create and sent certificate as per attached'}, {'TaskName': 'fvrduirghsbvsgbjfughyrf'}]"}

# User Question: list all the task of 'Shiv Joshi' which status marked as 'In Progress'
# Generated MongoDB Query:
# {'collection': 'tasks', 'aggregate': [{'$lookup': {'from': 'users', 'localField': 'AssigneeUserId', 'foreignField': '_id', 'as': 'user_info'}}, {'$unwind': '$user_info'}, {'$match': {'user_info.Employee_Name': 'Shiv Joshi','status.text': 'In Progress'}}, {'$project': {'TaskName': 1, '_id': 0}]}}

# User Question: list all the task of 'Shiv Joshi' which status marked as 'In Progress'
# Generated MongoDB Query:
# {'collection': 'tasks', 'aggregate': [{'$lookup': {'from': 'users', 'localField': 'AssigneeUserId', 'foreignField': '_id', 'as': 'user_info'}}, {'$unwind': '$user_info'}, {'$match': {'user_info.Employee_Name': 'Shiv Joshi','status.text': 'In Progress'}}, {'$project': {'TaskName': 1, '_id': 0}]}}

# Response: [{'TaskName': 'Audio record limit']
# """

# # Regex pattern to capture the first MongoDB query
# pattern = r'Generated MongoDB Query:\s*({.*?})\s*(?=User Question:|Response:|$)'
# match = re.search(pattern, text, re.DOTALL)

# if match:
#     first_query = match.group(1)
#     print(first_query)
# else:
#     print("No MongoDB query found.")



# import re
# import json
# from pymongo import MongoClient

# # MongoDB connection setup
# client = MongoClient('mongodb://localhost:27017')
# db = client['task_demo']  # Replace with your database name
# tasks_collection = db['tasks']
# users_collection = db['users']

# # Sample text containing the query
# text = """Generated text: You are an expert in converting English questions into MongoDB queries!
# MongoDB Query:
# {
#     "collection": "tasks",
#     "aggregate": [
#         { "$lookup": { "from": "users", "localField": "AssigneeUserId", "foreignField": "_id", "as": "user_info" } },
#         { "$unwind": "$user_info" },
#         { "$match": { "user_info.Employee_Name": "Shiv Joshi",
#                       "status.text": "In Progress"} },
#         { "$project": { "TaskName": 1, "_id": 0 } }
#     ]
# }
# """

# # Extract MongoDB query
# pattern = r'MongoDB Query:\s*({.*?})\s*(?=User Question:|Response:|$)'
# match = re.search(pattern, text, re.DOTALL)

# if match:
#     query_str = match.group(1).strip()
#     query_data = json.loads(query_str)  # Convert string to dict

#     # Extract collection and pipeline
#     collection_name = query_data['collection']
#     pipeline = query_data['aggregate']

#     # Execute MongoDB query
#     if collection_name == 'tasks':
#         result = list(tasks_collection.aggregate(pipeline))
#     elif collection_name == 'users':
#         result = list(users_collection.aggregate(pipeline))
#     else:
#         result = []

#     # Display result
#     if result:
#         print("Query Result:")
#         for doc in result:
#             print(doc)
#     else:
#         print("No data found matching the query.")
# else:
#     print("No MongoDB query found.")




import json

# Try loading the JSON file
try:
    with open('schema.json', 'r') as file:
        data = json.load(file)
    print("The file is a valid JSON.")
except json.JSONDecodeError as e:
    print(f"Invalid JSON. Error: {e}")
