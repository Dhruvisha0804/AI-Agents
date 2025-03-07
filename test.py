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



import transformers
import torch

model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": "Who are you?"},
]

outputs = pipeline(
    messages,
    max_new_tokens=256,
)
print(outputs[0]["generated_text"][-1])






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

