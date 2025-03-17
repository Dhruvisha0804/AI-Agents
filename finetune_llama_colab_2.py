from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
import torch
import os
import re
import json
import logging
from datetime import datetime
from bson import ObjectId
from pymongo import MongoClient
from huggingface_hub import login
from peft import PeftModel


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HUGGINGFACE_TOKEN"] = "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"
login(token=os.environ["HUGGINGFACE_TOKEN"])

# MongoDB Local Connection
MONGO_URI = "mongodb://localhost:27017"
client = MongoClient(MONGO_URI)
db = client["task_demo"]
tasks_collection = db["tasks"]
users_collection = db["users"]
projects_collection = db["projects"]

# Enable logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load tokenizer and model
model_path = "./llama-finetuned"  # Path to your fine-tuned model
base_model_path = "meta-llama/Llama-3.2-1B"  # Base model path
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(base_model_path, device_map="auto")
model = PeftModel.from_pretrained(model, model_path)
model.eval()

# Load data
logging.info("Loading dataset...")
dataset = load_dataset('json', data_files='formatted_data.jsonl')
logging.info(f"Dataset loaded successfully. Dataset size: {len(dataset['train'])}")

tokenizer.pad_token = tokenizer.eos_token

quantization_config = BitsAndBytesConfig(load_in_4bit=True)
model = AutoModelForCausalLM.from_pretrained(base_model_path, quantization_config=quantization_config, device_map="auto")
model = PeftModel.from_pretrained(model, model_path)
model.eval()

# Tokenization
def tokenize_data(example):
    tokenized_output = tokenizer(
        f"{example['prompt']} {example['completion']}",
        padding='max_length',
        truncation=True,
        max_length=256,
        return_tensors="pt"
    )
    labels = tokenized_output["input_ids"].clone()
    return {"input_ids": tokenized_output["input_ids"],
            "attention_mask": tokenized_output["attention_mask"],
            "labels": labels}

logging.info("Starting tokenization...")
tokenized_dataset = dataset.map(tokenize_data, batched=True, remove_columns=dataset['train'].column_names)
logging.info("Tokenization complete.")

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
            elif isinstance(value, dict) or isinstance(value, list):
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
            results = list(collection.aggregate(query["aggregate"]))
        elif "filter" in query:
            logging.info(f"Executing MongoDB filter query: {query}")
            collection = db[query["collection"]]
            results = list(collection.find(query["filter"], query.get("projection", {})))
        else:
            raise ValueError("Query must contain either 'aggregate' or 'filter' field")

        return results

    except Exception as e:
        logging.error(f"Error executing query: {str(e)}")
        return [f"Error: {str(e)}"]

prompt_template = """
You are an expert in converting English questions into MongoDB queries!
sample_question: {{"prompt": "### Instruction: How many tasks exist?\\n### MongoDB Query: {{'collection': 'tasks', 'aggregate': [{{'$count': 'total_tasks'}}]}}", "completion": "### Response: {{'total_tasks': 66543}}"}}
sample_question: {{"prompt": "### Instruction: list all the task of 'Shiv Joshi' which status marked as 'In Progress'\\n### MongoDB Query: {{'collection': 'tasks', 'aggregate': [{{'$lookup': {{'from': 'users', 'localField': 'AssigneeUserId', 'foreignField': '_id', 'as': 'user_info'}}}}, {{'$unwind': '$user_info'}}, {{'$match': {{'user_info.Employee_Name': 'Shiv Joshi','status.text': 'In Progress'}}}}, {{'$project': {{'TaskName': 1, '_id': 0}}]}}}}", "completion": "### Response: [{{'TaskName': 'Audio record limit'}}, {{'TaskName': 'Snapshot optimization'}}, {{'TaskName': 'Additional 2: Create and sent certificate as per attached'}}, {{'TaskName': 'fvrduirghsbvsgbjfughyrf'}}]"}}

User Question: {}
Generated MongoDB Query:
"""

user_question = "How many tasks exist?"
prompt = prompt_template.format(user_question)

tokenizer.pad_token = tokenizer.eos_token
input_ids = tokenizer(prompt, return_tensors="pt", padding=True).input_ids.to(model.device)

try:
    outputs = model.generate(input_ids, max_length=512, num_return_sequences=1, temperature=0.5, top_k=50)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    logging.info(f"Generated text: {generated_text}")

    # Regex pattern to capture the first MongoDB query (corrected)
    pattern = r"Generated MongoDB Query:\s*({.*?})\s*(?=User Question:|Response:|$)"
    match = re.search(pattern, generated_text, re.DOTALL)

    if match:
        query_str = match.group(1).strip()

        # Fix single quotes to double quotes for JSON compatibility
        query_str = query_str.replace("'", '"')

        # Load the corrected JSON query
        query_data = json.loads(query_str)

        print("Input Query: ", query_str)

        # Extract collection and pipeline
        collection_name = query_data['collection']
        pipeline = query_data['aggregate']

        # Execute MongoDB query
        if collection_name == 'tasks':
            result = list(tasks_collection.aggregate(pipeline))
        elif collection_name == 'users':
            result = list(users_collection.aggregate(pipeline))
        else:
            result = []

        # Display result
        if result:
            print("Query Result:")
            for doc in result:
                print(doc)
        else:
            print("No data found matching the query.")
    else:
        print("No MongoDB query found.")
except json.JSONDecodeError as e:
    logging.error(f"Query Parsing Error: {e}")
except Exception as e:
    logging.error(f"Unexpected Error: {e}")


    # query_start = generated_text.find('{')
    # query_end = generated_text.rfind('}') + 1
    # if query_start != -1 and query_end != 0:
    #     query_json = generated_text[query_start:query_end]
    #     query = json.loads(query_json)
    #     logging.info(f"Extracted query: {query}")

    # query_results = execute_mongo_query(first_query)
    # logging.info("*** MongoDB Query Result ***")
    # for res in query_results:
    #     logging.info(res)

    # else:
    #     logging.error("Could not extract valid JSON query from generated text.")