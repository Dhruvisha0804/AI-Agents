import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import json
from pymongo import MongoClient
from bson import ObjectId
import re
from pymongo.errors import PyMongoError
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load the schema from the JSON file
try:
    with open('schema.json', 'r') as file:
        db_schema = json.load(file)
except (FileNotFoundError, json.JSONDecodeError) as e:
    print(f"Error loading schema file: {e}")
    db_schema = {}


# Load sample queries and build FAISS index
try:
    with open('jf_sample.txt', 'r') as file:
        sample_queries = json.load(file)  # This will parse the JSON content into a list of dictionaries
except (FileNotFoundError, IOError, json.JSONDecodeError) as e:
    print(f"Error loading sample file: {e}")
    sample_queries = []

# Extract questions for embeddings
sample_questions = [item['question'] for item in sample_queries]

# Check if sample_questions is empty
if not sample_questions:
    print("No sample questions found. Please check f_sample.txt and ensure it is in correct JSON format.")
    exit()  # Terminate the program

# Encode sample questions for FAISS index
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
sample_embeddings = embedding_model.encode(sample_questions, convert_to_tensor=True).cpu().numpy()

# Create FAISS index
index = faiss.IndexFlatL2(sample_embeddings.shape[1])
index.add(sample_embeddings)

def get_similar_samples(user_query, top_k=3):
    query_vector = embedding_model.encode([user_query], convert_to_tensor=True).cpu().numpy()
    _, sample_indices = index.search(query_vector, top_k)
    return [sample_queries[i] for i in sample_indices[0]]

# MongoDB Connection
client = MongoClient("mongodb://localhost:27017")
db = client['task_demo']

# Load Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base_model_id = "microsoft/phi-2"
tokenizer = AutoTokenizer.from_pretrained(base_model_id, use_fast=True)

compute_dtype = getattr(torch, "float16")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    trust_remote_code=True,
    quantization_config=bnb_config,
    revision="refs/pr/23",
    device_map="auto",
    torch_dtype="auto",
)

adapter = 'Chirayu/phi-2-mongodb'
model = PeftModel.from_pretrained(model, adapter).to(device)

# User input
user_input = input("Enter your text here: ")

if user_input.strip():
    similar_samples = get_similar_samples(user_input, top_k=3)

    # print("\n-----", similar_samples, "\n")

    # Inject top samples into the prompt
    sample_references = "\n".join(
        [f"Question: {sample['question']}\nMongoDB Query: {json.dumps(sample['query'], indent=4)}"
         for sample in similar_samples]
    )

    print("\n*****", sample_references, "\n")

    prompt =  f"""
        <s>
        You are an expert MongoDB query generator. Follow these guidelines:
        1. Use `status.text` for filtering task statuses.
        2. Ensure all ObjectId values are formatted as `ObjectId("...")`.
        3. Refer to the following sample queries for guidance:

        {sample_references}

        Only return the MongoDB query, no additional details.

        ### Instruction:
        {user_input}

        ### MongoDB Query:
        """

    model_inputs = tokenizer(prompt, return_tensors="pt", max_length=2048, truncation=True).to(device)
    output = model.generate(
        **model_inputs,
        max_length=2048,
        no_repeat_ngram_size=10,
        repetition_penalty=1.02,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )[0]

    prompt_length = model_inputs['input_ids'].shape[1]
    query = tokenizer.decode(output[prompt_length:], skip_special_tokens=True)

    # Clean query output
    generated_query = query.split('</s>')[0].strip()

    print("Raw Generated Query from Model:")
    print(generated_query)

    try:
        collection_name = generated_query.split(".find(")[0].replace("db.", "").strip()
        query_body = generated_query.split(".find(")[1].rstrip(")")
        query_body = re.sub(r'"ObjectId\((.*?)\)"', r'ObjectId(\1)', query_body)

        try:
            query_result = eval(f"db.{collection_name}.find({query_body})")
            if query_result.count() == 0:
                print("No documents found for the given query.")
        except PyMongoError as e:
            print(f"MongoDB Error: {e}")
        except Exception as e:
            print(f"General Error: {e}")

        print("\nGenerated MongoDB Query:")
        print(generated_query)

        print("\nQuery Results:")
        for doc in query_result:
            print(doc)
    except Exception as e:
        print(f"Error executing query: {e}")
else:
    print("Please enter a valid text input to generate the query.")
