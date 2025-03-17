from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, BitsAndBytesConfig
from datasets import load_dataset
import torch.nn as nn
from typing import Dict, Union, Any
import torch.optim as optim
import torch
import os
from huggingface_hub import login
from peft import LoraConfig, get_peft_model
import logging
from datetime import datetime
from bson import ObjectId
from pymongo import MongoClient

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

# Load tokenizer and model
model_path = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(model_path)
print("Tokenizer loaded successfully.")
model = AutoModelForCausalLM.from_pretrained(model_path)
print("Model loaded successfully.")

# Load data
print("Loading dataset...")
dataset = load_dataset('json', data_files='formatted_data.jsonl')
print("Dataset loaded successfully.")
print(f"Dataset size: {len(dataset['train'])}")

tokenizer.pad_token = tokenizer.eos_token

quantization_config = BitsAndBytesConfig(load_in_4bit=True)

model = AutoModelForCausalLM.from_pretrained(model_path, quantization_config=quantization_config)

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

print("Starting tokenization...")
tokenized_dataset = dataset.map(tokenize_data, batched=True, remove_columns=dataset['train'].column_names)
print("Tokenization complete.")

# Training arguments
training_args = TrainingArguments(
    output_dir="./llama-finetuned",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=32,
    warmup_steps=100,
    max_steps=2000,
    learning_rate=2e-5,
    logging_dir="./logs",
    logging_steps=10,
    save_strategy="steps",
    save_steps=500,
    fp16=True
)

lora_config = LoraConfig(
    r=8,  # Rank of the update matrices
    lora_alpha=32,  # Scaling factor
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",  # Important for causal language models
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters() #display trainable parameters


# Trainer setup
class ShapeCheckingTrainer(Trainer):
    def training_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        optimizer: optim.Optimizer,
    ) -> torch.Tensor:
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        print(f"Input IDs shape: {input_ids.shape}")
        print(f"Attention mask shape: {attention_mask.shape}")
        return super().training_step(model, inputs, optimizer)

trainer = ShapeCheckingTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
)

# Enable logging
logging.basicConfig(level=logging.INFO)

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

# Fine-tuning
print("Starting training...")
trainer.train()
print("Training complete.")

# Save model
print("Saving model...")
trainer.save_model("./llama-finetuned")
print("Model saved successfully.")
tokenizer.save_pretrained("./llama-finetuned") #add this line
print("Tokenizer saved successfully.")


# model_path = "./llama-finetuned"  # Path where your fine-tuned model is saved
# tokenizer = AutoTokenizer.from_pretrained(model_path)
# model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B", device_map="auto") #load the base model
# model = PeftModel.from_pretrained(model, model_path) #load the trained adapters

# model.eval()  # Set the model to evaluation mode

# prompt_template = """
# You are an expert in converting English questions into MongoDB queries!
# sample_question: {{"prompt": "### Instruction: How many tasks exist?\\n### MongoDB Query: {{'collection': 'tasks', 'aggregate': [{{'$count': 'total_tasks'}}]}}", "completion": "### Response: {{'total_tasks': 66543}}"}}

# User Question: {}
# MongoDB Query:
# """

# user_question = "How many tasks are there by each different status?"

# prompt = prompt_template.format(user_question)

# tokenizer.pad_token = tokenizer.eos_token
# input_ids = tokenizer(prompt, return_tensors="pt", padding=True).input_ids.to(model.device)

# outputs = model.generate(input_ids, max_length=200, num_return_sequences=1, temperature=0.5, top_k=50) # Adjusted max_length and decoding params

# generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
# print("***", generated_text)



# Load the tokenizer and model
model_path = "./llama-finetuned"  # Path where your fine-tuned model is saved
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B", device_map="auto")  # Load the base model
model = PeftModel.from_pretrained(model, model_path)  # Load the trained adapters

model.eval()  # Set the model to evaluation mode

prompt_template = """
You are an expert in converting English questions into MongoDB queries!
sample_question: {{"prompt": "### Instruction: How many tasks exist?\\n### MongoDB Query: {{'collection': 'tasks', 'aggregate': [{{'$count': 'total_tasks'}}]}}", "completion": "### Response: {{'total_tasks': 66543}}"}}

User Question: {}
MongoDB Query:
"""

# Example user question
user_question = "How many tasks are there by each different status?"

# Format the prompt
prompt = prompt_template.format(user_question)

# Tokenization
tokenizer.pad_token = tokenizer.eos_token
input_ids = tokenizer(prompt, return_tensors="pt", padding=True).input_ids.to(model.device)

# Generate MongoDB query
outputs = model.generate(input_ids, max_length=200, num_return_sequences=1, temperature=0.5, top_k=50)  # Adjusted max_length and decoding params

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("*** Generated Query ***")
print(generated_text)

# Extract the MongoDB query from the generated text
# Assuming the generated query is in the proper format, you may need to parse or clean it before executing it

# Example: Parsing the query (you can modify this based on your specific output format)
query = generated_text.split("MongoDB Query:")[1].strip()

# Assuming the query is in valid JSON format, convert the query to a dictionary
# This step may require additional parsing based on the output from the model

# Example: Query might need to be adjusted or cleaned
query_dict = eval(query)  # Convert string representation of query to dictionary

# Execute the MongoDB query using the helper function
results = execute_mongo_query(query_dict)

# Output the result
print("*** MongoDB Query Result ***")
for res in results:
    print(res)