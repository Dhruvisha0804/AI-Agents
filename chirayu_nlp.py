from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from pymongo import MongoClient
from bson import ObjectId

# Load the model and tokenizer
model = AutoModelForSeq2SeqLM.from_pretrained("Chirayu/nl2mongo")
tokenizer = AutoTokenizer.from_pretrained("Chirayu/nl2mongo")

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# MongoDB Connection
client = MongoClient("mongodb://localhost:27017")
db = client['task_demo']

# Query generation function
def generate_query(
        textual_query: str,
        num_beams: int = 10,
        max_length: int = 128,
        repetition_penalty: float = 2.5,
        length_penalty: float = 1.0,
        early_stopping: bool = True,
        top_p: float = 0.95,
        top_k: int = 50,
        num_return_sequences: int = 1
    ) -> str:
    
    input_ids = tokenizer.encode(textual_query, return_tensors="pt", add_special_tokens=True).to(device)
    
    generated_ids = model.generate(
        input_ids=input_ids,
        num_beams=num_beams,
        max_length=max_length,
        repetition_penalty=repetition_penalty,
        length_penalty=length_penalty,
        early_stopping=early_stopping,
        top_p=top_p,
        top_k=top_k,
        num_return_sequences=num_return_sequences,
    )
    
    query = tokenizer.decode(
        generated_ids[0],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    )
    
    return query

# Function to execute the MongoDB query
def execute_query(query: str):
    try:
        # Convert string-based ObjectId to actual ObjectId
        if 'ObjectId' in query:
            query = query.replace('"ObjectId(',"ObjectId(").replace(')"',")")
        
        # Execute the query in MongoDB
        result = eval(f"db.{query}")
        return list(result) if result else "No matching data found."
    
    except Exception as e:
        return f"Error executing query: {e}"

# Example query
textual_query = "how many task are there which are marked as 'Inprogress'?"
# generated_query = generate_query(textual_query)
generated_query = "db.users.count()"

print(f"Generated Query: {generated_query}")
result = execute_query(generated_query)
print("Result:", result)
