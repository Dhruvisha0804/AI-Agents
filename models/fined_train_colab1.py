import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import Dataset
import json

# Load data
with open("mini-schema.json") as f:
    schema_data = json.load(f)

with open("sample_data.json") as f:
    sample_data = json.load(f)

# Format data for training
def format_data(sample_data, schema_data):
    schema_str = json.dumps(schema_data, indent=2)
    data = []
    for sample in sample_data:
        prompt = f"Schema Information:\n{schema_str}\n\nQuestion: {sample['question']}"
        response = sample['query']
        data.append({"prompt": prompt, "response": response})
    return data

dataset = format_data(sample_data, schema_data)

df = Dataset.from_list([{"text": f"{d['prompt']}\n{d['response']}"} for d in dataset])

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")

tokenizer.pad_token = tokenizer.eos_token

# Tokenization
def preprocess_function(examples):
    return tokenizer(examples["text"], padding=True, truncation=True)

dataset = df.map(preprocess_function, batched=True)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="no",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    save_strategy="epoch",
    load_best_model_at_end=False,
)

# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

# Train the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained("./fine_tuned_llama")
tokenizer.save_pretrained("./fine_tuned_llama")

# Inference function
def generate_query(question):
    input_text = f"Schema Information:\n{json.dumps(schema_data, indent=2)}\n\nQuestion: {question}"
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True).to("cuda")
    outputs = model.generate(**inputs, max_length=300, num_return_sequences=1)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example usage
question = "List all tasks for employee 'Shiv Joshi'."
print("Generated Query:", generate_query(question))
