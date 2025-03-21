# import json

# # Load schema and sample data
# with open("mini-schema.json", "r") as schema_file:
#     schema = json.load(schema_file)

# with open("sample_data.json", "r") as sample_file:
#     samples = json.load(sample_file)

# # Function to generate schema descriptions
# def generate_schema_info(schema):
#     schema_info = []
#     for collection, fields in schema.items():
#         schema_text = f"The '{collection}' collection includes the following fields:\n"
#         for field, details in fields.items():
#             details_str = ", ".join([f"{k}: {v}" for k, v in details.items()])
#             schema_text += f"- {field} ({details_str})\n"
#         schema_info.append(schema_text.strip())
#     return "\n\n".join(schema_info)

# # Combine data for fine-tuning
# training_data = []
# schema_info = generate_schema_info(schema)

# for sample in samples:
#     entry = f"Question: {sample['question']}\nSchema Info: {schema_info}\nQuery: {sample['query']}\n"
#     training_data.append(entry)

# # Save to text file
# with open("training_data.txt", "w") as output_file:
#     output_file.write("\n\n".join(training_data))

# print("Training data prepared successfully!")



# import json

# def txt_to_json(input_file, output_file):
#     data = []
#     with open(input_file, 'r') as file:
#         entry = {}
#         for line in file:
#             line = line.strip()
#             if line.startswith("Question:"):
#                 entry["input"] = line
#             elif line.startswith("Schema Info:"):
#                 entry["input"] += "\n" + line
#             elif line.startswith("Query:"):
#                 entry["output"] = line.replace("Query: ", "")
#                 data.append(entry)
#                 entry = {}  # Reset for the next entry

#     with open(output_file, 'w') as json_file:
#         json.dump(data, json_file, indent=4)

# # Convert your file
# txt_to_json("training_data.txt", "combined_data.json")
# print("Conversion completed successfully!")



from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import get_peft_model, LoraConfig
from datasets import load_dataset

# Load model and tokenizer
model_name = "meta-llama/Llama-3.2-1B"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# LoRA Configuration
config = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.1, bias='none')
model = get_peft_model(model, config)

tokenizer.pad_token = tokenizer.eos_token

# Load dataset
dataset = load_dataset('json', data_files='combined_data.json')

# Split dataset into train and validation
dataset = dataset['train'].train_test_split(test_size=0.1) 

# Formatting the dataset
def format_data(samples):
    return {
        "input_ids": tokenizer(samples["input"], padding=True, truncation=True, return_tensors="pt")["input_ids"],
        "labels": tokenizer(samples["output"], padding=True, truncation=True, return_tensors="pt")["input_ids"]
    }

dataset = dataset.map(format_data)

# Training arguments without evaluation
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="no",  # Disable evaluation
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    logging_dir="./logs",
    save_total_limit=2
)

# Trainer setup with eval_dataset
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['test']  # Provide the validation/test dataset for evaluation
)

# Train the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")

print("Model fine-tuning complete and saved!")
