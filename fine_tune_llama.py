# from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
# from datasets import load_dataset

# print("Script started...")  # Confirm the script runs at all

# # Load tokenizer and model
# model_path = "meta-llama/Llama-3.2-1B"
# tokenizer = AutoTokenizer.from_pretrained(model_path)
# # if tokenizer.pad_token is None:
# #     tokenizer.add_special_tokens({'pad_token': '[PAD]'})
# print("Tokenizer loaded successfully.")
# # tokenizer = AutoTokenizer.from_pretrained(model_path, legacy=False)  # Use AutoTokenizer with legacy=False
# model = AutoModelForCausalLM.from_pretrained(model_path)
# print("Model loaded successfully.")

# # Load data
# print("Loading dataset...")
# dataset = load_dataset('json', data_files='formatted_data.jsonl')
# print("Dataset loaded successfully.")
# print(f"Dataset size: {len(dataset['train'])}")

# tokenizer.pad_token = tokenizer.eos_token

# # Tokenization
# def tokenize_data(example):
#     print("******")
#     # return tokenizer(f"{example['prompt']} {example['completion']}", truncation=True, padding='max_length')
#     # return tokenizer(f"{example['prompt']} {example['completion']}",
#     #         truncation=True,
#     #         padding='max_length',
#     #         max_length=2048,
#     #         pad_token=tokenizer.eos_token)
#     return tokenizer(
#         f"{example['prompt']} {example['completion']}",
#         padding='max_length',  # Ensures consistent batch size
#         truncation=True,
#         max_length=512
#     )


# # tokenized_dataset = dataset.map(tokenize_data, batched=True)
# print("Starting tokenization...")
# tokenized_dataset = dataset.map(tokenize_data, batched=True, remove_columns=dataset['train'].column_names)
# # tokenized_dataset = dataset.map(tokenize_data, batched=True)
# print("Tokenization complete.")


# # Training arguments
# training_args = TrainingArguments(
#     output_dir="./llama-finetuned",
#     per_device_train_batch_size=2,  # Reduce batch size if needed
#     gradient_accumulation_steps=16,  # Adjust gradient accumulation steps
#     warmup_steps=100,
#     max_steps=2000,
#     learning_rate=2e-5,
#     logging_dir="./logs",
#     logging_steps=10,
#     save_strategy="steps",
#     save_steps=500
# )

# # Trainer setup
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=tokenized_dataset['train'],
# )

# # Fine-tuning
# print("Starting training...")
# trainer.train()
# print("Training complete.")

# # Save model
# # model.save_pretrained("./llama-finetuned")
# print(trainer.state)
# print("Saving model...")
# trainer.save_model("./llama-finetuned")
# print("Model saved successfully.")
# # tokenizer.save_pretrained("./llama-finetuned")



from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
import torch.nn as nn
from typing import Dict, Union, Any
import torch.optim as optim
import torch

print("Script started...")

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

# Tokenization
def tokenize_data(example):
    tokenized_output = tokenizer(
        f"{example['prompt']} {example['completion']}",
        padding='max_length',
        truncation=True,
        max_length=512,
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
    per_device_train_batch_size=2,
    gradient_accumulation_steps=16,
    warmup_steps=100,
    max_steps=2000,
    learning_rate=2e-5,
    logging_dir="./logs",
    logging_steps=10,
    save_strategy="steps",
    save_steps=500
)

# Trainer setup
class ShapeCheckingTrainer(Trainer):
    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], optimizer: optim.Optimizer) -> torch.Tensor:
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        print(f"Input IDs shape: {input_ids.shape}")
        print(f"Attention mask shape: {attention_mask.shape}")
        return super().training_step(model, inputs, optimizer)

trainer = ShapeCheckingTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
)

# Fine-tuning
print("Starting training...")
trainer.train()
print("Training complete.")

# Save model
print(trainer.state)
print("Saving model...")
trainer.save_model("./llama-finetuned")
print("Model saved successfully.")