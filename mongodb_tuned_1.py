import streamlit as st
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
import torch
from peft import PeftModel
import json

# Load the schema from the JSON file
with open('schema.json', 'r') as file:
    db_schema = json.load(file)

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
    device_map={"": 0},
    torch_dtype="auto",
    flash_attn=True,
    flash_rotary=True,
    fused_dense=True,
)
adapter = 'Chirayu/phi-2-mongodb'
model = PeftModel.from_pretrained(model, adapter).to(device)

# Streamlit UI
st.title("MongoDB Query Generator")
st.write("Enter your question, and the app will generate the corresponding MongoDB query.")

# User input block
user_input = st.text_area("Enter your text here:", "")

# Generate query on button click
if st.button("Generate Query"):
    if user_input.strip():
        prompt = f"""<s> 
            Task Description:
            You are an expert in converting English questions into MongoDB queries for the 'task_demo' database.
            This database contains three collections: 'tasks', 'users', and 'projects', with nested and embedded data.

            **Collections Structure:**
            - 'tasks' collection: fields like AssigneeUserId, TaskName, Status (embedded with text, key, type), ProjectID, Task_Priority, createdAt, SprintArray, Task_Leader.
            - 'users' collection: fields like _id, Employee_FName, Employee_LName, Employee_Name, createdAt.
            - 'projects' collection: fields like _id, AssigneeUserid, LeaderUserId, ProjectCategory, ProjectName.

            **Important Rules:**
            1. Always reference `status.text` for task status filtering (e.g., `status.text: "Done"`).
            2. Always use `sprintArray.folderName` for sprint folder filtering (e.g., `sprintArray.folderName: "Development"`).
            3. For simple field filtering, use `filter`. For aggregation, only include the aggregation pipeline—no `filter` field.

            MongoDB Schema: 
            {db_schema}

            ### Instruct:
            {user_input}

            ### Output:
            """

        model_inputs = tokenizer(prompt, return_tensors="pt").to(device)
        output = model.generate(
            **model_inputs,
            max_length=4096,
            no_repeat_ngram_size=10,
            repetition_penalty=1.02,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )[0]

        prompt_length = model_inputs['input_ids'].shape[1]
        query = tokenizer.decode(output[prompt_length:], skip_special_tokens=False)

        # Clean query output
        try:
            stop_idx = query.index("</s>")
        except Exception:
            stop_idx = len(query)

        generated_query = query[:stop_idx].strip()

        # Display generated query
        st.text_area("Generated MongoDB Query:", generated_query, height=200)
    else:
        st.warning("Please enter a valid text input to generate the query.")
