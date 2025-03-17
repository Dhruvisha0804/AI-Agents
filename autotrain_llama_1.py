import pandas as pd
from sklearn.model_selection import train_test_split

# Read the JSON data
df = pd.read_json("data.json")
df = df.fillna("")  # Fill any NaN values with empty strings

# Initialize an empty list to hold the text for each row
text_col = []

# Define the prompt
prompt = """
    You are an expert in converting English questions into MongoDB queries for the 'task_demo' database. This database contains three collections: 'tasks', 'users', and 'projects', with nested and embedded data.

    **Collections Structure:**
    - 'tasks' collection: fields like AssigneeUserId, TaskName, Status (embedded with text, key, type), ProjectID, Task_Priority, createdAt, SprintArray, Task_Leader.
    - 'users' collection: fields like _id, Employee_FName, Employee_LName, Employee_Name, createdAt.
    - 'projects' collection: fields like _id, AssigneeUserid, LeaderUserId, ProjectCategory, ProjectName.

    **Important Rules:**
    1. Always reference `status.text` for task status filtering (e.g., `status.text: "Done"`).
    2. Always use `sprintArray.folderName` for sprint folder filtering (e.g., `sprintArray.folderName: "Development"`).
    3. For simple field filtering, use `filter`. For aggregation, only include the aggregation pipeline—no `filter` field.

    **Relationships:** Each task (tasks.AssigneeUserId) can have multiple assignees (users._id), and each task (tasks.ProjectID) belongs to one project (projects._id), with a project potentially having multiple assignees.

    **Task:** Use the provided sample question and corresponding query as a reference. Only return the MongoDB query in json format, no additional details.
"""

# Iterate through the dataframe rows
for _, row in df.iterrows():
    question = str(row['question'])
    query = str(row['query'])
    response = row['response']

    # Ensure response is converted to a string if it's a list
    if isinstance(response, list):
        response = "\n".join([str(item) for item in response])  # Join list items into a single string
    else:
        response = str(response)

    # Form the text block with the prompt and data
    if len(query.strip()) == 0:
        text = prompt + "### Question:" + question + "\n###Response:\n" + response
    else:
        text = prompt + "### Question:" + question + "\n###Query:\n" + query + "\n###Response:\n" + response

    # Append the formed text to the list
    text_col.append(text)

# Check the length of text_col to ensure it matches the dataframe rows
if len(text_col) == len(df):
    df['text'] = text_col
else:
    print(f"Length mismatch: {len(text_col)} != {len(df)}")

# Split the data into train, validation, and test
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)  # 80% training data
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)  # Split the remaining 20% into 10% validation and 10% test

# Save each split as a separate JSON file
train_df.to_json("train.json", orient='records', lines=True)
val_df.to_json("val.json", orient='records', lines=True)
test_df.to_json("test.json", orient='records', lines=True)

# Print the first few rows of the training set to confirm
print("Training Data:")
print(train_df.head())
