import streamlit as st
from pymongo import MongoClient
import io, json, os
from langchain_openai import ChatOpenAI
# from langchain_core.prompts import PromptTemplate
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
# from langchain.chains.llm import LLMChain
from dotenv import load_dotenv
import sys
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    st.error("Missing OpenAI API Key. Set OPENAI_API_KEY in environment variables.")
    st.stop()

# Initialize OpenAI LLM
llm = ChatOpenAI(model="gpt-4", temperature=0.0, openai_api_key=OPENAI_API_KEY)

# MongoDB Local Connection
MONGO_URI = "mongodb://localhost:27017"
client = MongoClient(MONGO_URI)
db = client["task_demo"]
tasks_collection = db["tasks"]
users_collection = db["users"]

st.title("MongoDB AI Agent with LangChain-OpenAI")
st.write("Ask anything and get an answer")
input_text = st.text_area("Enter your question here")

# Load sample file
with io.open("sample.txt", "r", encoding="utf-8") as f1:
    sample = f1.read()



prompt ="""

    You are an expert in converting English questions into MongoDB queries!
    The database is named 'task_demo' and contains two collections: 'tasks' and 'users'.
    including nested and embedded data structures that add depth and detail to the document.
    The 'tasks' collection has fields: AssigneeUserId, TaskName, Status (embedded object with text, key, and type), ProjectID, Task_Priority, CreatedAt, etc.
    The 'users' collection has fields: _id, Employee_FName, Employee_LName, Employee_Name, createdAt, etc.

    **Important Rule**: Always reference the `status.text` field when filtering by task status (e.g., `status.text: "Done"`).
    
    **Additional Rule**: 
    - When you need to filter tasks or query simple fields, you can use the `filter` field.
    - If the query requires aggregation, **do not include a `filter` field**. Instead, only include the aggregation pipeline. 

    ***Relationship Between users and tasks***

    Each task can have one or more assignees, represented by AssigneeUserId in the tasks collection. This field stores an array of references to the _id of users in the users collection.
    Users can be assigned multiple tasks, as seen in the AssigneeUserId field of the tasks collection.

    Below are several sample user questions related to the MongoDB document provided, 
    and the corresponding MongoDB aggregation pipeline queries that can be used to fetch the desired data.
    Use them wisely.

    sample_question: {sample}
    As an expert you must use them whenever required.
    Note: You have to just return the query nothing else. Don't return any additional detail with the query.Please follow this strictly
    input:{question}
    output:

    Please return only the MongoDB query for the user's question. The output should be a valid aggregation pipeline query.

"""

#     You are an expert in converting English questions into MongoDB queries!
#     The database is named 'task_demo' and contains two collections: 'tasks' and 'users'.
#     The 'tasks' collection has fields: AssigneeUserId, TaskName, Status (embedded object with text, key, and type), ProjectID, Task_Priority, CreatedAt, etc.
#     The 'users' collection has fields: _id, Employee_FName, Employee_LName, Employee_Name, createdAt, etc.
#     In both collections, the '_id' from 'users' and 'AssigneeUserId' from 'tasks' are the same, and they need to be joined for certain queries.

#     **Important Rule**: Always reference the `status.{{{{text}}}}` field when filtering by task status (e.g., `status.{{{{text}}}}: "Done"`) If '{{{{text}}}}' in status otherwise take status.{{{{text}}}} as 'to-do'.
    
#     **Additional Rule**: 
#     - When you need to filter tasks or query simple fields, you can use the `filter` field.
#     - If the query requires aggregation, **do not include a `filter` field**. Instead, only include the aggregation pipeline. For example:

#     you are a very intelligent AI assitasnt who is expert in identifying relevant questions fro user
#     from user and converting into nosql mongodb agggregation pipeline query.
#     Note: You have to just return the query as to use in agggregation pipeline nothing else. Don't return any other thing
#     Please use the below schema to write the mongodb queries , dont use any other queries.
#     schema:
#     the mentioned mogbodb collection talks about listing for an accommodation on Airbnb. The schema for this document represents the structure of the data, describing various properties related to the listing, host, reviews, location, and additional features. 
#     your job is to get python code for the user question

#     ***users Collection Schema***

#     The users collection stores information about employees in the system. Each document in this collection contains the following fields:

#     _id (ObjectId): A unique identifier for the user document.
#     Example: "6571e71590622593f7e24f23"

#     legacyId (String): A unique identifier used in legacy systems.
#     Example: "1NY9PYJCUURu357sR8ws28n92El2"

#     Employee_Email (String): The email address of the user.
#     Example: "timothy@cinemano.com.au"

#     Employee_FName (String): The user's first name.
#     Example: "Timothy"

#     Employee_LName (String): The user's last name.
#     Example: "Nguyen"

#     Employee_Name (String): The full name of the user (concatenation of first and last name).
#     Example: "Timothy Nguyen"

#     updatedAt (Date): The last time the user document was updated.
#     Example: "2023-08-28T18:11:21.000Z"

#     createdAt (Date): The time when the user document was created.
#     Example: "2023-12-07T15:39:02.443Z"

#     ***tasks Collection Schema***

#     The tasks collection stores information about tasks assigned to users in the system. Each document in this collection contains the following fields:

#     _id (ObjectId): A unique identifier for the task document.
#     Example: "6571e71c5470e64b12032972"

#     legacyId (String): A unique identifier used in legacy systems.
#     Example: "fazFKZEcWAl04djXpdII"

#     TaskName (String): The name of the task.
#     Example: "Add Steps in the Question Preview"

#     TaskKey (String): A unique key assigned to the task.
#     Example: "AFCQBOSS-55"

#     AssigneeUserId (Array[ObjectId]): An array of user IDs who are assigned to the task. These are references to documents in the users collection.
#     Example: ["6571e715e5d066d39dd01b39"]

#     TaskType (String): The type of task.
#     Example: "task"

#     status (Object): An object containing information about the task’s status.
#     text (String): The status text (e.g., "Done")
#     key (Number): A key representing the status
#     type (String): The type of status (e.g., "done")
#     Example: {{ 
#         "text": "Done", 
#         "key": 6, 
#         "type": "done" }}

#     Task_Leader (ObjectId): The user ID of the task leader.
#     Example: "6571e715e5d066d39dd01b39"

#     Task_Priority (String): The priority of the task (e.g., "MEDIUM")
#     Example: "MEDIUM"

#     subTasks (Number): The number of sub-tasks associated with this task.
#     Example: 4

#     rawDescription (String): A raw version of the task description (unformatted).
#     Example: "1. Button color is still not correct..."

#     description (String): The task description in HTML format.
#     Example: "<p>1. Button color is still not correct...</p>"

#     statusType (String): The type of status (e.g., "done")
#     Example: "done"

#     startDate (Date): The start date of the task.
#     Example: "2023-12-05T05:32:00.000Z"

#     statusKey (Number): A key representing the task’s status.
#     Example: 6

#     createdAt (Date): The time when the task document was created.
#     Example: "2023-12-05T05:31:58.105Z"

#     updatedAt (Date): The last time the task document was updated.
#     Example: "2025-01-28T13:14:22.605Z"


#     ***Relationship Between users and tasks***

#     Each task can have one or more assignees, represented by AssigneeUserId in the tasks collection. This field stores an array of references to the _id of users in the users collection.
#     Users can be assigned multiple tasks, as seen in the AssigneeUserId field of the tasks collection.

#     This schema provides a comprehensive view of the data structure for an task_demo database in MongoDB with two major collections 'users' and 'tasks', 
#     including nested and embedded data structures that add depth and detail to the document.
#     use the below sample_examples to generate your queries perfectly
#     sample_example:

#     Below are several sample user questions related to the MongoDB document provided, 
#     and the corresponding MongoDB aggregation pipeline queries that can be used to fetch the desired data.
#     Use them wisely.

#     sample_question: {sample}
#     As an expert you must use them whenever required.
#     Note: You have to just return the query nothing else. Don't return any additional detail with the query.Please follow this strictly
#     input:{question}
#     output:

#     Please return only the MongoDB query for the user's question. The output should be a valid aggregation pipeline query.

# """


# def execute_mongo_query(query):
#     try:
#         logging.info(f"Executing MongoDB query: {query}")
        
#         # Handling count_documents queries
#         if "count_documents" in query:
#             count = db[query["collection"]].count_documents(query["filter"])
#             return [f"Count: {count}"]

#         # Handling aggregation queries (Fix for 'pipeline' issue)
#         elif "pipeline" in query:
#             results = list(db[query["collection"]].aggregate(query["pipeline"]))
        
#         # Handling standard find queries
#         else:
#             results = list(db[query["collection"]].find(query["filter"], query.get("projection", {})))
        
#         logging.info("MongoDB query executed successfully.")
#         return results if results else ["No records found."]

#     except Exception as e:
#         logging.error(f"Database Error: {e}")
#         return [f"Database Error: {str(e)}"]


def execute_mongo_query(query):
    try:
        # Check if the query requires aggregation
        if "aggregate" in query:
            # Execute aggregation pipeline
            logging.info(f"Executing MongoDB aggregation query: {query}")
            collection = db[query["collection"]]  # Assuming query["collection"] gives the correct collection
            results = collection.aggregate(query["aggregate"])  # Execute aggregation pipeline
        elif "filter" in query:
            # Execute simple filter query
            logging.info(f"Executing MongoDB filter query: {query}")
            collection = db[query["collection"]]  # Assuming query["collection"] gives the correct collection
            results = collection.find(query["filter"], query.get("projection", {}))  # Execute filter query
        else:
            raise ValueError("Query must contain either 'aggregate' or 'filter' field")

        return list(results)  # Return the query results as a list

    except Exception as e:
        logging.error(f"Error executing query: {str(e)}")
        return [f"Error: {str(e)}"]
    

    
# Prompt template and chain setup (no changes needed here)
query_with_prompt = PromptTemplate(
    template=prompt,
    input_variables=["question", "sample"]
)
llmchain = LLMChain(llm=llm, prompt=query_with_prompt, verbose=True)

if input_text is not None:
    button = st.button("Submit")
    if button:
        try:
            print(input_text)  # Debugging the user input
            response = llmchain.invoke({"question": input_text, "sample": sample})
            print("*********************", response)
            print("Raw LLM Output:", response["text"])  # Debugging raw LLM output
            query = json.loads(response["text"])
            print("Parsed Query:", query)  # Debugging the parsed query
            
            # Display query on Streamlit
            st.subheader("Generated MongoDB Query:")
            st.text(json.dumps(query, indent=4))  # Print the query as a formatted string

            query_results = execute_mongo_query(query)

            try:
                query_results = execute_mongo_query(query)
            except json.JSONDecodeError as e:
                logging.error(f"Query Parsing Error: {e}")
                query_results = [f"Query Parsing Error: {str(e)}"]
            except Exception as e:
                logging.error(f"Unexpected Error: {e}")
                query_results = [f"Unexpected Error: {str(e)}"]

            # Display query results
            st.subheader("Query Results:")
            for row in query_results:
                st.write(row)

        except Exception as e:
            st.error(f"Error: {e}")



        #     if type(query_results) != list:
        #         st.error("LLM did not return a list, which is required for a mongodb aggregate query.")
        #     else:
        #         results = tasks_collection.aggregate(query)
        #         for result in results:
        #             st.write(result)
        except json.JSONDecodeError as e:
            st.error(f"Error parsing JSON: {e}. LLM output was not valid JSON.")
        except Exception as e:
            st.error(f"Error executing query: {e}")






# - "List tasks created after a specific day" → 
# {
#     "collection": "tasks",
#     "filter": { "createdAt": { "$gt": { "$date": "2025-02-20T00:00:00.000Z" } } },
#     "projection": { "TaskName": 1, "_id": 0 }
# }