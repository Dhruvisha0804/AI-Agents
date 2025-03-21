import argparse
from dotenv import load_dotenv
from pymongo import MongoClient
from langchain.vectorstores import MongoDBAtlasVectorSearch
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
import warnings
import os

# Load environment variables from .env file
load_dotenv()

# Retrieve the OpenAI API key from the environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Filter out the UserWarning from langchain
warnings.filterwarnings("ignore", category=UserWarning, module="langchain.chains.llm")

# Process arguments
parser = argparse.ArgumentParser(description='Atlas Vector Search Demo')
parser.add_argument('-q', '--question', help="The question to ask")
args = parser.parse_args()

if args.question is None:
    # Some questions to try...
    query = "How many tasks are there by each different status?"
    query = "List out all users who have task status 'done'"

else:
    query = args.question

print("\nYour question:")
print("-------------")
print(query)

# MongoDB Local Connection
MONGO_URI = "mongodb://localhost:27017"
client = MongoClient(MONGO_URI)
db = client["task_demo"]
tasks_collection = db["tasks"]
users_collection = db["users"]
projects_collection = db["projects"]


vectorStore_tasks = MongoDBAtlasVectorSearch(
    tasks_collection, OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
)

# Perform a similarity search between the embedding of the query and the embeddings of the documents
# print("\nQuery Response:")
print("---------------")
docs = vectorStore_tasks.max_marginal_relevance_search(query, K=1)

print(docs[0].metadata['title'])
print(docs[0].page_content)

# Contextual Compression
llm = OpenAI(openai_api_key=OPENAI_API_KEY, temperature=0)
compressor = LLMChainExtractor.from_llm(llm)

compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vectorStore_tasks.as_retriever()
)

print("\nAI Response:")
print("-----------")
compressed_docs = compression_retriever.get_relevant_documents(query)
print(compressed_docs[0].metadata['title'])
print(compressed_docs[0].page_content)
