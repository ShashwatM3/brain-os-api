from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import os
from openai import OpenAI
from dotenv import load_dotenv
import chromadb

load_dotenv()

client = chromadb.CloudClient(
  api_key=os.environ.get("CHROMA_API_KEY"),
  tenant=os.environ.get("CHROMA_TENANT"),
  database=os.environ.get("CHROMA_DATABASE"),
)

# client = OpenAI(
#     # This is the default and can be omitted
#     api_key=os.environ.get("OPENAI_API_KEY"),
# )

app = FastAPI(title="RAG API for BrainOS")

# Allow requests from all origins (for simplicity)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get("/")
def read_root():
    return {"message": "Welcome to Fun OpenAI API!"}

@app.get("/query")
def query(query: str = Query(..., description="Input text")):
    collection = client.get_or_create_collection(
      name="myCollection",
    )
    result = collection.query(
      query_texts=[f"{query}"]
    )
    results_str = ""
    for i in range(len(result["documents"][0])):
      if "file_name" in result["metadatas"][0][i]:
        results_str += f"""
        Source: {result["metadatas"][0][i]["file_name"]}
        Content: {result["documents"][0][i]}
        """
    return {"query": query, "context": results_str}