# db.py
import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

# Load .env
load_dotenv()
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_env = os.getenv("PINECONE_ENVIRONMENT")  # ✅ Required
index_name = os.getenv("PINECONE_INDEX_NAME", "bonhoeffer-bot")  # Default if missing

# ✅ Initialize Pinecone client explicitly (required with new SDK)
pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index(index_name)

def load_vector_store():
    # Initialize embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Return vector store connected to existing index
    vectorstore = PineconeVectorStore(
        index=index,
        embedding=embeddings,
    )

    return vectorstore
