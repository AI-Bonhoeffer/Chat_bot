# db.py
import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

# Load .env variables
load_dotenv()
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_env = os.getenv("PINECONE_ENVIRONMENT")
index_name = os.getenv("PINECONE_INDEX_NAME")

# ✅ Initialize Pinecone
pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index(name=index_name)

def load_vector_store():
    # ✅ Init HuggingFace Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # ✅ Correct usage of new PineconeVectorStore
    vectorstore = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embeddings
    )

    return vectorstore
