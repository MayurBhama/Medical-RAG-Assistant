from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils import fetch_pubmed_abstracts
import os 

def build_vector_db(query):
    #Fetching data from utils.py
    print(f"Fetching research for: {query}...")
    raw_text = fetch_pubmed_abstracts(query , max_results = 10)

    #Splitting text into chunks 
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 600, 
        chunk_overlap = 50,
        separators = ["\n\n", "\n", " ", ""]
    )

    texts = text_splitter.split_text(raw_text)
    print(f"Split into {len(texts)} chunks.")

    #Creating Embeddings 
    print("Converting text into vectors")
    embedding = HuggingFaceEmbeddings(model_name = "all-MiniLM-L6-v2")

    #Save to chromaDB 
    vector_db = Chroma.from_texts(
        texts,
        embedding,
        persist_directory = "./medical_db"
    )

    print("Memory created and saved in 'medical_db' folder!")

if __name__ == "__main__":
    # Test by building a memory about a specific topic
    build_vector_db("Immunotherapy for lung cancer 2024")
