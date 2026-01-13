# Custom LangChain Tools for Medical RAG Assistant
from langchain.tools import tool
from utils import fetch_pubmed_abstracts
from core import get_medical_answer
from ingest import build_vector_db

@tool
def search_pubmed(query: str) -> str:
    """
    Search PubMed for medical research papers on a specific topic.
    Use this when the user wants to find NEW research or papers on a medical topic.
    Returns abstracts of relevant papers.
    """
    return fetch_pubmed_abstracts(query, max_results=5)

@tool
def query_medical_knowledge(question: str) -> str:
    """
    Query the existing medical knowledge base (ChromaDB) to answer questions.
    Use this when the user asks a medical question that can be answered from stored research.
    Returns an answer based on previously ingested medical papers.
    """
    answer, docs = get_medical_answer(question)
    return answer

@tool
def ingest_new_research(topic: str) -> str:
    """
    Fetch and store new medical research papers into the knowledge base.
    Use this when the user wants to ADD new research on a topic to the database.
    This updates the ChromaDB with fresh papers from PubMed.
    """
    build_vector_db(topic)
    return f"Successfully ingested new research papers on '{topic}' into the knowledge base."
