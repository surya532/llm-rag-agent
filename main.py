import os
from dotenv import load_dotenv
from src.retrieval import get_retriever

load_dotenv()

def run_agent(query: str):
    retriever = get_retriever()
    relevant_docs = retriever.invoke(query)
    # Pass context + query into your LLM tool-calling chain here
    print(f"Query: {query}")
    print(f"Retrieved {len(relevant_docs)} context documents.")

if __name__ == "__main__":
    run_agent("What is the system architecture?")