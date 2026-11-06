from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

def get_retriever(docs=None):
    embeddings = OpenAIEmbeddings()
    if docs:
        vectorstore = Chroma.from_texts(docs, embeddings, persist_directory="./chroma_db")
    else:
        vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    return vectorstore.as_retriever()