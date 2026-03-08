import os
from pathlib import Path
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader 

INDEX_NAME = os.getenv("PINECONE_INDEX")  # the name of the database
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY")) # pinecone client

# ensure index exists with correct dimensions for text-embedding-3-small
if not pc.has_index(INDEX_NAME):
    # create new index (index is essentially a vector database table)
    pc.create_index(
        name=INDEX_NAME,
        dimension=1536, # size of each vector stored in the index
        metric="cosine",  # how similarity between vectors is calculated
        spec=ServerlessSpec(cloud=os.getenv("PINECONE_CLOUD"), region=os.getenv("PINECONE_ENV"))
    )

# object that can convert text into embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

def ingest_pdf(file_path):
    """
    - file_path: path to pdf
    """
    
    loader = PyPDFLoader(file_path) # loader object
    pages = loader.load()  # one document per page

    # configure splitter
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        model_name="gpt-4o", # tokenizer to use
        chunk_size=600,  # each chunk about 600 tokens long
        chunk_overlap=150,  # chunks overlap to ensure info at boundaries not lost
    )
    
    # split pages into smaller chunks
    chunks = text_splitter.split_documents(pages)
    
    # normalize source names
    for chunk in chunks:
        src = chunk.metadata.get("source", file_path)
        # returns only the filename
        chunk.metadata["source"] = Path(src).name

    # embed text, convert to vector format, and insert into pinecone
    PineconeVectorStore.from_documents(
        chunks, # documents to embed
        embeddings,  # embedding model to use
        index_name=INDEX_NAME
    )
    
    return f"Successfully ingested {len(chunks)} chunks from {Path(file_path).name}"