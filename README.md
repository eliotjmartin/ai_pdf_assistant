# AI PDF Assistant 
A Retrieval-Augmented Generation (RAG) application that allows users to upload PDF documents and engage in a factual Q&A session with the content. The system is restricted to using only the retrieved text chunks to answer questions. If the required information is not present in the provided PDF, the system will state it does not know the answer.

### Features
- Document Ingestion: Extracts text and retains page-level metadata from uploaded files using PyPDFLoader.
- Token-Based Chunking: Splits text into 600-token chunks with a 150-token overlap using the gpt-4o tiktoken encoder to prevent information loss at chunk boundaries.
- Vector Search: Generates embeddings via OpenAI's text-embedding-3-small and stores them in a Pinecone Serverless index using cosine similarity.
- Strict Generation: Retrieves the 4 most relevant text chunks and generates answers using gpt-4o-mini with a temperature of 0 for highly factual, consistent outputs.
- Source Attribution: Automatically formats and returns the exact source filename and 1-indexed page number alongside the answer so users can verify claims.
- Web UI: Provides a straightforward, two-tab Gradio interface that separates the document ingestion setup from the querying environment.

### Installation
1. Clone and Environment Setup
```{bash}
git clone https://github.com/eliotjmartin/ai_pdf_assistant.git
cd pdf_ai_assistant
python -m venv myenv
myenv\Scripts\activate  
```
2. Install Dependencies
```{bash}
pip install --upgrade pip
pip install -r requirements.txt
```

Known Installation Issue: Greenlet / C++ Error
If you are on Windows using Python 3.9, the installation of langchain-community may fail while building the greenlet wheel with a "Microsoft Visual C++ 14.0 or greater is required" error.

I fixed this by installing full Microsoft C++ Build Tools (Click [here](https://visualstudio.microsoft.com/visual-cpp-build-tools/) to view build tools). 

3. Environment Variables

Create a .env file in the root directory:
```
OPENAI_API_KEY=your_openai_key
PINECONE_API_KEY=your_pinecone_key
PINECONE_INDEX=rag-pdf-demo
PINECONE_ENV=us-east-1
OPENAI_MODEL=gpt-4o-mini
```

### Usage
Start the app:

```{bash}
python app/app.py
```
- Ingest: Navigate to the "Setup" tab and upload your PDF.
- Chat: Switch to the "Ask Questions" tab and query your document.

### Project Structure
- app/app.py: Main entry point and user interface
- src/ingest.py: Logic for PDF processing and vector database upserting
- src/retrieve_and_answer.py: Logic for context retrieval and LLM response generation
- src/prompts.py: Central location for all prompts used in the RAG system