# AI PDF Assistant 
A Retrieval-Augmented Generation (RAG) application that allows users to upload PDF documents and engage in a factual Q&A session with the content. 

### Features
- Ingestion: Uses PyPDFLoader for page-level metadata and tiktoken for token-based chunking.
- Vector Search: Integrated with Pinecone for low-latency similarity searching.
- LLM Integration: Powered by OpenAI's gpt-4o-mini for cost-effective, intelligent responses.
- Web UI: A clean, two-tab interface built with Gradio

### Tech Stack
- Language: Python 3.9+
- Frameworks: LangChain, Gradio
- Database: Pinecone 
- Models: text-embedding-3-small (Embeddings), gpt-4o-mini (Chat)

### Installation
1. Clone & Environment Setup
```{bash}
git clone https://github.com/eliotjmartin/ai_pdf_assistant.git
cd llm-rag-lab
python -m venv myenv
myenv\Scripts\activate  
```
2. Install Dependencies
```{bash}
pip install --upgrade pip
pip install gradio python-dotenv pinecone langchain-openai langchain-pinecone langchain-text-splitters langchain-community pypdf openai
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
python app/gradio_demo.py
```
- Ingest: Navigate to the "Setup" tab and upload your PDF.
- Chat: Switch to the "Ask Questions" tab and query your document.

### Project Structure
- app/gradio_demo.py: Main entry point and user interface
- src/ingest.py: Logic for PDF processing and vector database upserting
- src/retrieve_and_answer.py: Logic for context retrieval and LLM response generation