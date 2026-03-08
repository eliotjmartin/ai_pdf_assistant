import os
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

INDEX_NAME = os.getenv("PINECONE_INDEX")  # the name of the database
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
client = OpenAI()

def retrieve_and_answer(question: str):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")  # turns question into numbers
    vectorstore = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings) # points langchain to pinecone db

    # takes question, turns it into a vector, and finds the 4 most similar chunks 
    # of text (k=4) stored in pinecone db
    docs = vectorstore.similarity_search(question, k=4)
    
    # combine text chunks 
    context_text = "\n\n".join([d.page_content for d in docs])
    
    # get sources with page numbers for each doc in docs
    sources = list(set(
        (d.metadata.get("source", "Unknown"), d.metadata.get("page", "Unknown"))
        for d in docs
    ))

    # send context to the LLM to write a human answer
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            # tell the LLM it must stay within the provided context (prevent hallucination)
            {"role": "system", "content": "You are a helpful assistant. Use the provided context to answer questions. If the answer isn't in the context, say you don't know."},
            # send the extracted text and the original question to the LLM
            {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {question}"}
        ],
        temperature=0 # sets the "creativity" level. setting to zero ensures factual, consistent answers
    )

    # extract text of the reply from OpenAI response
    answer = response.choices[0].message.content
    
    # formats filenames found with page numbers
    source_list = []
    for src, page in sources:
        if isinstance(page, float):
            page=round(page) 
            source = f"- {src} (page {page + 1})" # 0 based indexing
        elif isinstance(page, int):
            source = f"- {src} (page {page + 1})" # 0 based indexing
        else:
            source = f"- {src}"
        source_list.append(source)
    
    source_list_string = "\n".join(source_list) # join sources with newline
    
    return answer, f"Sources:\n{source_list_string}"