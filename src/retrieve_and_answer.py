import os
from openai import OpenAI
from pinecone import Pinecone
from src.prompts import SYSTEM_PROMPT, RETRIEVAL_PROMPT_TEMPLATE

INDEX_NAME = os.getenv("PINECONE_INDEX")  # the name of the database
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# initialize native clients
openai_client = OpenAI()
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

def retrieve_and_answer(question: str):
    index = pc.Index(INDEX_NAME)

    # takes question, turns it into a vector
    embed_response = openai_client.embeddings.create(
        input=question,
        model="text-embedding-3-small"
    )
    query_vector = embed_response.data[0].embedding

    # finds the 4 most similar chunks of text (k=4) stored in pinecone db
    search_results = index.query(
        vector=query_vector,
        top_k=4,
        include_metadata=True # tells Pinecone to return the actual text chunks
    )
    
    docs = search_results.matches

    # extract raw text chunks and sources from Pinecone metadata
    raw_contexts = []
    sources = []
    
    for match in docs:
        chunk_text = match.metadata.get("text", "")
        raw_contexts.append(chunk_text)
        
        # get sources with page numbers
        src = match.metadata.get("source", "Unknown")
        page = match.metadata.get("page", "Unknown")
        sources.append((src, page))
    
    # deduplicate sources
    sources = list(set(sources))

    # combine text chunks
    context_text = "\n\n".join(raw_contexts)
    
    # .format() replaces placeholders with below values
    prompt = RETRIEVAL_PROMPT_TEMPLATE.format(
        context=context_text,
        question=question
    )

    # call openai's chat completions API to write a human answer
    response = openai_client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT}, # define behavior rules
            {"role": "user", "content": prompt}  # the actual task
        ],
        temperature=0 # sets the "creativity" level. setting to zero ensures factual, consistent answers
    )

    # extract text of the reply from OpenAI response
    answer = response.choices[0].message.content
    
    # formats filenames found with page numbers
    source_list = []
    for src, page in sources:
        if isinstance(page, float):
            page = round(page) 
            source = f"- {src} (page {page + 1})" # 0 based indexing
        elif isinstance(page, int):
            source = f"- {src} (page {page + 1})" # 0 based indexing
        else:
            source = f"- {src}"
        source_list.append(source)
    
    source_list_string = "\n".join(source_list) # join sources with newline
    
    # return the raw list of strings for Ragas evaluation
    return answer, f"Sources:\n{source_list_string}", raw_contexts