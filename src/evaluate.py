import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics import Faithfulness, AnswerRelevancy
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import OpenAIEmbeddings

load_dotenv()

from src.retrieve_and_answer import retrieve_and_answer

def get_ragas_scorer():
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # llm judge
    evaluator_llm = llm_factory("gpt-4o-mini", client=client)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    evaluator_embeddings = LangchainEmbeddingsWrapper(embeddings)
    
    # metrics
    metrics = [
        # test if the answer is supported by the retrieved context (test hallucination)
        Faithfulness(llm=evaluator_llm),
        # test if the answer actually address the question (test irrelevance)
        AnswerRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings)
    ]
    return metrics

def run_eval(question):
    # my rag response
    answer, _, raw_contexts = retrieve_and_answer(question)
    
    # wrap into a dataset
    data = {
        "question": [question],
        "answer": [answer],
        "contexts": [raw_contexts],
    }
    dataset = Dataset.from_dict(data)
    metrics = get_ragas_scorer()
    results = evaluate(dataset, metrics=metrics)
    return results

if __name__ == "__main__":
    query = "What is multi headed attention?"
    results = run_eval(query)
    print(results)