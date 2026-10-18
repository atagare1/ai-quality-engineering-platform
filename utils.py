import json
import os
import pathlib

import requests
from langchain_openai import OpenAIEmbeddings
from ragas import SingleTurnSample
from ragas.embeddings import LangchainEmbeddingsWrapper

DEFAULT_RAG_API_URL = "https://rahulshettyacademy.com/rag-llm/ask"
DEFAULT_LLM_MODEL = "mistralai/Mixtral-8x7B-Instruct-v0.1"
DEFAULT_EMBEDDING_MODEL = "intfloat/multilingual-e5-large-instruct"
DEFAULT_REQUEST_TIMEOUT_SECONDS = 30

# Experimental POC gates only. Not production-validated.
DEFAULT_METRIC_THRESHOLDS = {
    "context_precision": 0.8,
    "context_recall": 0.7,
    "faithfulness": 0.8,
    "hallucination": 0.8,
    "answer_relevancy": 0.8,
    "factual_correctness": 0.8,
    "contextual_relevancy": 0.8,
    "contextual_precision": 0.8,
    "contextual_recall": 0.7,
}

THRESHOLD_ENV_VARS = {
    "context_precision": "RAGAS_THRESHOLD_CONTEXT_PRECISION",
    "context_recall": "RAGAS_THRESHOLD_CONTEXT_RECALL",
    "faithfulness": "RAGAS_THRESHOLD_FAITHFULNESS",
    "hallucination": "RAGAS_THRESHOLD_HALLUCINATION",
    "answer_relevancy": "RAGAS_THRESHOLD_ANSWER_RELEVANCY",
    "factual_correctness": "RAGAS_THRESHOLD_FACTUAL_CORRECTNESS",
    "contextual_relevancy": "RAGAS_THRESHOLD_CONTEXTUAL_RELEVANCY",
    "contextual_precision": "RAGAS_THRESHOLD_CONTEXTUAL_PRECISION",
    "contextual_recall": "RAGAS_THRESHOLD_CONTEXTUAL_RECALL",
}


def rag_api_url():
    return os.getenv("RAG_API_URL", DEFAULT_RAG_API_URL)


def llm_model_name():
    return os.getenv("RAGAS_LLM_MODEL", DEFAULT_LLM_MODEL)


def embedding_model_name():
    return os.getenv("RAGAS_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)


def build_embeddings_wrapper():
    openai_embeddings = OpenAIEmbeddings(model=embedding_model_name())
    return LangchainEmbeddingsWrapper(openai_embeddings)


def metric_threshold(metric_key):
    if metric_key not in DEFAULT_METRIC_THRESHOLDS:
        raise KeyError(f"Unknown metric threshold key: {metric_key}")
    env_name = THRESHOLD_ENV_VARS[metric_key]
    raw_value = os.getenv(env_name)
    if raw_value is None or raw_value == "":
        return DEFAULT_METRIC_THRESHOLDS[metric_key]
    return float(raw_value)


def read_test_data(filename, base_dir=None):
    project_directory = pathlib.Path(__file__).parent.absolute()
    directory = pathlib.Path(base_dir) if base_dir is not None else project_directory / "test-data"
    test_data_path = directory / filename
    with open(test_data_path, encoding="utf-8") as handle:
        return json.load(handle)


def get_api_response(passed_data):
    response = requests.post(
        url=rag_api_url(),
        json={
            "question": passed_data["question"],
            "chat_history": [],
        },
        timeout=DEFAULT_REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()


def extract_answer(response_data):
    if not isinstance(response_data, dict):
        raise ValueError("RAG API response must be a JSON object")
    if "answer" not in response_data:
        raise ValueError("RAG API response missing required field 'answer'")
    answer = response_data["answer"]
    if not isinstance(answer, str):
        raise ValueError("RAG API response field 'answer' must be a string")
    return answer


def extract_retrieved_contexts(response_data):
    if not isinstance(response_data, dict):
        raise ValueError("RAG API response must be a JSON object")
    docs = response_data.get("retrieved_docs")
    if docs is None:
        return []
    if not isinstance(docs, list):
        raise ValueError("retrieved_docs must be a list when present")
    contexts = []
    for doc in docs:
        if isinstance(doc, dict):
            page_content = doc.get("page_content")
            if isinstance(page_content, str) and page_content:
                contexts.append(page_content)
    return contexts


def map_rag_response(passed_data, response_data):
    return {
        "user_input": passed_data["question"],
        "response": extract_answer(response_data),
        "retrieved_contexts": extract_retrieved_contexts(response_data),
        "reference": passed_data.get("reference"),
    }


def build_single_turn_sample(passed_data, response_data):
    mapped = map_rag_response(passed_data, response_data)
    return SingleTurnSample(
        user_input=mapped["user_input"],
        response=mapped["response"],
        retrieved_contexts=mapped["retrieved_contexts"],
        reference=mapped["reference"],
    )
