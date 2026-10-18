from unittest.mock import Mock, patch

import pytest
import requests

from ragas.metrics import FactualCorrectness, ResponseRelevancy
from ragas.metrics.base import ModeMetric

from utils import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_LLM_MODEL,
    DEFAULT_RAG_API_URL,
    build_embeddings_wrapper,
    embedding_model_name,
    get_api_response,
    llm_model_name,
    metric_threshold,
    rag_api_url,
)


def test_default_rag_api_url_and_model(monkeypatch):
    monkeypatch.delenv("RAG_API_URL", raising=False)
    monkeypatch.delenv("RAGAS_LLM_MODEL", raising=False)
    monkeypatch.delenv("RAGAS_EMBEDDING_MODEL", raising=False)
    assert rag_api_url() == DEFAULT_RAG_API_URL
    assert llm_model_name() == DEFAULT_LLM_MODEL
    assert embedding_model_name() == DEFAULT_EMBEDDING_MODEL
    assert DEFAULT_EMBEDDING_MODEL == "intfloat/multilingual-e5-large-instruct"


def test_rag_api_url_and_model_are_configurable(monkeypatch):
    monkeypatch.setenv("RAG_API_URL", "https://example.test/rag")
    monkeypatch.setenv("RAGAS_LLM_MODEL", "example/model")
    monkeypatch.setenv("RAGAS_EMBEDDING_MODEL", "example/embedding")
    assert rag_api_url() == "https://example.test/rag"
    assert llm_model_name() == "example/model"
    assert embedding_model_name() == "example/embedding"


@patch("utils.LangchainEmbeddingsWrapper")
@patch("utils.OpenAIEmbeddings")
def test_build_embeddings_wrapper_uses_configured_model(mock_openai_embeddings, mock_wrapper, monkeypatch):
    monkeypatch.setenv("RAGAS_EMBEDDING_MODEL", "intfloat/multilingual-e5-large-instruct")
    inner = Mock(name="openai_embeddings")
    wrapped = Mock(name="ragas_wrapper")
    mock_openai_embeddings.return_value = inner
    mock_wrapper.return_value = wrapped

    result = build_embeddings_wrapper()

    mock_openai_embeddings.assert_called_once_with(model="intfloat/multilingual-e5-large-instruct")
    mock_wrapper.assert_called_once_with(inner)
    assert result is wrapped


def test_response_relevancy_receives_configured_embedding():
    configured_embeddings = object()
    metric = ResponseRelevancy(llm=None, embeddings=configured_embeddings)
    assert metric.embeddings is configured_embeddings


def test_ragas_evaluate_emits_mode_metric_key_for_factual_correctness():
    relevancy = ResponseRelevancy()
    factual = FactualCorrectness()
    assert not isinstance(relevancy, ModeMetric)
    assert isinstance(factual, ModeMetric)
    assert relevancy.name == "answer_relevancy"
    assert f"{factual.name}(mode={factual.mode})" == "factual_correctness(mode=f1)"


def test_metric_threshold_defaults_match_historical_poc(monkeypatch):
    for env_name in (
        "RAGAS_THRESHOLD_CONTEXT_PRECISION",
        "RAGAS_THRESHOLD_CONTEXT_RECALL",
        "RAGAS_THRESHOLD_FAITHFULNESS",
        "RAGAS_THRESHOLD_HALLUCINATION",
        "RAGAS_THRESHOLD_ANSWER_RELEVANCY",
        "RAGAS_THRESHOLD_FACTUAL_CORRECTNESS",
        "RAGAS_THRESHOLD_CONTEXTUAL_RELEVANCY",
        "RAGAS_THRESHOLD_CONTEXTUAL_PRECISION",
        "RAGAS_THRESHOLD_CONTEXTUAL_RECALL",
    ):
        monkeypatch.delenv(env_name, raising=False)

    assert metric_threshold("context_precision") == 0.8
    assert metric_threshold("context_recall") == 0.7
    assert metric_threshold("faithfulness") == 0.8
    assert metric_threshold("hallucination") == 0.8
    assert metric_threshold("answer_relevancy") == 0.8
    assert metric_threshold("factual_correctness") == 0.8
    assert metric_threshold("contextual_relevancy") == 0.8
    assert metric_threshold("contextual_precision") == 0.8
    assert metric_threshold("contextual_recall") == 0.7


def test_metric_threshold_is_configurable(monkeypatch):
    monkeypatch.setenv("RAGAS_THRESHOLD_CONTEXT_PRECISION", "0.9")
    assert metric_threshold("context_precision") == 0.9


@patch("utils.requests.post")
def test_get_api_response_posts_question_to_configured_url(mock_post, monkeypatch):
    monkeypatch.setenv("RAG_API_URL", "https://example.test/rag")
    mock_response = Mock()
    mock_response.json.return_value = {"answer": "23", "retrieved_docs": []}
    mock_post.return_value = mock_response

    result = get_api_response({"question": "How many articles?"})

    mock_post.assert_called_once_with(
        url="https://example.test/rag",
        json={"question": "How many articles?", "chat_history": []},
        timeout=30,
    )
    mock_response.raise_for_status.assert_called_once()
    assert result == {"answer": "23", "retrieved_docs": []}


@patch("utils.requests.post")
def test_get_api_response_raises_for_http_error(mock_post):
    mock_response = Mock()
    mock_response.raise_for_status.side_effect = requests.HTTPError("500")
    mock_post.return_value = mock_response

    with pytest.raises(requests.HTTPError):
        get_api_response({"question": "How many articles?"})
