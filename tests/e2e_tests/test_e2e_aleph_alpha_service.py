"""Test the aleph alpha service."""
import os

from loguru import logger

from agent.backend.aleph_alpha_service import (
    embedd_documents_aleph_alpha,
    explain_qa,
    process_documents_aleph_alpha,
    qa_aleph_alpha,
    search_documents_aleph_alpha,
)


def test_embedd_documents_aleph_alpha() -> None:
    """Test that embedd_documents_aleph_alpha does not raise an error."""
    # assert that it does not raise an error
    token = os.getenv("ALEPH_ALPHA_API_KEY")
    assert token is not None
    embedd_documents_aleph_alpha("tests/resources/", token)


def test_search() -> None:
    """Test that embedd_documents_aleph_alpha does not raise an error."""
    # assert that it does not raise an error
    token = os.getenv("ALEPH_ALPHA_API_KEY")
    assert token is not None
    docs = search_documents_aleph_alpha(aleph_alpha_token=token, query="What is Attention?", amount=3, threshold=0.1)
    assert docs is not None
    assert len(docs) > 0
    assert len(docs) == 3


def test_qa() -> None:
    """Test the QA functionality."""
    token = os.getenv("ALEPH_ALPHA_API_KEY")
    assert token is not None
    docs = search_documents_aleph_alpha(aleph_alpha_token=token, query="What ist Attention?", amount=1)
    answer, prompt, meta_data = qa_aleph_alpha(aleph_alpha_token=token, documents=docs, query="What are Attentions?")
    assert answer is not None
    assert prompt is not None
    assert meta_data is not None
    assert len(answer) > 0
    assert len(prompt) > 0


def test_process_documents_aleph_alpha() -> None:
    """Test the process_documents_aleph_alpha function."""
    token = os.getenv("ALEPH_ALPHA_API_KEY")
    assert token is not None
    answers = process_documents_aleph_alpha(folder="tests/resources", token=token, type="invoice")

    assert answers is not None
    assert len(answers) > 0
    assert answers[0] is not None
    assert answers[1] is not None


def test_explain_qa() -> None:
    """Test Explain  QA."""
    token = os.getenv("ALEPH_ALPHA_API_KEY")

    assert token is not None

    docs = search_documents_aleph_alpha(aleph_alpha_token=token, query="Was sind meine Vorteile?", amount=1)

    explanation, score, text, answer, meta_data = explain_qa(aleph_alpha_token=token, document=docs, explain_threshold=0.7, query="Was sind meine Vorteile?")

    logger.info(f"ANSWER: {answer}")
    logger.info(f"EXPLANATION: {explanation}")
    logger.info(f"METADATA: {meta_data}")
    logger.info(f"SCORE: {score}")
    logger.info(f"TEXT: {text}")
