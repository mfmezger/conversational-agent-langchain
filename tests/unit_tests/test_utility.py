"""Tests for the utility functions."""
import pytest
from langchain_core.documents import Document
from agent.utils.utility import (
    combine_text_from_list,
    load_prompt_template,
    convert_qdrant_result_to_retrieval_results,
    create_tmp_folder,
    format_docs_for_citations,
)
from agent.data_model.internal_model import RetrievalResults


def test_combine_text_from_list():
    """Test that combine_text_from_list returns the correct string."""
    input_list = ["a", "b", "c"]
    result = combine_text_from_list(input_list)
    assert result == "a\nb\nc"

def test_combine_text_from_list_with_non_string():
    """Test that combine_text_from_list raises a TypeError if the list contains a non-string."""
    input_list = ["a", "b", 1]
    with pytest.raises(TypeError):
        combine_text_from_list(input_list)

def test_load_prompt_template():
    """Test that load_prompt_template returns the correct prompt template."""
    prompt_template = load_prompt_template(prompt_name="cohere_chat.j2", task="chat")
    assert "Context" in prompt_template.template
    assert "Question" in prompt_template.template

def test_load_prompt_template_not_found():
    """Test that load_prompt_template raises a FileNotFoundError if the prompt template is not found."""
    with pytest.raises(FileNotFoundError):
        load_prompt_template(prompt_name="not_found.j2", task="chat")

def test_convert_qdrant_result_to_retrieval_results():
    """Test that convert_qdrant_result_to_retrieval_results returns the correct list of tuples."""
    docs = [
        (Document(page_content="This is a test document.", metadata={"source": "test"}), 0.9),
        (Document(page_content="This is another test document.", metadata={"source": "test"}), 0.8),
    ]
    result = convert_qdrant_result_to_retrieval_results(docs)
    assert len(result) == 2
    assert isinstance(result[0], RetrievalResults)
    assert result[0].document == "This is a test document."
    assert result[0].score == 0.9
    assert result[0].metadata == {"source": "test"}

def test_create_tmp_folder():
    """Test that create_tmp_folder returns a valid directory name."""
    import os
    tmp_dir = create_tmp_folder()
    assert os.path.isdir(tmp_dir)
    os.rmdir(tmp_dir)

def test_format_docs_for_citations():
    """Test that format_docs_for_citations returns the correct string."""
    docs = [
        Document(page_content="This is a test document.", metadata={"source": "test"}),
        Document(page_content="This is another test document.", metadata={"source": "test"}),
    ]
    result = format_docs_for_citations(docs)
    assert result == "<doc id='0'>This is a test document.</doc>\n<doc id='1'>This is another test document.</doc>"
