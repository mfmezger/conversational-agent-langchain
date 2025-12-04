import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document

# Patch reranker and retriever before importing Graph to prevent external connections
with patch("agent.utils.reranker.get_reranker", return_value=lambda docs, query: docs):
    from agent.backend.graph import Graph, AgentState, Grade

# --- Tests for Graph Class ---

@pytest.fixture
def graph_instance():
    with patch("agent.backend.graph.get_reranker") as mock_reranker, \
         patch("agent.backend.graph.ChatCohere") as mock_cohere:
        mock_reranker.return_value = lambda docs, query: docs
        mock_cohere.return_value = MagicMock()
        yield Graph()

def test_route_to_retriever_single_message(graph_instance):
    state = {"messages": [HumanMessage(content="hi")]}
    result = graph_instance.route_to_retriever(state)
    assert result == "retriever"

def test_route_to_retriever_multi_message(graph_instance):
    state = {"messages": [HumanMessage(content="hi"), AIMessage(content="hello"), HumanMessage(content="bye")]}
    result = graph_instance.route_to_retriever(state)
    assert result == "retriever_with_chat_history"

def test_get_chat_history(graph_instance):
    messages = [
        HumanMessage(content="hi"),
        AIMessage(content="hello"),
        HumanMessage(content="bye")
    ]
    history = graph_instance.get_chat_history(messages)
    assert len(history) == 3
    assert history[0]["role"] == "human"
    assert history[1]["role"] == "ai"

def test_route_to_response_synthesizer_default(graph_instance):
    config = {"configurable": {"model_name": "openai_gpt_3_5_turbo"}}
    result = graph_instance.route_to_response_synthesizer({}, config)
    assert result == "response_synthesizer"

def test_route_to_response_synthesizer_cohere(graph_instance):
    config = {"configurable": {"model_name": "cohere_command"}}
    result = graph_instance.route_to_response_synthesizer({}, config)
    assert result == "response_synthesizer_cohere"

@patch("agent.backend.graph.get_retriever")
def test_retrieve_documents(mock_get_retriever, graph_instance):
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = [Document(page_content="doc1")]
    mock_get_retriever.return_value = mock_retriever

    state = {"messages": [HumanMessage(content="query")]}
    config = {"metadata": {"collection_name": "test_coll"}}

    result = graph_instance.retrieve_documents(state, config)

    assert result["query"] == "query"
    assert len(result["documents"]) == 1
    assert result["documents"][0].page_content == "doc1"
    mock_get_retriever.assert_called_with(collection_name="test_coll", k=graph_instance.cfg.retrieval_k)

@patch("agent.backend.graph.get_retriever")
def test_retrieve_documents_with_chat_history(mock_get_retriever, graph_instance):
    # Mock the retriever
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = [Document(page_content="doc1")]
    mock_get_retriever.return_value = mock_retriever

    # Mock the LLM and the chain
    graph_instance.llm = MagicMock()

    # We need to mock the chain that is constructed inside the method:
    # condense_question_chain = (condense_queston_prompt | model | StrOutputParser()).with_config(...)
    # retriever_with_condensed_question = condense_question_chain | retriever

    # Since we can't easily mock the internal chain construction without patching the classes,
    # we will patch the classes involved in the chain construction to return mocks that we can control.

    with patch("agent.backend.graph.PromptTemplate.from_template") as mock_prompt, \
         patch("agent.backend.graph.StrOutputParser") as mock_parser:

        # Setup the mock chain components
        mock_prompt_instance = MagicMock()
        mock_prompt.return_value = mock_prompt_instance

        mock_parser_instance = MagicMock()
        mock_parser.return_value = mock_parser_instance

        # The chain is: prompt | model | parser
        # We mock the pipe operations
        mock_chain_step1 = MagicMock()
        mock_prompt_instance.__or__.return_value = mock_chain_step1

        mock_chain_step2 = MagicMock()
        mock_chain_step1.__or__.return_value = mock_chain_step2

        mock_condense_question_chain = MagicMock()
        mock_chain_step2.with_config.return_value = mock_condense_question_chain

        # The final chain is: condense_question_chain | retriever
        mock_final_chain = MagicMock()
        mock_condense_question_chain.__or__.return_value = mock_final_chain

        mock_final_chain.invoke.return_value = [Document(page_content="doc1")]

        state = {
            "messages": [
                HumanMessage(content="hi"),
                AIMessage(content="hello"),
                HumanMessage(content="followup")
            ]
        }
        config = {"metadata": {"collection_name": "test_coll"}}

        result = graph_instance.retrieve_documents_with_chat_history(state, config)

        # Verify the result
        assert result["query"] == "followup"
        assert len(result["documents"]) == 1
        assert result["documents"][0].page_content == "doc1"

        # Verify that the final chain was invoked with the expected input
        # The input to the final chain is {"question": query, "chat_history": ...}
        args, _ = mock_final_chain.invoke.call_args
        assert args[0]["question"] == "followup"
        assert len(args[0]["chat_history"]) == 2


def test_grade_documents(graph_instance):
    # Mock the LLM and structured output
    graph_instance.llm = MagicMock()
    mock_structured_model = MagicMock()
    # We need to mock the with_config return value to return the model itself or a mock that has with_structured_output
    mock_model_with_config = MagicMock()
    graph_instance.llm.with_config.return_value = mock_model_with_config
    mock_model_with_config.with_structured_output.return_value = mock_structured_model

    # Mock the chain
    mock_chain = MagicMock()
    # We need to mock the prompt | structured_model chain
    # Since we can't easily mock the pipe of a locally created PromptTemplate,
    # we will patch PromptTemplate to return a mock that supports piping

    with patch("agent.backend.graph.PromptTemplate") as mock_prompt_cls:
        mock_prompt_instance = MagicMock()
        mock_prompt_cls.return_value = mock_prompt_instance

        mock_prompt_instance.__or__.return_value = mock_chain

        # Case 1: Relevant documents
        mock_chain.invoke.return_value = Grade(is_relevant=True)
        state = {
            "documents": [Document(page_content="doc1")],
            "query": "test query",
            "retry_count": 0
        }
        config = {"configurable": {"model_name": "gemini"}}

        result = graph_instance.grade_documents(state, config)
        assert result == "response_synthesizer"

        # Case 2: Irrelevant documents
        mock_chain.invoke.return_value = Grade(is_relevant=False)
        state["retry_count"] = 0
        result = graph_instance.grade_documents(state, config)
        assert result == "rewrite_query"

        # Case 3: Irrelevant documents but max retries reached
        mock_chain.invoke.return_value = Grade(is_relevant=False)
        state["retry_count"] = 2
        result = graph_instance.grade_documents(state, config)
        assert result == "response_synthesizer"


def test_rewrite_query(graph_instance):
    # Mock the LLM
    graph_instance.llm = MagicMock()
    mock_model_with_config = MagicMock()
    graph_instance.llm.with_config.return_value = mock_model_with_config

    # Mock the chain
    mock_chain = MagicMock()

    with patch("agent.backend.graph.PromptTemplate") as mock_prompt_cls, \
         patch("agent.backend.graph.StrOutputParser") as mock_parser:

        mock_prompt_instance = MagicMock()
        mock_prompt_cls.return_value = mock_prompt_instance

        mock_parser_instance = MagicMock()
        mock_parser.return_value = mock_parser_instance

        # Chain: prompt | model | parser
        # 1. prompt | model -> step1
        mock_chain_step1 = MagicMock()
        mock_prompt_instance.__or__.return_value = mock_chain_step1

        # 2. step1 | parser -> chain
        # Note: StrOutputParser() is the argument to __or__
        mock_chain_step1.__or__.return_value = mock_chain

        mock_chain.invoke.return_value = "rewritten query"

        state = {
            "query": "original query",
            "retry_count": 0
        }

        result = graph_instance.rewrite_query(state)

        assert result["query"] == "rewritten query"
        assert result["retry_count"] == 1


def test_generate_response(graph_instance):
    # Mock the LLM
    graph_instance.llm = MagicMock()

    # Mock the chain
    mock_chain = MagicMock()

    with patch("agent.backend.graph.ChatPromptTemplate.from_messages") as mock_prompt_cls:
        mock_prompt_instance = MagicMock()
        mock_prompt_cls.return_value = mock_prompt_instance

        # Chain: prompt | model
        mock_chain_step1 = MagicMock()
        mock_prompt_instance.__or__.return_value = mock_chain_step1

        mock_chain_step1.invoke.return_value = AIMessage(content="synthesized response")

        state = {
            "query": "test query",
            "documents": [Document(page_content="doc1")],
            "messages": [HumanMessage(content="test query")]
        }

        # Test generate_response_default
        result = graph_instance.generate_response_default(state)
        assert len(result["messages"]) == 1
        assert result["messages"][0].content == "synthesized response"

        # Test generate_response_cohere
        # We need to mock the bind method
        mock_bound_model = MagicMock()
        graph_instance.llm.bind.return_value = mock_bound_model
        mock_prompt_instance.__or__.return_value = mock_chain # Reset chain return for new call
        mock_chain.invoke.return_value = AIMessage(content="cohere response")

        # Note: generate_response_cohere calls generate_response which constructs a NEW chain
        # So we need to ensure our mocks work for the second call too.
        # The prompt template is different, but we mocked the class so it returns the same mock instance.

        result = graph_instance.generate_response_cohere(state)
        assert len(result["messages"]) == 1
        # Since we mocked the chain invoke to return "synthesized response" initially,
        # and we didn't change what mock_chain_step1.invoke returns, it will still be "synthesized response"
        # unless we update it.
        # However, generate_response_cohere uses `model` which is `self.llm.bind(...)`.
        # So `prompt | model` will use the bound model.
        # `mock_prompt_instance | mock_bound_model` -> mock_chain_cohere

        mock_chain_cohere = MagicMock()
        mock_prompt_instance.__or__.side_effect = [mock_chain_step1, mock_chain_cohere]
        mock_chain_cohere.invoke.return_value = AIMessage(content="cohere response")

        # We need to reset the side_effect or re-run the default test first to set it up correctly.
        # Let's just mock the invoke return value to be dynamic or just check it returns an AIMessage.

    # Re-doing the test with cleaner separation

    # Test Default
    with patch("agent.backend.graph.ChatPromptTemplate.from_messages") as mock_prompt_cls:
        mock_prompt_instance = MagicMock()
        mock_prompt_cls.return_value = mock_prompt_instance

        mock_chain = MagicMock()
        mock_prompt_instance.__or__.return_value = mock_chain
        mock_chain.invoke.return_value = AIMessage(content="default response")

        result = graph_instance.generate_response_default(state)
        assert result["messages"][0].content == "default response"

    # Test Cohere (now uses ChatCohere directly with documents passed to invoke)
    with patch("agent.backend.graph.ChatPromptTemplate.from_messages") as mock_prompt_cls:
        mock_prompt_instance = MagicMock()
        mock_prompt_cls.return_value = mock_prompt_instance

        # Mock the cohere_llm
        mock_cohere_llm = MagicMock()
        graph_instance.cohere_llm = mock_cohere_llm

        mock_chain = MagicMock()
        mock_prompt_instance.__or__.return_value = mock_chain
        mock_chain.invoke.return_value = AIMessage(content="cohere response")

        result = graph_instance.generate_response_cohere(state)
        assert result["messages"][0].content == "cohere response"

        # Verify documents were passed in Cohere's expected format
        call_args = mock_chain.invoke.call_args
        assert "documents" in call_args.kwargs
        docs = call_args.kwargs["documents"]
        assert len(docs) == 1
        assert docs[0]["text"] == "doc1"

# --- Tests for RAG Routes ---

@pytest.fixture
def client():
    """Lazily create TestClient to avoid import-time initialization."""
    from fastapi.testclient import TestClient
    from agent.api import app
    return TestClient(app)

@pytest.mark.asyncio
@patch("agent.routes.rag.graph")
async def test_rag_question_answer(mock_graph, client):
    # Mock the graph.ainvoke method
    mock_graph.with_config.return_value.ainvoke = AsyncMock(return_value={
        "documents": [Document(page_content="doc1", metadata={"source": "test"})],
        "messages": [AIMessage(content="The answer")]
    })

    payload = {
        "messages": [{"role": "user", "content": "question"}],
        "collection_name": "test"
    }

    response = client.post("/rag/", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "The answer"
    assert len(data["meta_data"]) == 1
    assert data["meta_data"][0]["document"][0] == "doc1"

@pytest.mark.asyncio
@patch("agent.routes.rag.graph")
async def test_rag_stream(mock_graph, client):
    # Mock the graph.astream_events method
    async def mock_stream(*args, **kwargs):
        yield {
            "event": "on_chain_start",
            "name": "retriever",
            "data": {}
        }
        yield {
            "event": "on_chain_end",
            "name": "retriever",
            "data": {"output": {"documents": [Document(page_content="doc1")]}}
        }
        yield {
            "event": "on_chat_model_start",
            "name": "model",
            "data": {}
        }
        yield {
            "event": "on_chat_model_stream",
            "metadata": {"langgraph_node": "response_synthesizer"},
            "data": {"chunk": AIMessage(content="chunk1")}
        }
        yield {
            "event": "on_chain_end",
            "name": "LangGraph",
            "data": {"output": {"documents": [Document(page_content="doc1", metadata={"s": "1"})]}}
        }

    mock_graph.with_config.return_value.astream_events = mock_stream

    payload = {
        "messages": [{"role": "user", "content": "question"}],
        "collection_name": "test"
    }

    with client.stream("POST", "/rag/stream", json=payload) as response:
        assert response.status_code == 200
        lines = list(response.iter_lines())
        assert len(lines) > 0
        # Verify we got some expected events
        assert "Starting request..." in lines[0]
        # We can check for specific content in the lines
