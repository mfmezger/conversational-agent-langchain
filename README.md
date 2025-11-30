![conv](https://socialify.git.ci/mfmezger/conversational-agent-langchain/image?description=1&font=Inter&language=1&name=1&owner=1&pattern=Charlie%20Brown&stargazers=1&theme=Dark)

[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>

# Conversational Agent
This is a Rest-Backend for a Conversational Agent, that allows you to embed Documents, search for them using Semantic Search, to QA based on Documents and do document processing with Large Language Models.


## Latest Changes
- Changed to LiteLLM to allow for every model provider. Default is Cohere for Embedding and Google AI Studio Gemini for Generation.


## Table of Contents
- [Conversational Agent](#conversational-agent)
  - [Latest Changes](#latest-changes)
  - [Table of Contents](#table-of-contents)
  - [LLMs and Backend Providers](#llms-and-backend-providers)
  - [Quickstart](#quickstart)
  - [Project Description](#project-description)
  - [Semantic Search](#semantic-search)
    - [Hybrid Search](#hybrid-search)
  - [Architecture](#architecture)
  - [Installation \& Development Backend](#installation--development-backend)
    - [Load Demo Data](#load-demo-data)
  - [Development Frontend](#development-frontend)
    - [Mypy](#mypy)
  - [Vector Database](#vector-database)
  - [Qdrant API Key](#qdrant-api-key)
  - [Bulk Ingestion](#bulk-ingestion)
  - [Testing the API](#testing-the-api)
  - [Star History](#star-history)


## LLMs and Backend Providers

I have decided to stop creating different services for different provider and switchting to LiteLLM which allows to use basically every provider you want.

Some providers i would recommend are:

- [Cohere](https://cohere.com/) Awesome models and great free tier.
- [Ollama](https://ollama.com/) If you want to keep your data your data.
- [Google AI Studio](https://aistudio.google.com) The Google Integration that is not really suited for enterprise but perfect for everybody else.

> [!NOTE]
> The `EmbeddingManagement` class in `src/agent/backend/services/embedding_management.py` contains placeholders for Google and OpenAI embedding providers. These are intended as extension points for you to implement if you wish to use these specific providers directly.


## Quickstart
To run the complete system with docker use this command:

```bash
git clone https://github.com/mfmezger/conversational-agent-langchain.git
cd conversational-agent-langchain
```


Create a .env file from the `template.env` and set the necessary API Keys.
Absolutely necessary are:
- GEMINI_API_KEY
- COHERE_API_KEY



Then start the system with
```bash
  docker compose up -d
```

Then go to http://127.0.0.1:8001/docs or http://127.0.0.1:8001/redoc to see the API documentation.

Frontend: localhost:8501
Qdrant Dashboard: localhost:6333/dashboard




## Project Description
This project is a conversational rag agent that uses Google Gemini Large Language Models to generate responses to user queries. The agent also includes a vector database and a REST API built with FastAPI.

Features
- Uses Large Language Models to generate responses to user queries.
- Includes a vector database to store and retrieve information.
- Provides a REST API built with FastAPI for easy integration with other applications.
- Has a basic GUI.

![UI](resources/ui.png)

## Semantic Search
![Semantic Search Architecture](resources/search_flow.png)

Semantic search is an advanced search technique that aims to understand the meaning and context of a user's query, rather than matching keywords. It involves natural language processing (NLP) and machine learning algorithms to analyze and interpret user intent, synonyms, relationships between words, and the structure of content. By considering these factors, semantic search improves the accuracy and relevance of search results, providing a more intuitive and personalized user experience.

### Hybrid Search

For Hybrid Search the BM25 FastEmbed from Qdrant is used.

## Architecture
![Semantic Search Architecture](resources/Architecture.png)

## Installation & Development Backend

On Linux or Mac you need to adjust your /etc/hosts file to include the following line:

```bash
127.0.0.1 qdrant
```



First install Python Dependencies:

You need to install uv if you want to use it for syncing the requirements.lock file. [UV Installation](https://docs.astral.sh/uv/getting-started/installation/).

```bash
uv sync
```

### Load Demo Data

In src/agent/scripts use the load dummy data script to load some example data in the rag.


Start the complete system with:

```bash
docker compose up -d
```

To run the Qdrant Database local just run:

```bash
docker compose up qdrant
```


To run the Backend use this command in the root directory:

```bash
poetry run uvicorn agent.api:app --reload
```

To run the tests you can use this command:

```bash
poetry run coverage run -m pytest -o log_cli=true -vvv tests
```

## Development Frontend

To run the Frontend use this command in the root directory:

```bash
poetry run streamlit run gui.py --theme.base="dark"
```

### Mypy

mypy src/agent --explicit-package-bases

## Vector Database

Qdrant Dashboard is available at http://127.0.0.1:6333/dashboard. There you need to enter the api key.

## Qdrant API Key
To use the Qdrant API you need to set the correct parameters in the .env file.
QDRANT_API_KEY is the API key for the Qdrant API.
And you need to change it in the qdrant.yaml file in the config folder.

## Bulk Ingestion
If you want to ingest large amount of data i would recommend you use the scripts located in agent/ingestion.


## Testing the API
To Test the API i would recommend [Bruno](https://www.usebruno.com/). The API Requests are store in ConvAgentBruno folder.


## Star History

<a href="https://star-history.com/#mfmezger/conversational-agent-langchain&Date">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=mfmezger/conversational-agent-langchain&type=Date&theme=dark" />
    <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=mfmezger/conversational-agent-langchain&type=Date" />
    <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=mfmezger/conversational-agent-langchain&type=Date" />
  </picture>
</a>
