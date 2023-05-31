# Conversational Agent


## Project Description
This project is a conversational agent that uses Aleph Alpha and OpenAI Large Language Models to generate responses to user queries. The agent also includes a vector database and a REST API built with FastAPI.

Features
- Uses Aleph Alpha and OpenAI Large Language Models to generate responses to user queries.
- Includes a vector database to store and retrieve information.
- Provides a REST API built with FastAPI for easy integration with other applications.

## Components

It ueses Langchain, FastAPI and a Vectordatabase.

## Deployment

If you want to use a default token for the LLM Provider you need to create a .env file. Do this by copying the .env-template file and add the necessary api keys.

If you are working in an envoironment with internet connection the easiest way is to use this command:

```bash
docker compose -f docker-compose-hub.yml up
```

This will pull the image from docker hub and run it. Instead of building it on your local machine.

If you want to build the image on your local machine you can use this command:

```bash
docker compose up
```

## Development Backend

To run the Backend use this command in the root directory:

```bash
poetry run uvicorn agent.api:app --reload
```

To run the tests you can use this command:

```bash
poetry run coverage run -m pytest -o log_cli=true -vvv tests
```

## Development Frontend
