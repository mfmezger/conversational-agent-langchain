"""The gui for the semantic search."""
import requests
import streamlit as st

# create a input field for the query
query = st.text_input("Query")

# create a select box to choose  the number of answers from 1-10
amount = st.selectbox("Number of answers", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# create a button to start the search
if st.button("Search", key="search"):
    # sent the query to the backend using the rest api
    response = requests.post("http://qdrant:8001/semantic/search", json={"query": query, "token": st.session_state.api_key, "llm_backend": "aa", "amount": amount})

    # extract the data from the response
    data = response.json()

    results = [
        {
            "text": d["text"],
            "score": d["score"],
            "page": d["page"],
            "source": d["source"],
        }
        for d in data
    ]

    # display the documents in a list view with metadata that are extandable
