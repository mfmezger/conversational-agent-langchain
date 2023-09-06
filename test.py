
import requests
url = "http://localhost:8001/semantic/search"
documents = requests.post(url, json={"query": "wAS IST pYTORCH","llm_backend": "aa","token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoxMjM2LCJ0b2tlbl9pZCI6Mjk1MH0.2HQBfubZb1yx9Mi3TnbeoTY4p_0xubaT5LjMdOqeCe0",        "amount": 5,},)
documents = documents.json()
# # convert string to json
# import json
# documents = json.dumps(documents)


print(documents)