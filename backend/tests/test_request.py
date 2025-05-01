import requests
import json

url = "http://localhost:8000/recommend"
payload = {"query": "software developer python skills"}
headers = {"Content-Type": "application/json"}

response = requests.post(url, data=json.dumps(payload), headers=headers)
print(response.status_code)
print(json.dumps(response.json(), indent=2))