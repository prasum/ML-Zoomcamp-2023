import requests
client = {"job": "unknown", "duration": 270, "poutcome": "failure"}
res = requests.post('http://127.0.0.1:8080/predict', json=client)
if res.ok:
    print(res.json())