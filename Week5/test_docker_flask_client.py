import requests
client = {"job": "retired", "duration": 445, "poutcome": "success"}
res = requests.post('http://127.0.0.1:9696/predict', json=client)
if res.ok:
    print(res.json())