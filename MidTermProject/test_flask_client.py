import requests
import pandas as pd
test = pd.read_csv('test_one_record.csv')
client = test.to_dict(orient='records')
res = requests.post('http://127.0.0.1:8080/predict', json=client[0])
if res.ok:
    print(res.json())