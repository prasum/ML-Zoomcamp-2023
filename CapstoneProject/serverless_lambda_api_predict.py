import requests
import pandas as pd
test = pd.read_csv('eval_set.csv')
client = test.to_dict(orient='records')
res = requests.post(' https://sls2oq7vo6.execute-api.ap-south-1.amazonaws.com/predict', json=client[0])
if res.ok:
    print(res.json())