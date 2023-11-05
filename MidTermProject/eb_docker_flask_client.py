import requests
import pandas as pd
test = pd.read_csv('test_one_record.csv')
client = test.to_dict(orient='records')
res = requests.post('http://titanic-survivor-prediction-env.eba-5d3pzxdd.ap-south-1.elasticbeanstalk.com/predict', json=client[0])
if res.ok:
    print(res.json())