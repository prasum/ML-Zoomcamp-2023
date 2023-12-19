import joblib
from flask import Flask
from flask import request
from flask import jsonify
import pandas as pd
import serverless_wsgi

model = joblib.load('model.pkl')

app = Flask('Mercedes Car Manufacture Time')

@app.route('/predict', methods=['POST'])
def predict():
    client = request.get_json()
    X = pd.DataFrame.from_dict([client])
    y_pred = model.predict(X)[0]
    result = {
        'Predicted Time': y_pred
    }
    return jsonify(result)

def handler(event, context):
    return serverless_wsgi.handle_request(app, event, context)