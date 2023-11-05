import joblib
from flask import Flask
from flask import request
from flask import jsonify
import pandas as pd

def create_app():
    model = joblib.load('model.pkl')
    app = Flask('Titanic Survival')
    @app.route('/predict', methods=['POST'])
    def predict():
        client = request.get_json()
        X = pd.DataFrame.from_dict([client])
        y_pred = model.predict(X)[0]
        result = {
            'Survived': "Yes" if y_pred == 1 else "No"
        }
        return jsonify(result)

    return app
