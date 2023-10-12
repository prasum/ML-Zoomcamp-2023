import pickle
from flask import Flask
from flask import request
from flask import jsonify


def create_app():
    with open('dv.bin', 'rb') as f:
        dv = pickle.load(f)
    with open('model1.bin', 'rb') as g:
        model = pickle.load(g)

    app = Flask('Credit Scoring')

    @app.route('/predict', methods=['POST'])
    def predict():
        client = request.get_json()
        X = dv.transform([client])
        y_pred = model.predict_proba(X)[0, 1]
        result = {
            'credit_probability': round(y_pred, 3)
        }
        return jsonify(result)

    return app
