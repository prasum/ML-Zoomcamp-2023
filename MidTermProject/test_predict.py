import joblib
import pandas as pd

def predict(X, model):
    y_pred = model.predict(X)[0]
    return "Yes" if y_pred == 1 else "No"

if __name__ == "__main__":
    model = joblib.load('model.pkl')
    test = pd.read_csv('test_one_record.csv')
    print(predict(test,model))