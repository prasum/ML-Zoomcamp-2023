import joblib
import pandas as pd
import sys
sys.path.append('../')

def predict(X, model):
    y_pred = model.predict(X)[0]
    return y_pred

if __name__ == "__main__":
    model = joblib.load('model.pkl')
    test = pd.read_csv('eval_set.csv')
    print(predict(test,model))