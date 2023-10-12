import pickle
def predict(test, dv, model):
    X = dv.transform([test])
    y_pred = model.predict_proba(X)[0, 1]
    return y_pred

if __name__ == "__main__":
    with open('dv.bin','rb') as f:
        dv = pickle.load(f)
    with open('model1.bin','rb') as g:
        model = pickle.load(g)
    test = {"job": "retired", "duration": 445, "poutcome": "success"}
    print(round(predict(test,dv,model),3))
