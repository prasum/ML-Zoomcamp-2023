from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
import numpy as np

def evaluate_model(model, X_tr, y_tr, X_te, y_te,evaluate=False):
    # set a fixed random seed for model weights initialization 
    # to obtain reproducible results.
    # Do not change the value.
    if isinstance(model, Pipeline):
        model.set_params(estimator__random_state=0)
    else:
        model.set_params(random_state=0)

    if not evaluate:
        model.fit(X_tr,y_tr)
        ytrainpred = model.predict(X_tr)
        R2_tr = r2_score(y_tr, ytrainpred)
    ytestpred = model.predict(X_te)
    R2_te=r2_score(y_te,ytestpred)
    return R2_tr, R2_te if not evaluate else ytestpred
	
def get_abs_corr_coef(X, y):
    """
    Compute

    Parameters
    ----------
    X : numpy.ndarray of shape (n_samples, n_features)
        Feature matrix.
    y : numpy.ndarray of shape (n_samples,)
        Target variable

    Returns
    -------
    corr_coefs : numpy.ndarray of shape (n_features,)
        Vector of absolute values of correlation coefficients
        for all features
    """
    # your code here
    corr_coefs = []
    for i in range(X.shape[1]):
        corr_coefs.append(abs(np.corrcoef(X[:, i], y)[0, 1]))
    corr_coefs = np.array(corr_coefs)
    return corr_coefs