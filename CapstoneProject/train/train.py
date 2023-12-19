import warnings
import sys
sys.path.append('../')
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from util import *

warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', 100)
import joblib


def train():
    file_path = "../Data/train_mercedes.csv"
    df = pd.read_csv(file_path)
    y = df["y"]
    X = df.drop(["ID", "y"], axis=1)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=1)
    bin_cols = X_tr.columns[X_tr.dtypes == "int64"].tolist()
    print("Number of binary features =", len(bin_cols))
    cat_cols = X_tr.columns[X_tr.dtypes == "object"].tolist()
    print("Number of categorical features =", len(cat_cols))
    print("Number of columns with any missing values:")
    print(X_tr.isnull().any(axis=0).sum())
    # TEST evaluate_model and baseline model
    res = evaluate_model(Ridge(), X_tr.iloc[:100, 10:11], y_tr[:100], X_tr.iloc[100:150, 10:11], y_tr[100:150])
    print('R2 on train:', res[0])
    print('R2 on test:', res[1])

    assert np.allclose(
        np.round(evaluate_model(Ridge(),
                                X_tr.values[:100, 8:16], y_tr.values[:100],
                                X_tr.values[100:150, 8:16], y_tr.values[100:150]), 3),
        [0.136, 0.094]
    )

    col_transformer = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ], remainder='passthrough')
    model = Pipeline([
        ('col_transformer', col_transformer),
        ('estimator', Ridge())
    ])
    # TEST baseline model
    print('Column transformers:')
    print(col_transformer.transformers)

    print('Pipeline:')
    print(model.steps[0])
    print(model.steps[1])

    # test transformers
    assert len(col_transformer.transformers) == 1, 'col_transformer should exactly 1 transformation'
    assert isinstance(col_transformer.transformers[0][1], OneHotEncoder), 'col_transformer should use OneHotEncoder'
    assert set(col_transformer.transformers[0][2]) == set(cat_cols), 'col_transformer should be applied to all cat_cols'

    # test model pipeline
    assert isinstance(model.steps[0][1], ColumnTransformer), 'First step of the pipeline should be columns transformer'
    assert model.steps[1][0] == 'estimator', 'Second step of the pipeline should be named "estimator"'
    assert isinstance(model.steps[1][1], Ridge), 'Second step of the pipeline should be Ridge regression'

    R2_tr, R2_te = evaluate_model(model, X_tr, y_tr, X_te, y_te)
    print("Train R2 = %.2f" % R2_tr)
    print("Test R2 = %.2f" % R2_te)

    print("Number of features before one-hot-encoding: ")
    print(X_tr.shape[1])
    print()
    print("Number of features after one-hot-encoding: ")
    print(col_transformer.fit_transform(X_tr).shape[1])
    # TEST get_abs_corr_coef
    A = np.array(
        [[1, 0, 0],
         [0, 0.5, -0.5],
         [0, 1, -1]]
    )
    b = [0, 1, 1]
    print(get_abs_corr_coef(A, b))

    A = np.array(
        [[1, 0, 0],
         [0, 0.5, -0.5],
         [0, 1, -1]]
    )
    b = [0, 1, 1]
    corr = get_abs_corr_coef(A, b)
    assert len(corr) == 3, 'Length of the outut vector should be equal to the number of features'
    assert np.all(corr >= 0), 'Function should return absolute values'
    assert np.allclose(corr, [1., 0.86603, 0.86603]), 'Correlations are not computed correctly'

    feat_selector = SelectKBest(get_abs_corr_coef, k=20)

    model_k_best = Pipeline([
        ('col_transformer', col_transformer),
        ('feat_selector', feat_selector),
        ('estimator', Ridge())
    ])
    # TEST model with filter-based feature selection

    print('Pipeline:')
    print(model_k_best.steps[0])
    print(model_k_best.steps[1])
    print(model_k_best.steps[2])

    # test model pipeline
    assert isinstance(model_k_best.steps[0][1],
                      ColumnTransformer), 'First step of the pipeline should be columns transformer'
    assert isinstance(model_k_best.steps[1][1], SelectKBest), 'Second step of the pipeline should be SelectKBest'
    assert model_k_best.steps[2][0] == 'estimator', 'Third step of the pipeline should be named "estimator"'
    assert isinstance(model_k_best.steps[2][1], Ridge), 'Third step of the pipeline should be Ridge regression'
    R2_tr_kbest, R2_te_kbest = evaluate_model(model_k_best, X_tr, y_tr, X_te, y_te)
    print("Train R2 = %.2f" % R2_tr_kbest)
    print("Test R2 = %.2f" % R2_te_kbest)
    train_scores = [R2_tr, R2_tr_kbest]
    test_scores = [R2_te, R2_te_kbest]
    models = ['All features', 'KBest']
    alphas = np.logspace(-4, 4, 9)
    param_grid = {
        "estimator__alpha": alphas,
    }

    grid_cv = GridSearchCV(model, param_grid, cv=3, scoring='r2')
    grid_cv_k_best = GridSearchCV(model_k_best, param_grid, cv=3, scoring='r2')
    grid_cv.fit(X_tr, y_tr)
    grid_cv_k_best.fit(X_tr, y_tr)
    # TEST grid_cv model
    assert "estimator__alpha" in grid_cv.param_grid
    assert "estimator__alpha" in grid_cv_k_best.param_grid
    assert grid_cv.scoring == 'r2'
    assert grid_cv_k_best.scoring == 'r2'

    assert (grid_cv.param_grid["estimator__alpha"] == np.logspace(-4, 4, 9)).all()
    best = grid_cv.best_estimator_
    assert isinstance(best.steps[0][1], ColumnTransformer)
    assert isinstance(best.steps[1][1], Ridge)

    best_k = grid_cv_k_best.best_estimator_
    assert isinstance(best_k.steps[0][1], ColumnTransformer)
    assert isinstance(best_k.steps[1][1], SelectKBest)
    assert isinstance(best_k.steps[2][1], Ridge)

    print("Without feature selection:")
    print("Optimal alpha  = %.4f" % grid_cv.best_params_['estimator__alpha'])
    print("Optimal R2 score = %.4f" % grid_cv.best_score_)
    print()
    print("With feature selection:")
    print("Optimal alpha  = %.4f" % grid_cv_k_best.best_params_['estimator__alpha'])
    print("Optimal R2 score = %.4f" % grid_cv_k_best.best_score_)

    model = grid_cv_k_best.best_estimator_
    R2_tr, R2_te = evaluate_model(model, X_tr, y_tr, X_te, y_te)

    print("Train R2 = %.2f" % R2_tr)
    print("Test R2 = %.2f" % R2_te)

    model_k_best.set_params(estimator__random_state=0);
    # alphas = np.logspace(-4, 4, 9)
    # ks = np.arange(20, 310, 30)
    alphas = np.array([1.0])
    ks = np.array([80])
    param_grid = {
        "estimator__alpha": alphas,
        "feat_selector__k": ks
    }

    grid_cv_k_best = GridSearchCV(model_k_best, param_grid, cv=3, scoring='r2')
    grid_cv_k_best.fit(X_tr, y_tr)

    print('Best pipeline:')
    print(grid_cv_k_best.best_estimator_.steps[0])
    print(grid_cv_k_best.best_estimator_.steps[1])
    print(grid_cv_k_best.best_estimator_.steps[2])

    best_k = grid_cv_k_best.best_estimator_
    assert isinstance(best_k.steps[0][1], ColumnTransformer)
    assert isinstance(best_k.steps[1][1], SelectKBest)
    assert isinstance(best_k.steps[2][1], Ridge)


    print("Optimal alpha  = %.4f" % grid_cv_k_best.best_estimator_.steps[2][1].alpha)
    print("Optimal number of features  = %.4f" % grid_cv_k_best.best_estimator_.steps[1][1].k)
    print("Optimal R2 score = %.4f" % grid_cv_k_best.best_score_)

    # fit the best model and evaluate performance on test set
    best_model = grid_cv_k_best.best_estimator_
    R2_tr, R2_te = evaluate_model(best_model, X_tr, y_tr, X_te, y_te)
    print("Train R2 = %.2f" % R2_tr)
    print("Test R2 = %.2f" % R2_te)

    joblib.dump(best_model, '../model.pkl', compress=1)

if __name__=='__main__':
    train()
