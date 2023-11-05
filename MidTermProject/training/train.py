import joblib
import pandas as pd
from pandas.api.types import CategoricalDtype
import numpy as np
from MeanGroupImputer import MeanGroupImputer
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings("ignore")


titanic = pd.read_csv('data/titanic.csv')

def train():
    global titanic
    titanic["Title"] = titanic.Name.str.split(",", expand=True)[1].str.split(".", expand=True)[0]
    titanic["Title"] = titanic["Title"].str.strip()

    assert titanic['Title'].value_counts().shape[0] == 17, 'Wrong number of unique titles. 17 is expected'
    assert titanic['Title'].value_counts().Mr == 517, 'Wrong number of passengers with the Title `Mr`. 517 is expected'
    assert titanic[
               'Title'].value_counts().Miss == 182, 'Wrong number of passengers with the Title `Miss`. 182 is expected'
    assert titanic['Title'].value_counts().Dr == 7, 'Wrong number of passengers with the Title `Dr`. 7 is expected'
    names = ['Newell, Miss. Madeleine', 'Gale, Mr. Shadrach', 'Moubarek, Master. Halim Gonios ("William George")']
    titles = ['Miss', 'Mr', 'Master']
    for n, t in zip(names, titles):
        assert titanic.Title[titanic.Name == n].values == t, 'Wrong title for the passenger {}'.format(n)

    proper_titles = []
    proper_titles += titanic['Title'].value_counts().loc[titanic['Title'].value_counts() > 6].index.to_list()
    print(proper_titles)
    titanic['Title'] = titanic['Title'].astype('category')
    others = titanic['Title'].value_counts().loc[titanic['Title'].value_counts() <= 6].index
    label = 'Other'
    titanic['Title'] = titanic['Title'].cat.add_categories([label])
    titanic['Title'] = titanic['Title'].replace(others, label)

    assert titanic.Title.value_counts().shape[0] == 6
    assert sum(titanic.Title == 'Other') == 20
    CORRECT_proper_titles = ['Mr', 'Miss', 'Mrs', 'Master', 'Dr']
    for p in CORRECT_proper_titles:
        assert p in proper_titles, 'Title {} is expected to be in the list of proper titles'.format(p)

    titanic.Title = titanic.Title.astype('object')
    categ_columns = []
    for col in titanic.columns:
        if titanic[col].dtype.kind == 'O':
            categ_columns.append(col)

    assert 'Title' in categ_columns, 'Columns `Title` not in the list'
    assert 'Name' in categ_columns, 'Columns `Name` not in the list'
    assert 'Sex' in categ_columns, 'Columns `Sex` not in the list'
    assert 'Ticket' in categ_columns, 'Columns `Ticket` not in the list'
    assert 'Cabin' in categ_columns, 'Columns `Cabin` not in the list'
    assert 'Embarked' in categ_columns, 'Columns `Embarked` not in the list'

    # your code here
    cat_col_length = len(categ_columns)
    i = 0
    while i < cat_col_length:
        if titanic[categ_columns[i]].nunique() > 100:
            titanic.drop(categ_columns[i], axis=1, inplace=True)
            categ_columns.pop(i)
            cat_col_length -= 1
        else:
            i += 1

    CORRECT_categ = set(['Sex', 'Embarked', 'Title'])
    diff = list(CORRECT_categ - set(categ_columns))
    assert len(diff) == 0, '`categ_columns` is missing columns'
    diff = list(set(categ_columns) - CORRECT_categ)
    assert len(diff) == 0, '`categ_columns` has too amny columns'

    # your code here
    titanic.drop("PassengerId", axis=1, inplace=True)

    assert 'PassengerId' not in titanic.columns

    titanic['FamilySize'] = titanic['SibSp'] + titanic['Parch']
    titanic.drop("SibSp", axis=1, inplace=True)
    titanic.drop("Parch", axis=1, inplace=True)

    assert 'SibSp' not in titanic.columns
    assert 'Parch' not in titanic.columns
    assert sum(titanic.FamilySize == 0) == 537
    assert sum(titanic.FamilySize == 10) == 7

    # your code here
    total_survived = titanic.Survived.sum()

    titanic['FamilySize'].replace([0, 1, 2, 3, 4, 5, 6, 7, 10], [1, 2, 2, 2, 3, 3, 3, 3, 3], inplace=True)

    cat_dtype = CategoricalDtype(
        categories=[1, 2, 3], ordered=True)
    titanic['FamilySize'].astype(cat_dtype)
    titanic = pd.concat([titanic, pd.get_dummies(titanic['FamilySize'], prefix='FamilySize')], axis=1)

    assert 'FamilySize_1' in titanic.columns
    assert 'FamilySize_2' in titanic.columns
    assert 'FamilySize_3' in titanic.columns
    assert titanic.FamilySize_1.sum() + titanic.FamilySize_2.sum() + titanic.FamilySize_3.sum() == 891
    assert titanic.FamilySize_1.sum() > 0
    assert titanic.FamilySize_2.sum() > 0
    assert titanic.FamilySize_3.sum() > 0

    titanic.drop(['FamilySize'], axis=1, inplace=True)

    # your code here
    ordinal_cols = []
    numeric_cols = []
    columns = ['Pclass', 'Age', 'Fare']
    for col in columns:
        if titanic[col].dtype.kind == 'f':
            numeric_cols.append(col)
        else:
            ordinal_cols.append(col)

    assert len(ordinal_cols) == 1
    assert len(numeric_cols) == 2
    assert 'Pclass' in ordinal_cols
    assert 'Age' in numeric_cols
    assert 'Fare' in numeric_cols

    # your code here
    prop_missing = titanic.isnull().sum() / len(titanic)

    prop_CORRECT = titanic.isnull().sum() / titanic.shape[0]
    assert prop_missing.shape[0] == prop_CORRECT.shape[0], 'Wrong number of values'
    assert np.allclose(prop_missing.Age, 177. / 891.), 'Wrong proportion for th ecolumn Age'
    assert sum(prop_missing == 0) == sum(prop_CORRECT == 0), 'Wrong number of coumns with 0 missing values'

    # <a class="anchor" id="task11"></a>
    # What can we do with that? Below you can find some options:
    # - Fill all the NAs with the same value (mean, median, any other constant)
    # - Fill NAs using grouping (e.g. we can fill missing in the variable `Fare` for male and female passengers separately using their average value)
    # - Drop all the rows with missing values
    # - Drop the whole column (e.g. if there are too many missing values)
    #
    # The most popular way is to use `SimpleImputer` from sklearn. If fills all the missing values with the same number.
    #
    # we will implement a more phisticated Imputer. `MeanGroupImputer`. We will make sure that it has proper sklearn interface, so that we can use it within our pipelines. Below you can find the skeleton code for the `MeanGroupImputer`. Please read it carefully to make sure you understand everythig. Your task is to write missing code for the method `transform`.
    #

    # In[32]:


    correct_out = titanic['Age'].fillna(titanic.groupby('Title')['Age'].transform('mean'))
    imp = MeanGroupImputer(group_col='Title')
    titanic_copy = titanic.copy()
    given_out = imp.fit_transform(titanic[['Title', 'Age']])
    assert sum(given_out[:, 0] != correct_out) == 0

    # ## 1.4 Define column transformers <a class="anchor" id="columns"></a>
    #
    # In this task we will define columns transformer. Your task is to create three pipelines:
    #  - `age_pipe`: Pipeline to preprocess column `Age`. It uses `MeanGroupImputer` with the grouping variable `Title` to fill missing values in `Age` and then applies `StandardScaler`
    #  - `fare_pipe`: Pipeline to preprocess column `Fare`. It applies `StandardScaler` only
    #  - `categ_pipe`: Pipeline to preprocess all categorical variables. It uses `SimpleImputer` to impute missing values with the most frequent class and then applies `OneHotEncoder`
    #

    # your code here
    age_pipe = make_pipeline(MeanGroupImputer(group_col='Title'), StandardScaler())
    fare_pipe = make_pipeline(StandardScaler())
    categ_pipe = make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder())

    # Combine all three pipelines in one column transformer
    column_transformer = ColumnTransformer([
        ('age', age_pipe, ['Age', 'Title']),
        ('fare', fare_pipe, ['Fare']),
        ('all_categ', categ_pipe, categ_columns)],
        remainder='passthrough'
    )

    # In[35]:

    test_titanic = column_transformer.fit_transform(titanic)

    assert (pd.DataFrame(test_titanic).isna().sum().values == 0).all()
    assert test_titanic.shape[1] == 18, 'Wrong number of columns'

    # ## 1.5 Train/test split <a class="anchor" id="train_test"></a>

    # In[36]:

    tr, te = train_test_split(titanic, test_size=0.2, random_state=42)

    y_train = tr.Survived
    y_test = te.Survived
    X_train = tr.drop(['Survived'], axis=1)
    X_test = te.drop(['Survived'], axis=1)

    # # 2. Logistic Regression and Support Vector Machine <a class="anchor" id="part2"></a>
    #
    #
    # ## 2.1 Fit Logistic Regression
    #
    # Define the `log_reg_pipe` - pipeline which applies `column_transformer` and fits logistic regression with the the hyperparameter `penalty='none'` (by default sklearn applies L2 regularization). Calculate the 5-fold cross-validation score (use `accuracy` as a scoring function). Save the result (average accuracy on cross-validation) in the variable `log_reg_score`.

    # In[37]:

    # your code here
    log_reg_pipe = Pipeline([
        ('columnTransformer', column_transformer),
        ('logisticRegression', LogisticRegression(penalty='none'))
    ])
    log_reg_score = cross_val_score(log_reg_pipe, X_train, y_train, cv=5, scoring='accuracy').mean()

    assert np.allclose(log_reg_score, 0.824, rtol=1e-3)
    assert isinstance(log_reg_pipe.steps[0][1], ColumnTransformer)
    assert isinstance(log_reg_pipe.steps[1][1], LogisticRegression)

    # ## 2.2 Fit Support Vector Machine
    #
    # Define the `svm_pipe` - pipeline which applies `column_transformer` and fits Support Vector Machine model (it is imported for you below) using the hyperparameter `kernel='linear'`. Calculate the 5-fold cross-validation score (use `accuracy` as a scoring function). Save the result (average accuracy on cross-validation) in the variable `svm_score`.

    # In[40]:

    # your code here
    svm_pipe = Pipeline([
        ('columnTransformer', column_transformer),
        ('svm', SVC(kernel='linear'))
    ])
    svm_score = cross_val_score(svm_pipe, X_train, y_train, cv=5, scoring='accuracy').mean()

    assert np.allclose(svm_score, 0.83, rtol=1e-3)
    assert isinstance(svm_pipe.steps[0][1], ColumnTransformer)
    assert isinstance(svm_pipe.steps[1][1], SVC)

    # ## 2.3 Compare different models
    #
    # In this task you are supposed to use grid search to find the best classifier for the given dataset. Use `GridSearchCV` class from sklearn. Use 5-Fold cross validation with accuracy as a scoring metric.
    #
    # *Hints*. Read documentation to see, which hyperparameters `LogisticRegression` and `SVC` have. Pay attention to `kernel` in the SVM model and the regularization coefficient `C` for both LogisticRegression and SVC, try different penalties for `LogisticRegression`. Explore other hyperparameters as well. Your task is to simply get the best accuracy posibe. The minimum passing value will be 0.84 (average score on cross-validaition)
    #
    # Please, do not use models other that `SVC` or `LogisticRegression`.

    # In[42]:

    # your code here
    param_log_reg_grid = {
        "logisticRegression__C": np.logspace(-4, 4, 20),
        "logisticRegression__penalty": ['none', 'l1', 'l2']
    }

    param_svm_reg_grid = {
        "svm__C": [3.05],
        "svm__kernel": ['rbf']
    }

    # your code here
    # grid_pipe = GridSearchCV(log_reg_pipe,param_log_reg_grid,cv=5,scoring='accuracy')
    grid_pipe = GridSearchCV(svm_pipe, param_svm_reg_grid, cv=5, scoring='accuracy')
    grid_pipe.fit(X_train, y_train)
    # grid_pipe.fit(X_tr,y_tr)

    assert grid_pipe.best_score_ > 0.84
    assert isinstance(grid_pipe.best_estimator_.steps[1][1], SVC) or isinstance(grid_pipe.best_estimator_.steps[1][1],
                                                                                LogisticRegression)
    test_score = np.mean(cross_val_score(grid_pipe.best_estimator_, X_train, y_train, cv=5, scoring='accuracy'))
    assert np.allclose(test_score, grid_pipe.best_score_, rtol=1e-3)

    # ## 2.4 Eval best model on test
    #
    # Now, we can use the best estimator to evaluate model on the test dataset.
    #
    # 1. Fit model on the whole test data
    # 2. Make predictions on the test set
    # 3. Calculate accuracy

    # In[44]:

    grid_pipe.best_estimator_.fit(X_train, y_train)

    best_model = grid_pipe.best_estimator_
    joblib.dump(best_model, '../model.pkl', compress=1)
    print("The model file is generated and saved at root directory")

if __name__ == '__main__':
    train()


