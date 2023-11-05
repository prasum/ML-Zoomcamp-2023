from sklearn.base import BaseEstimator, TransformerMixin

class MeanGroupImputer(BaseEstimator, TransformerMixin):
    '''
    Class used for imputing missing values in a pd.DataFrame using mean value within group

    Parameters
    ----------
    group_cols : str
        A column used to calculate the mean values
    Returns
    -------
    X : array-like
        The array with imputed values in the target column
    '''

    def __init__(self, group_col):
        assert type(group_col) == str, 'group_col should be a string'

        self.group_col = group_col

    def fit(self, X, y=None):
        assert X[self.group_col].isna().sum() == 0, 'There are missing values in the group_col'

        # Group dataset by `group_col` and calculate mean value of all the other columns within this group
        self.mapping = X.groupby(self.group_col).mean()
        return self

    def transform(self, X, y=None):
        X = X.copy()
        # make sure that the imputer was fitted
        assert self.mapping is not None

        # loop over all the groups
        for index, row in self.mapping.iterrows():
            # Fill in missing values for the group `index` with the values `row`
            # your code here
            # print(X.loc[self.group_col,row.index])
            # print(X[self.group_col])
            # print(X[self.group_col].index)
            for i in row.index:
                mask = X[self.group_col] == index
                X.loc[mask, i] = X.loc[mask, i].fillna(self.mapping.loc[index, i])

        # Then drop grouping column (we did not transform it, so it is not needed anymore)
        X.drop(self.group_col, axis=1, inplace=True)
        return X.values