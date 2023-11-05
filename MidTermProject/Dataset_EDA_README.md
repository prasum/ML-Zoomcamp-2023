# DATASET

The sinking of the Titanic is one of the most infamous shipwrecks in history.
On April 15, 1912, during her maiden voyage, the widely considered “unsinkable” RMS Titanic sank after colliding with an iceberg. Unfortunately, there weren’t enough lifeboats for everyone on board, resulting in the death of 1502 out of 2224 passengers and crew.

While there was some element of luck involved in surviving, it seems some groups of people were more likely to survive than others.

We will build a predictive model that answers the question: “what sorts of people were more likely to survive?” using passenger data (i.e. name, age, gender, socio-economic class, etc.).

## EDA

We will segregate the numerical and categorical variables and analyze their distribution with respect to the target variable i.e. Survived which is categorical variable. Also we will impute the missing values in features and drop features if the missing values are more than threshold. For imputing the missing values, we would be using a custom scikit learn transformer class i.e. MeanGroupImputer which will override the base fit and transform functions of scikit learn. Also we will create new Features out of the existing features based on the analysis of the target variable. We will also drop insignificant features in the dataset. Also numerical features are scaled using StandardScaler and categorical variables are encoded numerically using OneHotEncoding. For playing with the EDA process, refer to the jupyter notebook in the training folder.


## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

[MIT](https://choosealicense.com/licenses/mit/)