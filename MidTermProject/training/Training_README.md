# Training

Two models are used i.e. Logistic Regression Model which is a parametric model and SVC(Support Vector Classifier) which is a non parametric model. The dataset is split into train and test. GridSearchCV is used for hyperparameter tuning of both the models where the scoring metric chosen is accuracy which is maximized and the best model is saved using joblib and corresponding predictions are made on the test set. Sklearn pipeline method is used for performing transformations like scaling, imputing missing values, encoding categorical variables etc.

## Usage

- Go to the root folder
- install pipenv in base python environment
  ```bash
      pip install pipenv
   ```
- Setup the installed python libraries using the pipenv command
   ```bash
      pipenv install
   ```
- Activate virtual environment using the below command
  ```bash
      pipenv shell
   ```
- Go to the training folder
- For using jupyter notebook in virtual environment, create kernel by using the below command
  ```bash
     python -m ipykernel install --user --name=my-virtualenv-name
     pipenv shell
     jupyter notebook
  ```
- For playing with the training process, refer the jupyter notebook
- Alternatively run the python script for training the model and saving it in root directory

   ```bash
      python train.py
   ```

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

[MIT](https://choosealicense.com/licenses/mit/)