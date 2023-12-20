# DATASET

Since the first automobile, the Benz Patent Motor Car in 1886, Mercedes-Benz has stood for important automotive innovations. These include, for example, the passenger safety cell with crumple zone, the airbag and intelligent assistance systems. Mercedes-Benz applies for nearly 2000 patents per year, making the brand the European leader among premium car makers. Daimler’s Mercedes-Benz cars are leaders in the premium car industry. With a huge selection of features and options, customers can choose the customized Mercedes-Benz of their dreams. .
To ensure the safety and reliability of each and every unique car configuration before they hit the road, Daimler’s engineers have developed a robust testing system. But, optimizing the speed of their testing system for so many possible feature combinations is complex and time-consuming without a powerful algorithmic approach. As one of the world’s biggest manufacturers of premium cars, safety and efficiency are paramount on Daimler’s production lines.
Here we predict the car manufacturing time and is a regression problem since target variable is continuous.

# EDA

The dataset is split into 80:20 train test ratio and the test set is kept separately for evaluation. The numerical and categorical variables are identified. The missing values are checked in the dataset.
The categorical variables are encoded using OneHotEncoding to convert into numeric form. Since the number of predictors are many, feature selection technique i.e. SelectKBest is used to prevent the model from overfitting.
Refer the ```Training.ipynb``` inside the train folder.

# Setup Virtual Environment

- Run `pipenv shell` and `pipenv sync`
- The virtual environment can also be run through jupyter using the following
  ```python -m ipykernel install --user --name=my-virtualenv-name```

# Model Hyperparameter Tuning and Training

- Two model pipelines i.e. baseline ridge regression model without feature selection and ridge regression with SelectKBest feature selection are used.
- Hyperparameters like regularisation parameter and number of features are tuned using GridSearchCV.
- The best model is saved and deployed to local, docker and aws
- To train, go to train folder and run the following
  ```python train.py```
- Alternatively ```Training.ipynb``` can be run for cell by cell execution

# Deployment

The local Flask web service can be deployed to Docker container

## Setting Local Docker Environment
- Create Dockerfile as below which includes the model and custom python file used for transformation during model training in the root folder
  ```bash
      FROM python:3.10.9-slim
      RUN pip install pipenv
      WORKDIR /app
      COPY ["Pipfile", "Pipfile.lock", "./"]
      RUN pipenv install --deploy --system
      COPY ["*.py", "model.pkl", "./"]
      EXPOSE 9696
      ENTRYPOINT ["gunicorn", "--bind", "0.0.0.0:9696", "app_docker:app"]
   ```
- Build the docker image using the below command
  ```bash
      docker build -t mercedes-manufacture-time-predictor -f local_Dockerfile . 
   ```
- Run the docker image using the below command
   ```bash
      docker run -it -p 9696:9696 mercedes-manufacture-time-predictor:latest
   ```

## Setting AWS Lambda + API Gateway Environment
- Install AWS Serveless package using the below command
  ```npm install -g serverless```
- Install serverless-wsgi in virtual environment using below command
  ```pipenv install serverless-wsgi>=2.0.2```
- Create Dockerfile for deployment to AWS lambda function
  ```bash
      FROM public.ecr.aws/lambda/python:3.10
      COPY . ${LAMBDA_TASK_ROOT}
      RUN pip install pipenv
      RUN pipenv requirements > requirements.txt
      RUN pipenv run pip install -t "${LAMBDA_TASK_ROOT}" -r requirements.txt
      CMD ["app_lambda.handler"]
   ```
- Create .dockerignore file to exclude metadata refs for deployment to docker
  ```bash
      __pycache__/
      .git/
      .serverless/
      .gitignore
      .dockerignore
      serverless.yml 
   ```
- Create aws lambda handler function for flask web service through serverless-wsgi
   ```python
      import joblib
      from flask import Flask
      from flask import request
      from flask import jsonify
      import pandas as pd
      import serverless_wsgi
      
      model = joblib.load('model.pkl')
      
      app = Flask('Mercedes Car Manufacture Time')
      
      @app.route('/predict', methods=['POST'])
      def predict():
          client = request.get_json()
          X = pd.DataFrame.from_dict([client])
          y_pred = model.predict(X)[0]
          result = {
              'Predicted Time': y_pred
          }
          return jsonify(result)
      
      def handler(event, context):
          return serverless_wsgi.handle_request(app, event, context)
   ```
- Create .serverless.yml file with AWS lambda function and REST API Endpoint details
   ```bash
       service: mercedes-manufacturing-prediction #name this whatever you want
       provider:
         name: aws
         region: ap-south-1
         ecr:
           images:
             appimage:
             path: ./
        
       functions:
         app:
           image:
             name: appimage
           timeout: 30
           events:
             - httpApi: 'POST /predict'
   ```
- Run ```sls deploy``` to create AWS Lambda Function which gets triggered through AWS API Gateway Service
- Use the below AWS API Gateway Post Endpoint Url for getting prediction
  ```https://sls2oq7vo6.execute-api.ap-south-1.amazonaws.com/predict```

# Evaluation

The evaluation test set is one record in csv format which will be universally used in the local, flask based, docker and aws serverless lambda and api gateway predict methods.
The evaluation metric chosen here is R-squared since it is a regression problem i.e. the time in seconds required for Mercedes car manufacturing.

## Usage

- Go to the root folder
- For local predict, run the following
  ```bash
      python test_predict.py
   ```
- For flask based predict, we have to start the flask server and run predict in separate command prompt instance
  - First Instance
   ```bash
      waitress-serve --port=80 --call "app:create_app"
   ```
  - Second Instance
   ```bash
      python test_flask_client.py
   ```
- For flask based predict, we have to build and run docker container discussed previously and run predict in separate command prompt instance
  ```bash
     python test_docker_flask_client.py
   ```
- For AWS Serverless Lambda Function with API Gateway Endpoint url based predict, the app is deployed to aws using Lambda and API Gateway Service discussed previously and run predict in separate command prompt instance
  ```bash
     python serverless_lambda_api_predict.py
   ```

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

[MIT](https://choosealicense.com/licenses/mit/)
