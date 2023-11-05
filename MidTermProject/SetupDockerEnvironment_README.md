# Deploying to Docker

The local Flask web service can be deployed to Docker container

## Setting Docker Environment
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
      docker build -t titanic-survivor-predictor . 
   ```
- Run the docker image using the below command
   ```bash
      docker run -it -p 9696:9696 titanic-survivor-predictor:latest
   ```

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

[MIT](https://choosealicense.com/licenses/mit/)