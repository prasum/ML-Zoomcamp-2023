# Evaluation

The evaluation test set is one record in csv format which will be universally used in the local, flask based, docker and aws elb predict methods.
The evaluation metric chosen is accuracy since it is a classification problem i.e. the passenger has survived or not in titanic shipwreck.

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
- For flask based predict, we have to build and run docker container which is part of separate readme and run predict in separate command prompt instance
  ```bash
     python test_docker_flask_client.py
   ```
- For AWS Elastic Load Balancer based predict, the app is deployed to aws elb which is part of separate readme and run predict in separate command prompt instance
  ```bash
     python eb_docker_flask_client.py
   ```

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

[MIT](https://choosealicense.com/licenses/mit/)