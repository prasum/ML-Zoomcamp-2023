# Deploying to AWS ELB

The local Docker Based environment can be deployed to AWS ELB

## Setting AWS ELB Environment
- The awselb cli is alreadly installed as it is a requirement in pipenv
  
- Activate virtual environment using the below command
  ```bash
      pipenv shell 
   ```
- Initialize aws eb environment using the below command
   ```bash
      eb init -p "Docker running on 64bit Amazon Linux 2" -r ap-south-1 titanic-survivor-prediction-env
   ```
- Test if the eb environment runs locally
  ```bash
      eb local run --port 9696
   ```
   
- Deploy the eb web service on cloud using the below command
  ```bash
      eb create titanic-survivor-prediction-env
   ```
- After the environment gets completed, we get the following url
  ```bash
      titanic-survivor-prediction-env.eba-5d3pzxdd.ap-south-1.elasticbeanstalk.com
  ```
  
- The  above url can be used as host address in flask predict script for getting the predictions

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

[MIT](https://choosealicense.com/licenses/mit/)