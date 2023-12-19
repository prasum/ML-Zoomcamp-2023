FROM public.ecr.aws/lambda/python:3.10
COPY . ${LAMBDA_TASK_ROOT}
RUN pip install pipenv
RUN pipenv requirements > requirements.txt
RUN pipenv run pip install -t "${LAMBDA_TASK_ROOT}" -r requirements.txt
CMD ["app_lambda.handler"]