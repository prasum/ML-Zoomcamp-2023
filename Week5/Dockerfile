FROM svizor/zoomcamp-model:3.10.12-slim
RUN pip install pipenv
WORKDIR /app
COPY ["Pipfile","Pipfile.lock","./"]
RUN pipenv install --system --deploy
COPY ["app_docker.py","./"]
EXPOSE 9696
ENTRYPOINT ["gunicorn" ,"--bind=0.0.0.0:9696", "app_docker:app"]