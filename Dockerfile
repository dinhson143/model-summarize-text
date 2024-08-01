FROM python:3.12

WORKDIR /tmp
ENV PYTHONPATH=.:/usr/local/lib/python3.12
RUN pip3 install --upgrade pip setuptools
RUN pip3 install --user rouge-score pyvi joblib pandas numpy nltk gensim boto3

COPY ./src ./src
RUN chmod 755 ./src/run_batch.py

ENTRYPOINT ["python3", "./src/run_batch.py"]
