FROM python:3.7

WORKDIR /home

RUN apt-get update -y && apt-get install -y g++ build-essential
RUN pip install --upgrade pip

RUN pip install numpy torch flask tensorboard ray

ADD src /home

ENV FLASK_APP=server

EXPOSE 5000

ENTRYPOINT python server.py