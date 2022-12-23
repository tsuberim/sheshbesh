FROM python:alpine

RUN pip install flask numpy ray torch

ADD src /home/src

WORKDIR /home/src

ENTRYPOINT [ "python", "server.py" ]