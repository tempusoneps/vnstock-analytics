FROM python:3.12

RUN apt-get update && apt-get install -y nano

WORKDIR /notebooks

COPY requirements.txt .

RUN pip install -r requirements.txt
