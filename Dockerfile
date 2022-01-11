FROM python:3.9-slim as builder
WORKDIR /app
COPY requirements.txt .
COPY app.py .
COPY assets .
RUN pip install -r requirements.txt
RUN python app.py