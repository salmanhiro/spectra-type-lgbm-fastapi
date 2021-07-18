FROM tiangolo/uvicorn-gunicorn-fastapi:python3.8

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY ./app /app
COPY ./model /model

WORKDIR /app

CMD ["python", "main.py"]