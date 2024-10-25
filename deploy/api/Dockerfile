FROM python:3.11
WORKDIR /deploy/anticrawler-api

COPY requirements.txt /deploy/anticrawler-api/requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY deploy/api /deploy/anticrawler-api

COPY datasets/dataset.py /deploy/anticrawler-api/datasets/dataset.py
COPY datasets/__init__.py /deploy/anticrawler-api/datasets/__init__.py

COPY models /deploy/anticrawler-api/models

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]


