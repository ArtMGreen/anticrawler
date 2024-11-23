FROM python:3.11
WORKDIR /deploy/anticrawler-api

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y

COPY requirements.txt /deploy/anticrawler-api/requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY deploy/api/main.py /deploy/anticrawler-api/main.py

COPY datasets/dataset.py /deploy/anticrawler-api/datasets/dataset.py

COPY models /deploy/anticrawler-api/models
COPY attacks /deploy/anticrawler-api/attacks
COPY defences /deploy/anticrawler-api/defences

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]


