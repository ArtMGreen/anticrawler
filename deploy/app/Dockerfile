FROM python:3.11

WORKDIR /deploy/anticrawler-app

COPY deploy/app/app.py /deploy/anticrawler-app/app.py

RUN pip install streamlit
RUN pip install requests

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501"]
