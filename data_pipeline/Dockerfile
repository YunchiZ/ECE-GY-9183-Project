FROM python:3.10-slim

WORKDIR /app

COPY .

RUN pip install flask pandas scikit-learn

CMD ["python3", "etl_app.py"]
