# MLProject/Dockerfile

FROM python:3.9-slim-buster

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

COPY best_logistic_regression_model_artifact ./best_logistic_regression_model_artifact

CMD ["python", "predict_model.py"]
