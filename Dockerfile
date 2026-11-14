# Inference service image (FastAPI + scikit-learn model exported from MLflow).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app
COPY serving/requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

COPY serving/app.py ./serving/app.py
COPY serving/__init__.py ./serving/__init__.py
COPY serving/model/model.joblib serving/model/metadata.json ./model/

# Run as an unprivileged user.
RUN useradd --uid 10001 --no-create-home appuser
USER 10001

EXPOSE 8000
CMD ["uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
