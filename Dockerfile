FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements-pipeline.txt ./
RUN python -m pip install --upgrade pip \
    && pip install -r requirements-pipeline.txt

COPY src ./src
COPY reports ./reports
RUN mkdir -p data/raw data/interim data/processed reports/tables reports/figures

CMD ["python", "src/pipeline/prefect_etl.py"]
