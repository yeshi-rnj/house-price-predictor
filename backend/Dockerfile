# Build from the backend folder:  docker build -t house-price-api backend
# Requires models/ to contain the trained files (run train.py first).
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /srv

COPY requirements.txt requirements-deploy.txt ./
COPY models/ ./models/
# model-requirements.txt pins the exact library versions the model was trained with.
RUN pip install --no-cache-dir -r requirements-deploy.txt -r models/model-requirements.txt

COPY app/ ./app/
COPY wsgi.py ./

RUN useradd --create-home appuser
USER appuser

EXPOSE 5000
# One worker keeps memory low on small hosts; raise --workers if you have the RAM.
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-5000} --workers 1 --threads 4 --timeout 60 wsgi:app"]
