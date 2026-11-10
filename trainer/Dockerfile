# Imagen del servicio de entrenamiento. No trae datos ni modelos: los lee y los
# escribe en DATA_URI y MODELS_URI (carpeta montada o gs://, con Workload Identity).
FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ventas/ ventas/
COPY train.py .

# Sin root: el Job no necesita privilegios
RUN useradd --create-home --uid 10001 trainer
USER trainer

ARG GIT_SHA=local
ENV GIT_SHA=${GIT_SHA} DATA_URI=/data MODELS_URI=/modelos

ENTRYPOINT ["python", "train.py"]
