FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir \
    transformers \
    gradio

COPY app.py .
COPY toxic_multilabel_model ./toxic_multilabel_model

EXPOSE 7860

CMD ["python", "app.py"]
