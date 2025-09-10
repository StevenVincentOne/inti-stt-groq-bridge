FROM python:3.12-slim
RUN apt-get update && apt-get install -y libopus0 libopus-dev && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY bridge.py ./
ENV PYTHONUNBUFFERED=1
EXPOSE 8080 8081
CMD ["python", "bridge.py"]
