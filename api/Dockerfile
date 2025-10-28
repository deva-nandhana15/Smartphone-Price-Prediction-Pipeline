FROM python:3.10-slim

WORKDIR /app

# Copy API code
COPY . /app

# Install system deps (if any)
RUN apt-get update && apt-get install -y build-essential gcc --no-install-recommends && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
