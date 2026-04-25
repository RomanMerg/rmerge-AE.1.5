FROM python:3.11-slim

WORKDIR /app

# Install system deps for psycopg2 and pymupdf
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Create uploads directory
RUN mkdir -p /app/uploads

EXPOSE 7860

CMD ["python", "app.py"]
