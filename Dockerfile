FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Copy semua kode
COPY . .

# Expose port FastAPI
EXPOSE 8000

# Jalankan FastAPI
CMD ["uvicorn", "src.inference.api:app", "--host", "0.0.0.0", "--port", "8000"]