FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Start the server
CMD ["sh", "-c", "uvicorn src.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
