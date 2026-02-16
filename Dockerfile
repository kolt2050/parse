FROM python:3.13-slim

# Install system dependencies if any are needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create directory for static files
RUN mkdir -p static

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Port for Flask
EXPOSE 8080

# Default command (will be overridden)
CMD ["python", "app.py"]
