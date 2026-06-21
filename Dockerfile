# Use an explicit slim Python runtime baseline image
FROM python:3.12-slim

# Install system utilities needed for compiling C++ tools (XGBoost dependencies)
RUN apt-get update && apt-get install -y \
    build-essential \
    libomp-dev \
    && rm -rf /var/lib/apt/lists/*

# Set up runtime isolation directory
WORKDIR /app

# Copy dependency mappings and install packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application layers, configs, database logs, and script metrics
COPY src/ ./src/
COPY app.py .
COPY config.yaml .
COPY mlflow.db .

# Expose the network transport channel for Streamlit web pipelines
EXPOSE 8501

# Enforce stable environment configurations
ENV PYTHONUNBUFFERED=1

# Command execution to launch app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
