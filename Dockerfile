FROM python:3.11-slim

# Install system tools needed
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working folder inside container
WORKDIR /app

# Copy requirements file first
COPY requirements.txt .

# Install all Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy all your project files into container
COPY . .

# Create folders the app needs
RUN mkdir -p ./model_cache ./lora_output ./lora_adapter ./data_storage

# Tell Docker which ports this app uses
EXPOSE 5000 7860

# Environment settings
ENV PYTHONUNBUFFERED=1
ENV HF_HUB_DOWNLOAD_TIMEOUT=300
ENV TRANSFORMERS_OFFLINE=0
ENV AUTO_LOAD=true

# Command to start the app
CMD ["python", "inference_api.py"]

