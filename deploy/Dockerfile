FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Copy application code
COPY elc/ /app/elc/
COPY deploy/requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r /app/requirements.txt
RUN pip install --no-cache-dir jupyterlab

# Expose Jupyter port
EXPOSE 8888

# Create entrypoint script
RUN cat <<'SCRIPT' > /app/entrypoint.sh
#!/bin/bash
set -e

# Start Ollama in the background
ollama serve &

# Wait for Ollama to be ready
echo "Waiting for Ollama to start..."
until curl -s http://localhost:11434/api/tags > /dev/null 2>&1; do
    sleep 1
done
echo "Ollama is ready."

# Pull required models
echo "Pulling phi3:mini..."
ollama pull phi3:mini

echo "Pulling qwen3.5:4b..."
ollama pull qwen3.5:4b

echo "Models pulled successfully."

# Start JupyterLab
exec jupyter lab \
    --no-browser \
    --ip=0.0.0.0 \
    --port=8888 \
    --allow-root \
    --NotebookApp.token='' \
    --NotebookApp.password=''
SCRIPT
RUN chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]
