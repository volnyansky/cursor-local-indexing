FROM python:3.12-slim
ARG TARGETARCH
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install --no-cache-dir poetry

# Configure Poetry: Don't create a virtual environment (we're in a container)
RUN poetry config virtualenvs.create false

# Copy Poetry configuration files first to leverage Docker cache
COPY pyproject.toml poetry.lock* ./

# Install Python dependencies with cache mount for Poetry cache
RUN --mount=type=cache,target=/root/.cache/pypoetry,id=potery-${TARGETARCH} \
    poetry install --no-interaction --no-ansi --no-root

# Copy the application code
COPY code_indexer_server.py .

# Create directory for ChromaDB
RUN mkdir -p /app/chroma_db

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["python", "code_indexer_server.py"] 