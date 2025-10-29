# Builder stage
FROM python:3.11-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

COPY requirements.txt .

# Install dependencies into virtual environment
RUN /usr/local/bin/uv venv /app/.venv && \
    /usr/local/bin/uv pip install --no-cache-dir -r requirements.txt --python /app/.venv/bin/python

# Download model from Hugging Face (no token needed for public repo)
RUN /usr/local/bin/uv pip install huggingface_hub --python /app/.venv/bin/python
RUN mkdir -p /app/models && \
    /app/.venv/bin/python -c "from huggingface_hub import hf_hub_download; \
    hf_hub_download(repo_id='brianmalone/car-price-model', \
    filename='model.pkl', local_dir='/app/models')"

COPY src/ ./src/

# Runtime stage
FROM python:3.11-slim AS runtime

ENV PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# Copy virtual environment and source code from builder
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src /app/src
COPY --from=builder /app/models /app/models

EXPOSE 8000

CMD ["uvicorn", "src.main:api", "--host", "0.0.0.0", "--port", "8000"]
