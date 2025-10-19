# Builder stage
FROM python:3.11-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

COPY requirements.txt .

# Install dependencies into virtual environment
RUN /usr/local/bin/uv venv /app/.venv && \
    /usr/local/bin/uv pip install --no-cache-dir -r requirements.txt --python /app/.venv/bin/python

COPY src/ ./src/
COPY models/ ./models/

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
