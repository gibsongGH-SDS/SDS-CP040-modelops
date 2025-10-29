# Use official Python image (pick Python 3.12 or similar version)
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all code to the container
COPY . .

# Expose the port FastAPI runs on
EXPOSE 8000

# Run the app (change main:app if the FastAPI app variable is not "app")
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
