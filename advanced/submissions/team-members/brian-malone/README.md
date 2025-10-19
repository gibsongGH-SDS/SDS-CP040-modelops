# Car Price Prediction - End-to-End ML Deployment

A production deployment of a machine learning model using FastAPI, Docker, and Streamlit. This project demonstrates the complete pipeline from model serving to user-facing application.

## Live Demo

- **Web Interface**: https://car-price-ml-deployment-gxj6x73mff2phxcf8fokyq.streamlit.app

## Architecture

```
User → Streamlit Frontend → FastAPI Backend → XGBoost Model → Prediction
        (Streamlit Cloud)      (Render)
```

The system consists of two independently deployed services:
- **Backend API**: Containerized FastAPI application serving model predictions
- **Frontend**: Streamlit web application providing user interface

## Project Structure

```
car-price-ml-deployment/
├── fast-api-car-price/          # Backend API
│   ├── src/
│   │   ├── __init__.py
│   │   └── main.py              # FastAPI application
│   ├── models/
│   │   └── model.pkl            # Trained XGBoost model (gitignored)
│   ├── Dockerfile               # Multi-stage container build
│   ├── requirements.txt         # Unpinned dependencies
│   └── requirements-frozen.txt  # Version snapshot
└── streamlit-car-price/         # Frontend application
    ├── app.py                   # Streamlit interface
    └── requirements.txt
```

## Technology Stack

### Backend
- **FastAPI**: Async Python web framework with automatic OpenAPI documentation
- **XGBoost**: Pre-trained gradient boosting model
- **Docker**: Container runtime with multi-stage builds
- **uv**: Ultra-fast Python package installer (10-100x faster than pip)
- **Render**: Cloud platform for containerized deployments

### Frontend
- **Streamlit**: Python framework for data applications
- **Streamlit Community Cloud**: Free hosting for Streamlit apps

## Key Technical Decisions

### Unpinned Dependencies

The `requirements.txt` uses unpinned versions:
```
fastapi
uvicorn[standard]
xgboost
```

**Rationale**: The model was trained with unknown library versions. Pinned dependencies caused binary incompatibility errors (numpy version mismatches, sklearn warnings). Unpinned dependencies allowed `uv` to resolve compatible versions automatically.

**Trade-off**: Reproducibility vs compatibility. This worked for development but should be frozen for production using:
```bash
docker run --rm bjmalone724/car-price-api:v1 pip list --format=freeze > requirements-frozen.txt
```

### Multi-Stage Docker Build with uv

The Dockerfile uses a **multi-stage build** to optimize image size and security:

```dockerfile
# Builder stage - install dependencies
FROM python:3.11-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
WORKDIR /app
COPY requirements.txt .
RUN /usr/local/bin/uv venv /app/.venv && \
    /app/.venv/bin/uv pip install --no-cache-dir -r requirements.txt
COPY src/ ./src/
COPY models/ ./models/

# Runtime stage - minimal production image
FROM python:3.11-slim AS runtime
ENV PATH="/app/.venv/bin:$PATH"
WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src /app/src
COPY --from=builder /app/models /app/models
EXPOSE 8000
CMD ["uvicorn", "src.main:api", "--host", "0.0.0.0", "--port", "8000"]
```

**Benefits**:
- **Smaller image size**: Runtime stage excludes build tools and uv binary
- **Better security**: Fewer installed packages in production image
- **Virtual environment isolation**: Dependencies installed in venv instead of system-wide
- **Fast builds**: uv is 10-100x faster than pip (49 seconds first build, 0.6 seconds for code-only changes vs 3-4 minutes with traditional pip)
- **Layer caching**: Optimizes for code iteration by separating dependency installation from code copying

### Deployment Architecture: Pre-built Images vs Build-on-Deploy

This project uses a **container registry approach** rather than building directly on the deployment platform.

**Architecture chosen:**
1. Build Docker image locally or in CI/CD
2. Push to Docker Hub (`docker.io/bjmalone724/car-price-api:v1`)
3. Render pulls pre-built image for deployment

**Alternative approach (not used):**
- Push source code to GitHub
- Render clones repository and builds Docker image on their infrastructure
- No container registry required

**Why I chose this approach:**

**Advantages:**
- **Faster deployments**: Image is pre-built, Render only pulls and runs (seconds vs minutes)
- **Build once, deploy anywhere**: Same image can be deployed to multiple platforms (AWS, GCP, Azure)
- **Consistent environments**: Eliminates "works on my machine" problems
- **Testing in production**: Can run the exact production image locally before deploying
- **Enables CI/CD**: Automated pipelines can build, test, and push images (Week 3 requirement)
- **Version control**: Tag images (`v1`, `v2`, `latest`) for easy rollback
- **Resource efficiency**: Build happens on your machine or CI/CD runners, not on free-tier deployment platform

**Trade-offs:**
- Requires Docker Hub account (or alternative registry like GitHub Container Registry)
- More complex initial setup compared to "git push to deploy"
- Must rebuild and push for every code change
- Need to manage image versioning strategy

**Industry context:** This approach mirrors production practices at companies running Kubernetes or containerized workloads. Building on the deployment platform is simpler for prototypes but doesn't scale to multi-environment deployments (dev/staging/production).


### Monorepo Structure

Used a monorepo to keep API and frontend code together rather than separate repositories.

**Advantages**:
- Single source of truth
- Easier to track integration changes
- Better for portfolios (shows full stack)

**Disadvantages**:
- Each service still deploys independently
- Slightly more complex CI/CD setup

### Health Check Endpoint

```python
@api.get("/health")
async def health():
    return {"status": "healthy"}
```

**Purpose**: Cloud platforms use health checks to verify service availability. Render defaults to `/healthz` (Kubernetes convention), but `/health` was chosen for simplicity.

## Deployment Process

### Backend (FastAPI on Render)

1. Build Docker image:
```bash
docker build -t bjmalone724/car-price-api:v1 .
```

2. Push to Docker Hub:
```bash
docker push bjmalone724/car-price-api:v1
```

3. Deploy on Render:
   - Service type: Web Service
   - Image: `docker.io/bjmalone724/car-price-api:v1`
   - Port: 8000
   - Health check path: `/health`

**Free tier limitation**: Service spins down after 15 minutes of inactivity. First request after sleep has 30-50 second cold start.

### Frontend (Streamlit on Streamlit Cloud)

1. Push code to GitHub
2. Connect Streamlit Cloud to repository
3. Configure deployment:
   - Main file: `streamlit-car-price/app.py`
   - Branch: `main`

Streamlit Cloud automatically redeploys on git push.

## Local Development

### Run API Locally

```bash
cd fast-api-car-price
docker build -t car-price-api .
docker run -p 8000:8000 car-price-api

# Test endpoint
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Manufacturer": "Toyota",
    "Model": "Corolla",
    "Fuel type": "Petrol",
    "Engine size": 1.8,
    "Year of manufacture": 2018,
    "Mileage": 45000
  }'
```

### Run Frontend Locally

```bash
cd streamlit-car-price
pip install -r requirements.txt
streamlit run app.py
```

Application opens at `http://localhost:8501`

## Common Issues and Solutions

### Model Binary Incompatibility

**Problem**: `ModuleNotFoundError: No module named 'numpy._core'`

**Cause**: Model was pickled with different library versions than deployment environment

**Solution**: Use unpinned dependencies and let package manager resolve compatibility

### API Response Field Mismatch

**Problem**: Frontend shows "No prediction returned"

**Cause**: API returns `predicted_price_gbp` but frontend expects `predicted_price`

**Solution**: Always test API responses manually before writing integration code:
```bash
curl -X POST <api-url>/predict -H "Content-Type: application/json" -d '{...}'
```

### Docker Layer Caching

**Problem**: Long rebuild times when only code changes

**Solution**: Order Dockerfile commands from least to most frequently changed:
1. Install dependencies (changes rarely)
2. Copy model file (changes rarely)
3. Copy source code (changes frequently)

## API Documentation

Interactive API documentation available at:
- Swagger UI: `https://car-price-api-v1.onrender.com/docs`
- ReDoc: `https://car-price-api-v1.onrender.com/redoc`

### Prediction Endpoint

**POST** `/predict`

Request body:
```json
{
  "Manufacturer": "string",
  "Model": "string",
  "Fuel type": "string",
  "Engine size": "number",
  "Year of manufacture": "integer",
  "Mileage": "integer"
}
```

Response:
```json
{
  "predicted_price_gbp": "number"
}
```

## Learning Outcomes

This project demonstrates:
- Docker containerization with optimization techniques
- RESTful API design with FastAPI
- Cloud deployment on multiple platforms
- Integration debugging between services
- Dependency management trade-offs
- Monorepo project structure
- Production considerations (health checks, cold starts, caching)

## Next Steps

Potential enhancements for production readiness:
- Add request logging and monitoring
- Implement model versioning (A/B testing)
- Add input validation with meaningful error messages
- Set up CI/CD pipeline with GitHub Actions
- Store predictions in database for analysis
- Add authentication and rate limiting

## Requirements

- Docker and Docker Hub account (for API deployment)
- Render account (for API hosting)
- GitHub account (for code hosting)
- Streamlit Community Cloud account (for frontend hosting)

## License

MIT