# Car Price Prediction - End-to-End ML Deployment

A production deployment of a machine learning model using FastAPI, Docker, and Streamlit. This project demonstrates the complete pipeline from model serving to user-facing application.

## Live Demo

- **Web Interface**: https://car-price-ml-deployment-gxj6x73mff2phxcf8fokyq.streamlit.app
- **API**: https://car-price-api-v1.onrender.com
- **API Documentation**: https://car-price-api-v1.onrender.com/docs

## Project Status

✅ **Production-ready ML deployment with complete CI/CD pipeline**
- FastAPI backend with `/health`, `/predict`, and `/metadata` endpoints
- Docker containerization with multi-stage builds
- Automated testing and deployment pipeline
- Live production deployment on Render
- Comprehensive test suite (8 tests, 100% passing)

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
├── .github/
│   └── workflows/
│       └── ci.yml               # CI/CD pipeline configuration
├── fast-api-car-price/          # Backend API
│   ├── src/
│   │   ├── __init__.py
│   │   └── main.py              # FastAPI application
│   ├── tests/
│   │   ├── __init__.py
│   │   └── test_api.py          # Automated test suite
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

### CI/CD
- **GitHub Actions**: Automated testing and deployment pipeline
- **pytest**: Testing framework with 8 integration tests
- **Mock objects**: Fast test execution (1.5s vs 57s with real model)

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

## CI/CD Pipeline

This project implements continuous integration and deployment using GitHub Actions.

### Workflow Overview

Every push to `main` triggers an automated pipeline:

1. **Test Stage**
   - Runs 8 integration tests using pytest
   - Tests health endpoint, prediction endpoint, input validation
   - Uses mocked ML model for fast execution (1.5s)
   - Fails fast if any test fails

2. **Build and Push Stage** (only if tests pass)
   - Builds Docker image using multi-stage Dockerfile
   - Tags with git commit SHA (`main-abc1234`) and `latest`
   - Pushes to Docker Hub registry
   - Uses layer caching for faster builds

### Running Tests Locally

```bash
cd fast-api-car-price

# Create virtual environment with uv
uv venv
source .venv/bin/activate

# Install dependencies
uv pip install pytest httpx -r requirements.txt

# Run tests
pytest tests/ -v
```

**Expected output:**
```
======================== 8 passed in 1.56s =========================
```

### Test Coverage

- `test_health_endpoint_returns_200` - Verifies API is reachable
- `test_health_endpoint_returns_correct_structure` - Validates health response format
- `test_predict_endpoint_accepts_valid_data` - Happy path prediction test
- `test_predict_endpoint_returns_price` - Validates prediction response structure
- `test_predict_endpoint_rejects_missing_fields` - Input validation test
- `test_predict_endpoint_rejects_invalid_types` - Type validation test
- `test_predict_handles_edge_cases` - Boundary condition testing
- `test_full_prediction_workflow` - End-to-end integration test

### Workflow Configuration

**File:** `.github/workflows/ci.yml`

**Triggers:**
- Push to `main` branch (runs tests + build)
- Pull requests to `main` (runs tests only)
- Manual dispatch via GitHub UI

**Path filters:** Only triggers on changes to `fast-api-car-price/` directory

**Secrets required:**
- `DOCKER_USERNAME` - Docker Hub username
- `DOCKER_PASSWORD` - Docker Hub access token

### Deployment Process

1. Make code changes locally
2. Test locally: `pytest tests/ -v`
3. Commit: `git commit -m "description"`
4. Push: `git push origin main`
5. GitHub Actions automatically:
   - ✅ Runs all tests
   - 🐳 Builds Docker image (if tests pass)
   - 📤 Pushes to Docker Hub
6. Render pulls updated image and redeploys

### Monitoring Builds

- **GitHub Actions**: [View workflows](https://github.com/brianjmalone/SDS-CP040-modelops/actions)
- **Docker Hub**: [View images](https://hub.docker.com/r/bjmalone724/car-price-api)
- **Build time**: ~3-4 minutes (first build), ~1-2 minutes (cached)

### Why Mock the Model in Tests?

**Problem:** Loading the real XGBoost model takes ~50 seconds, making tests painfully slow.

**Solution:** Use Python's `unittest.mock` to fake model predictions:
```python
mock_model = Mock()
mock_model.predict.return_value = [15000.0]
```

**Benefits:**
- Tests run in 1.5s instead of 57s (36x faster)
- Tests verify API logic, not model accuracy
- No need to commit large model.pkl file to git
- Follows industry best practices

**Trade-off:** Tests don't catch model loading errors. The Docker build process validates the real model loads correctly.

### Model Storage: Hugging Face Instead of Git

**Problem:** ML models (even small ones) shouldn't be committed to git repositories.

**Solution:** Store model on Hugging Face Hub, download during Docker build.

**Steps:**
1. Upload model to HF (one-time):
   - Create model repo at https://huggingface.co/new
   - Upload `model.pkl` via web UI
   - Repo: `brianmalone/car-price-model`

2. Dockerfile downloads model during build:
```dockerfile
RUN /usr/local/bin/uv pip install huggingface_hub --python /app/.venv/bin/python
RUN mkdir -p /app/models && \
    /app/.venv/bin/python -c "from huggingface_hub import hf_hub_download; \
    hf_hub_download(repo_id='brianmalone/car-price-model', \
    filename='model.pkl', local_dir='/app/models')"
```

**GOTCHA - Docker Cache Issue:**
If you previously had `COPY models/ ./models/` in your Dockerfile and switch to downloading from HF, GitHub Actions may fail with:
```
ERROR: failed to compute cache key: "/models": not found
```

**Fix:** Disable Docker layer caching temporarily:
```yaml
# In .github/workflows/ci.yml
- name: Build and push Docker image
  uses: docker/build-push-action@v5
  with:
    no-cache: true  # Forces fresh build, ignores old cached layers
```

After one successful build, you can re-enable caching by removing `no-cache: true`.

### GitHub Actions Workflow Trigger Gotcha

**Problem:** The `build-and-push` job only runs on `push` events, not manual `workflow_dispatch`.

**Why:** The workflow has this condition:
```yaml
if: github.ref == 'refs/heads/main' && github.event_name == 'push'
```

**What this means:**
- ✅ `git push origin main` → Tests run AND Docker builds
- ⊘ Manual "Run workflow" button → Tests run, Docker build SKIPPED

**When you'll see this:** If you manually trigger the workflow from GitHub UI, the build job shows a skip icon (⊘) instead of running.

**Solution:** Use `git push` to trigger workflows when you need the Docker build to run.

### Testing the CI/CD Pipeline Without Real Changes

**Problem:** You want to test the pipeline but don't have any meaningful code changes to commit.

**Solution:** Add a blank line to a file inside the watched path to trigger the workflow:

```bash
# Add a blank line to main.py (or any file in fast-api-car-price/)
echo "" >> /Users/bjmalone724/Desktop/SDS-CP040-modelops/advanced/submissions/team-members/brian-malone/fast-api-car-price/src/main.py

# Stage, commit, and push
git add advanced/submissions/team-members/brian-malone/fast-api-car-price/src/main.py
git commit -m "Trigger CI/CD workflow"
git push origin main
```

**Why this works:**
- The workflow has a path filter: `paths: ['advanced/submissions/team-members/brian-malone/fast-api-car-price/**']`
- Any change to files in this directory triggers the workflow
- Adding a blank line is a harmless change that git detects
- Changes to files outside this path (like the top-level README.md) won't trigger the workflow

**Alternative:** If you don't want to modify source files, create a dummy file:
```bash
touch /Users/bjmalone724/Desktop/SDS-CP040-modelops/advanced/submissions/team-members/brian-malone/fast-api-car-price/.trigger
git add advanced/submissions/team-members/brian-malone/fast-api-car-price/.trigger
git commit -m "Trigger CI/CD pipeline"
git push origin main
```

However, this creates clutter in your repository. The blank line approach is cleaner.

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

### Health Check Endpoint

**GET** `/health`

Response:
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### Model Metadata Endpoint

**GET** `/metadata`

Response:
```json
{
  "model_name": "XGBoost Car Price Predictor",
  "version": "1.0.0",
  "last_updated": "2024-10-12",
  "features": [
    "Manufacturer",
    "Model",
    "Fuel type",
    "Engine size",
    "Year of manufacture",
    "Mileage"
  ],
  "derived_features": [
    "age",
    "mileage_per_year",
    "vintage"
  ],
  "target": "price (GBP)"
}
```

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

## Key Lessons Learned

### 1. Don't Put ML Models in Git
**Problem**: Git is designed for source code, not large binary files. Even "small" models (50MB) cause repository bloat.

**Solution**: Store models on external platforms (Hugging Face Hub, S3, cloud storage) and download during Docker build or at runtime.

**Impact**: Cleaner repository, faster CI/CD, follows industry best practices.

### 2. Mock Expensive Operations in Tests
**Problem**: Loading ML models in tests makes them painfully slow (57 seconds for this project).

**Solution**: Use Python's `unittest.mock` to fake model predictions. Tests verify API logic, not model accuracy.

**Impact**: 36x faster test execution (1.5s), enabling rapid iteration and fast CI/CD pipelines.

### 3. Docker Layer Caching Can Bite You
**Problem**: Changed Dockerfile from `COPY models/` to Hugging Face download, but GitHub Actions kept failing with "models not found".

**Root cause**: Docker's layer cache referenced old build instructions.

**Solution**: Temporarily disable caching with `no-cache: true` when making structural Dockerfile changes.

**Lesson**: Understand your build system's caching behavior—it can save time but also cause confusing errors.

### 4. Multi-Stage Builds Aren't Just About Size
**Benefit #1**: Smaller images (exclude build tools from runtime)
**Benefit #2**: Better security (fewer packages = smaller attack surface)
**Benefit #3**: Cleaner separation (build dependencies vs runtime dependencies)

**Real impact**: Using `uv` in builder stage gave 10-100x faster installs without bloating the runtime image.

### 5. Path Filtering Prevents Wasted CI/CD Runs
**Problem**: Documentation changes (README updates) were triggering full Docker rebuilds.

**Solution**: Configure GitHub Actions to only trigger on changes to the `fast-api-car-price/` directory.

**Impact**: Faster feedback loops, lower CI/CD costs, more intentional about what triggers deployments.

## Learning Outcomes

This project demonstrates:
- Docker containerization with multi-stage builds and layer optimization
- RESTful API design with FastAPI and automatic OpenAPI documentation
- Cloud deployment strategies (container registries vs build-on-deploy)
- CI/CD pipeline implementation with GitHub Actions
- Automated testing with pytest and mocking strategies
- External artifact storage (Hugging Face Hub for models)
- Production considerations (health checks, cold starts, caching, logging)

## What's Completed

✅ **Core Infrastructure**
- FastAPI backend with complete endpoint suite (`/health`, `/predict`, `/metadata`)
- Multi-stage Docker builds with uv for fast package installation
- CI/CD pipeline with GitHub Actions (automated testing and deployment)
- Python logging with structured output
- Production deployment on Render with Docker Hub registry

✅ **Testing & Quality**
- Comprehensive test suite (8 integration tests)
- Mocked model for fast test execution (36x speedup)
- Automated testing on every push
- Path-filtered workflows to avoid unnecessary builds

✅ **Best Practices**
- External model storage (Hugging Face Hub, not git)
- Container registry workflow (build → test → push → deploy)
- Virtual environment isolation
- Automated deployment on successful tests

## Potential Future Enhancements

Ideas for extending this project:
- **Monitoring**: Add request logging, metrics dashboards, and alerting
- **Model Versioning**: Implement A/B testing with multiple model versions
- **Database Integration**: Store predictions for analysis and model retraining
- **Security**: Add authentication, rate limiting, and input sanitization
- **DevOps**: Add code coverage reporting, blue-green deployments
- **Performance**: Implement model caching and request batching

## Requirements

- Docker and Docker Hub account (for API deployment)
- Render account (for API hosting)
- GitHub account (for code hosting)
- Streamlit Community Cloud account (for frontend hosting)

## License

MIT