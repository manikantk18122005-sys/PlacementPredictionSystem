# ==============================================================================
# Placement Prediction System - Production Dockerfile
# ==============================================================================

# 1. Base Image: Lightweight Python 3.13 slim distribution
FROM python:3.13-slim

# 2. Performance & Logging environment flags
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# 3. Define workspace directory inside container
WORKDIR /app

# 4. Copy dependency requirements first to leverage Docker layer caching
COPY requirements.txt .

# 5. Install Python dependencies without storing local cache
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 6. Copy application source code, dataset, models, and static assets
COPY . .

# 7. Expose Flask HTTP port
EXPOSE 5000

# 8. Container Healthcheck to verify the web portal is responding
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/')" || exit 1

# 9. Launch the Placement Prediction System
CMD ["python", "main.py"]
