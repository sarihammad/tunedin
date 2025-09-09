# TunedIn Deployment Guide

## Prerequisites

- Docker & Docker Compose
- Make (optional, for convenience)
- 4GB RAM minimum
- 10GB disk space

## Quick Start

### 1. Clone and Setup

```bash
git clone <repository-url>
cd tunedin
cp env.example .env
```

### 2. Run Full Pipeline

```bash
make pipeline
```

This will:

1. Download sample data
2. Train the LightGCN model
3. Export embeddings
4. Build FAISS index
5. Start all services

### 3. Access the Application

- **Frontend**: http://localhost:5173
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3001 (admin/admin)

## Manual Setup

### Step 1: Data Preparation

```bash
make seed
```

### Step 2: Model Training

```bash
make train
```

### Step 3: Export Embeddings

```bash
make export
```

### Step 4: Build FAISS Index

```bash
make faiss
```

### Step 5: Start Services

```bash
make up
```

## Configuration

### Environment Variables

Edit `.env` file:

```bash
# Database
POSTGRES_DSN=postgresql://postgres:postgres@postgres:5432/postgres

# Redis
REDIS_URL=redis://redis:6379/0

# Service Configuration
SERVICE_PORT=8000
RECS_TOPK=50

# Model Paths
FAISS_INDEX_PATH=/models/index.faiss
ITEM_IDS_PATH=/models/item_ids.npy
USER_EMB_PATH=/models/user_emb.npy
ITEM_EMB_PATH=/models/item_emb.npy

# Fusion Parameters
FUSION_ALPHA=0.7
```

### Docker Compose Configuration

The `docker/docker-compose.yml` file configures:

- **Backend**: FastAPI service
- **Frontend**: React application
- **Redis**: Caching layer
- **Prometheus**: Metrics collection
- **Grafana**: Monitoring dashboard

## Production Deployment

### 1. Build Production Images

```bash
docker-compose -f docker/docker-compose.yml build
```

### 2. Configure Production Environment

```bash
# Set production environment variables
export NODE_ENV=production
export DEBUG=false
export LOG_LEVEL=info
```

### 3. Deploy with Docker Compose

```bash
docker-compose -f docker/docker-compose.yml up -d
```

### 4. Verify Deployment

```bash
# Check service health
curl http://localhost:8000/healthz

# Check metrics
curl http://localhost:8000/metrics

# Test recommendations
curl "http://localhost:8000/rec/users/1?n=10"
```

## Kubernetes Deployment

### 1. Create Namespace

```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: tunedin
```

### 2. Deploy Redis

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: redis
  namespace: tunedin
spec:
  replicas: 1
  selector:
    matchLabels:
      app: redis
  template:
    metadata:
      labels:
        app: redis
    spec:
      containers:
        - name: redis
          image: redis:7-alpine
          ports:
            - containerPort: 6379
```

### 3. Deploy Backend

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: backend
  namespace: tunedin
spec:
  replicas: 3
  selector:
    matchLabels:
      app: backend
  template:
    metadata:
      labels:
        app: backend
    spec:
      containers:
        - name: backend
          image: tunedin-backend:latest
          ports:
            - containerPort: 8000
          env:
            - name: REDIS_URL
              value: "redis://redis:6379/0"
            - name: SERVICE_PORT
              value: "8000"
```

### 4. Deploy Frontend

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: frontend
  namespace: tunedin
spec:
  replicas: 2
  selector:
    matchLabels:
      app: frontend
  template:
    metadata:
      labels:
        app: frontend
    spec:
      containers:
        - name: frontend
          image: tunedin-frontend:latest
          ports:
            - containerPort: 5173
```

## Monitoring Setup

### Prometheus Configuration

```yaml
# docker/prometheus/prometheus.yml
global:
  scrape_interval: 5s

scrape_configs:
  - job_name: "backend"
    static_configs:
      - targets: ["backend:8000"]
    metrics_path: "/metrics"
```

### Grafana Dashboard

The system includes a pre-configured dashboard with:

- Request rate and latency
- FAISS search performance
- Cache hit ratios
- Error rates

## Troubleshooting

### Common Issues

#### 1. Models Not Loading

```bash
# Check if model files exist
ls -la models/

# Verify file permissions
chmod 644 models/*.npy models/*.faiss
```

#### 2. Redis Connection Issues

```bash
# Check Redis status
docker-compose -f docker/docker-compose.yml ps redis

# Check Redis logs
docker-compose -f docker/docker-compose.yml logs redis
```

#### 3. High Memory Usage

```bash
# Monitor memory usage
docker stats

# Adjust FAISS index type (use IVF-PQ for lower memory)
cd ml && python -m models.build_faiss --index-type ivf-pq
```

#### 4. Slow Recommendations

```bash
# Check FAISS index performance
curl http://localhost:8000/status

# Monitor metrics
curl http://localhost:8000/metrics | grep faiss_latency
```

### Logs and Debugging

#### View Service Logs

```bash
# All services
make logs

# Specific service
docker-compose -f docker/docker-compose.yml logs backend
```

#### Debug Mode

```bash
# Enable debug logging
export DEBUG=true
export LOG_LEVEL=debug

# Restart services
make rebuild
```

## Performance Tuning

### Backend Optimization

- **Worker Processes**: Increase uvicorn workers
- **Connection Pooling**: Configure Redis connection pool
- **Caching**: Tune Redis TTL values

### FAISS Optimization

- **Index Type**: HNSW for recall, IVF-PQ for memory
- **Search Parameters**: Tune efSearch for speed/quality
- **Batch Processing**: Process multiple queries together

### Frontend Optimization

- **Code Splitting**: Lazy load components
- **Caching**: Implement service worker
- **Bundle Size**: Optimize dependencies

## Security Hardening

### Production Checklist

- [ ] Change default passwords
- [ ] Enable HTTPS/TLS
- [ ] Configure firewall rules
- [ ] Set up log rotation
- [ ] Enable audit logging
- [ ] Configure backup strategy

### Network Security

```bash
# Restrict network access
docker-compose -f docker/docker-compose.yml up -d --scale frontend=0
# Access via reverse proxy only
```

## Backup and Recovery

### Model Backup

```bash
# Backup model files
tar -czf models-backup-$(date +%Y%m%d).tar.gz models/

# Restore models
tar -xzf models-backup-20231201.tar.gz
```

### Data Backup

```bash
# Backup Redis data
docker exec redis redis-cli BGSAVE

# Backup configuration
cp .env .env.backup
cp docker/docker-compose.yml docker-compose.yml.backup
```

## Scaling

### Horizontal Scaling

```bash
# Scale backend
docker-compose -f docker/docker-compose.yml up -d --scale backend=3

# Scale frontend
docker-compose -f docker/docker-compose.yml up -d --scale frontend=2
```

### Load Balancing

```nginx
# nginx.conf
upstream backend {
    server backend1:8000;
    server backend2:8000;
    server backend3:8000;
}

server {
    listen 80;
    location / {
        proxy_pass http://backend;
    }
}
```
