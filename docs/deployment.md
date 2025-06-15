# Deployment Guide

## Overview

GeoAirQuality is designed for cloud-native deployment using Docker containers and Kubernetes orchestration. This guide covers deployment strategies, configuration, and operational procedures.

## Architecture

### Components
- **API Service**: FastAPI application with PostGIS integration
- **Data Pipeline**: Dask-based ingestion and processing
- **Edge Nodes**: NVIDIA Jetson preprocessing modules
- **Database**: PostgreSQL with PostGIS extension
- **Cache**: Redis for high-performance queries
- **Monitoring**: Prometheus + Grafana stack

### Deployment Topology
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Load Balancer │────│   API Pods      │────│   PostgreSQL    │
│   (Ingress)     │    │   (3 replicas)  │    │   (PostGIS)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                       ┌─────────────────┐    ┌─────────────────┐
                       │   Redis Cache   │    │   Data Pipeline │
                       │   (Cluster)     │    │   (2 replicas)  │
                       └─────────────────┘    └─────────────────┘
                                                        │
                                               ┌─────────────────┐
                                               │   Edge Nodes    │
                                               │   (Jetson)      │
                                               └─────────────────┘
```

## Prerequisites

### Infrastructure Requirements
- Kubernetes cluster (v1.25+)
- Helm 3.x
- kubectl configured
- Container registry access
- Persistent storage (100GB+ recommended)

### Resource Requirements

#### Minimum (Development)
- **CPU**: 4 cores
- **Memory**: 8GB RAM
- **Storage**: 50GB

#### Production
- **CPU**: 16+ cores
- **Memory**: 32GB+ RAM
- **Storage**: 500GB+ SSD
- **Network**: 1Gbps+

## Docker Images

### Building Images

```bash
# Build API image
docker build -t geoairquality/api:latest ./api

# Build pipeline image
docker build -t geoairquality/pipeline:latest ./data-pipeline

# Build edge node image
docker build -t geoairquality/edge:latest ./edge-nodes/jetson

# Tag for registry
docker tag geoairquality/api:latest your-registry.com/geoairquality/api:v1.0.0
docker tag geoairquality/pipeline:latest your-registry.com/geoairquality/pipeline:v1.0.0
docker tag geoairquality/edge:latest your-registry.com/geoairquality/edge:v1.0.0

# Push to registry
docker push your-registry.com/geoairquality/api:v1.0.0
docker push your-registry.com/geoairquality/pipeline:v1.0.0
docker push your-registry.com/geoairquality/edge:v1.0.0
```

### Multi-stage Build Optimization

The Dockerfiles use multi-stage builds to minimize image size:
- **Base stage**: System dependencies and GDAL
- **Dependencies stage**: Python packages
- **Runtime stage**: Application code only

## Helm Deployment

### Installation

```bash
# Add required Helm repositories
helm repo add bitnami https://charts.bitnami.com/bitnami
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo add grafana https://grafana.github.io/helm-charts
helm repo update

# Install dependencies
helm dependency build ./api/chart

# Deploy to development
helm install geoairquality ./api/chart \
  --namespace geoairquality-dev \
  --create-namespace \
  --values ./api/chart/values-dev.yaml

# Deploy to production
helm install geoairquality ./api/chart \
  --namespace geoairquality-prod \
  --create-namespace \
  --values ./api/chart/values-prod.yaml
```

### Configuration Values

#### Development (values-dev.yaml)
```yaml
replicaCount:
  api: 1
  pipeline: 1

resources:
  api:
    requests:
      memory: "256Mi"
      cpu: "100m"
    limits:
      memory: "512Mi"
      cpu: "250m"

postgresql:
  enabled: true
  auth:
    database: geoairquality_dev
    username: geoair_user
  primary:
    persistence:
      size: 20Gi

redis:
  enabled: true
  auth:
    enabled: false
  master:
    persistence:
      size: 5Gi

monitoring:
  enabled: false
```

#### Production (values-prod.yaml)
```yaml
replicaCount:
  api: 3
  pipeline: 2

resources:
  api:
    requests:
      memory: "512Mi"
      cpu: "250m"
    limits:
      memory: "1Gi"
      cpu: "500m"
  pipeline:
    requests:
      memory: "1Gi"
      cpu: "500m"
    limits:
      memory: "4Gi"
      cpu: "2"

postgresql:
  enabled: true
  auth:
    database: geoairquality
    username: geoair_user
  primary:
    persistence:
      size: 500Gi
      storageClass: fast-ssd
  metrics:
    enabled: true

redis:
  enabled: true
  auth:
    enabled: true
  master:
    persistence:
      size: 50Gi
      storageClass: fast-ssd
  replica:
    replicaCount: 2

autoscaling:
  enabled: true
  minReplicas: 3
  maxReplicas: 10
  targetCPUUtilizationPercentage: 70
  targetMemoryUtilizationPercentage: 80

monitoring:
  enabled: true
  prometheus:
    enabled: true
  grafana:
    enabled: true
```

## Horizontal Pod Autoscaler (HPA)

### Configuration

The HPA automatically scales pods based on CPU and memory utilization:

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: geoairquality-api-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: geoairquality-api
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 10
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
```

### Scaling Policies

- **Scale Up**: 50% increase every 60 seconds when thresholds exceeded
- **Scale Down**: 10% decrease every 60 seconds with 5-minute stabilization
- **CPU Threshold**: 70% average utilization
- **Memory Threshold**: 80% average utilization

### Custom Metrics (Advanced)

For production environments, consider custom metrics:

```yaml
metrics:
- type: Pods
  pods:
    metric:
      name: api_requests_per_second
    target:
      type: AverageValue
      averageValue: "100"
- type: External
  external:
    metric:
      name: postgres_connections
    target:
      type: Value
      value: "80"
```

## Environment-Specific Deployments

### Development Environment

```bash
# Quick development setup
kubectl create namespace geoairquality-dev

# Deploy with minimal resources
helm install geoairquality-dev ./api/chart \
  --namespace geoairquality-dev \
  --set replicaCount.api=1 \
  --set replicaCount.pipeline=1 \
  --set postgresql.primary.persistence.size=10Gi \
  --set redis.master.persistence.size=2Gi
```

### Staging Environment

```bash
# Staging with production-like configuration
helm install geoairquality-staging ./api/chart \
  --namespace geoairquality-staging \
  --create-namespace \
  --values ./api/chart/values-staging.yaml
```

### Production Environment

```bash
# Production deployment with all features
helm install geoairquality ./api/chart \
  --namespace geoairquality-prod \
  --create-namespace \
  --values ./api/chart/values-prod.yaml \
  --timeout 10m
```

## Database Migrations

### Initial Setup

```bash
# Run migrations in init container
kubectl exec -it deployment/geoairquality-api -- alembic upgrade head

# Verify schema
kubectl exec -it deployment/geoairquality-api -- python -c "from models import *; print('Schema loaded successfully')"
```

### Migration Strategy

1. **Blue-Green Deployment**: Zero-downtime migrations
2. **Rolling Updates**: Backward-compatible schema changes
3. **Maintenance Windows**: Breaking changes during low traffic

## Monitoring and Observability

### Health Checks

```bash
# Check pod health
kubectl get pods -n geoairquality-prod

# View logs
kubectl logs -f deployment/geoairquality-api -n geoairquality-prod

# Check HPA status
kubectl get hpa -n geoairquality-prod
```

### Metrics Endpoints

- **API Health**: `http://api-service/health`
- **API Metrics**: `http://api-service/metrics`
- **Pipeline Status**: `http://pipeline-service/status`
- **Database Metrics**: Exposed via PostgreSQL exporter

## Troubleshooting

### Common Issues

#### Pod Startup Failures
```bash
# Check events
kubectl describe pod <pod-name> -n geoairquality-prod

# Check logs
kubectl logs <pod-name> -n geoairquality-prod --previous
```

#### Database Connection Issues
```bash
# Test database connectivity
kubectl exec -it deployment/geoairquality-api -- \
  python -c "from sqlalchemy import create_engine; engine = create_engine('$DATABASE_URL'); print(engine.execute('SELECT 1').scalar())"
```

#### Performance Issues
```bash
# Check resource usage
kubectl top pods -n geoairquality-prod

# Check HPA metrics
kubectl describe hpa geoairquality-api-hpa -n geoairquality-prod
```

### Recovery Procedures

#### Rollback Deployment
```bash
# Rollback to previous version
helm rollback geoairquality -n geoairquality-prod

# Check rollback status
helm history geoairquality -n geoairquality-prod
```

#### Database Recovery
```bash
# Restore from backup
kubectl exec -it postgresql-primary-0 -- \
  pg_restore -U geoair_user -d geoairquality /backups/latest.dump
```

## Security Considerations

### Network Policies

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: geoairquality-network-policy
spec:
  podSelector:
    matchLabels:
      app: geoairquality-api
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app: nginx-ingress
    ports:
    - protocol: TCP
      port: 8000
```

### Secrets Management

```bash
# Create secrets
kubectl create secret generic geoairquality-secrets \
  --from-literal=database-url="postgresql://user:pass@host:5432/db" \
  --from-literal=redis-password="secure-password" \
  -n geoairquality-prod
```

## Performance Optimization

### Resource Tuning

1. **CPU Requests**: Set based on baseline usage
2. **Memory Limits**: Account for data processing peaks
3. **Storage**: Use SSD for database and cache
4. **Network**: Ensure sufficient bandwidth for data ingestion

### Scaling Strategies

1. **Horizontal Scaling**: Add more pods for increased load
2. **Vertical Scaling**: Increase pod resources for complex queries
3. **Database Scaling**: Read replicas for query distribution
4. **Cache Optimization**: Redis clustering for high availability

## Maintenance

### Regular Tasks

```bash
# Update Helm chart
helm upgrade geoairquality ./api/chart \
  --namespace geoairquality-prod \
  --values ./api/chart/values-prod.yaml

# Database maintenance
kubectl exec -it postgresql-primary-0 -- \
  psql -U geoair_user -d geoairquality -c "VACUUM ANALYZE;"

# Clear Redis cache
kubectl exec -it redis-master-0 -- redis-cli FLUSHALL
```

### Backup Procedures

```bash
# Database backup
kubectl exec -it postgresql-primary-0 -- \
  pg_dump -U geoair_user geoairquality > backup-$(date +%Y%m%d).sql

# Configuration backup
kubectl get configmap geoairquality-config -o yaml > config-backup.yaml
```