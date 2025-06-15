# GeoAirQuality - Environmental Data Engineering Platform

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)

A scalable, production-ready air quality monitoring and prediction platform built with modern data engineering practices. This platform combines real-time data processing, spatial analytics, and high-performance APIs to deliver comprehensive environmental insights.

## 🚀 Features

- **Real-time Data Processing**: Efficient ingestion and processing of air quality data
- **Spatial Analytics**: PostGIS-powered geospatial queries and analysis
- **High-Performance API**: FastAPI with Redis caching for sub-second response times
- **Containerized Deployment**: Docker and Kubernetes ready
- **Comprehensive Documentation**: Detailed guides for deployment and optimization
- **Performance Monitoring**: Built-in metrics and optimization strategies

## 🏗️ Architecture

### Core Components

- **`/api/`** - FastAPI-based REST API with Redis caching and PostGIS integration
- **`/data-pipeline/`** - Data ingestion and processing pipelines
- **`/docs/`** - Comprehensive documentation and guides
- **`/k8s/`** - Kubernetes deployment manifests
- **`/tests/`** - Test suites and validation

### Technology Stack

- **Backend**: FastAPI, SQLAlchemy, Alembic
- **Database**: PostgreSQL with PostGIS extension
- **Caching**: Redis with optimized TTL strategies
- **Containerization**: Docker, Kubernetes
- **Data Processing**: Pandas, GeoPandas
- **Monitoring**: Prometheus-compatible metrics

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Docker and Docker Compose
- PostgreSQL with PostGIS
- Redis (optional, for caching)

### Local Development

1. **Clone the repository**
   ```bash
   git clone https://github.com/kakashi3lite/GeoAirQuality.git
   cd GeoAirQuality
   ```

2. **Set up the API service**
   ```bash
   cd api
   pip install -r requirements.txt
   uvicorn main:app --reload
   ```

3. **Run the data pipeline**
   ```bash
   cd data-pipeline
   pip install -r requirements.txt
   python ingest.py
   ```

### Docker Deployment

1. **Build and run with Docker Compose**
   ```bash
   docker-compose up --build
   ```

2. **Access the API**
   - API Documentation: http://localhost:8000/docs
   - Health Check: http://localhost:8000/health

### Kubernetes Deployment

1. **Deploy to Kubernetes**
   ```bash
   kubectl apply -f k8s/
   ```

2. **Monitor deployment**
   ```bash
   kubectl get pods -l app=geoairquality
   ```

## 📊 API Endpoints

- `GET /health` - Health check endpoint
- `GET /air-quality/stations` - List all monitoring stations
- `GET /air-quality/data` - Query air quality data with filters
- `GET /air-quality/spatial` - Spatial queries within geographic bounds
- `POST /air-quality/batch` - Batch data ingestion

For detailed API documentation, visit `/docs` when running the service.

## 📚 Documentation

- [Deployment Guide](docs/deployment.md) - Comprehensive deployment instructions
- [Performance Optimization](docs/performance.md) - Performance tuning and monitoring
- [Database Schema](docs/database.md) - Database design and migrations
- [Data Pipeline](docs/pipeline.md) - Data processing workflows

## 🔧 Configuration

### Environment Variables

```bash
# Database
DATABASE_URL=postgresql://user:password@localhost/geoairquality

# Redis (optional)
REDIS_URL=redis://localhost:6379

# API Settings
API_HOST=0.0.0.0
API_PORT=8000
```

## 🚀 Performance

- **API Response Time**: < 100ms for cached queries
- **Database Queries**: Optimized with spatial indexing
- **Caching Strategy**: Redis with intelligent TTL management
- **Scalability**: Horizontal scaling with Kubernetes HPA

## 🧪 Testing

```bash
# Run tests
pytest tests/

# Run with coverage
pytest --cov=api tests/
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with modern data engineering best practices
- Optimized for production deployment
- Designed for scalability and performance

---

**GeoAirQuality** - Empowering environmental insights through data engineering excellence.