# Development Environment Setup Script

#!/bin/bash

set -e

echo "🚀 GeoAirQuality Development Environment Setup"
echo "=============================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is not installed. Please install Python 3.8+ first."
        exit 1
    fi
    
    # Check Node.js (for future frontend development)
    if ! command -v node &> /dev/null; then
        print_warning "Node.js is not installed. You'll need it for frontend development."
    fi
    
    print_success "All prerequisites satisfied!"
}

# Create necessary directories
create_directories() {
    print_status "Creating necessary directories..."
    
    mkdir -p data/raw
    mkdir -p data/processed
    mkdir -p data/cache
    mkdir -p logs
    mkdir -p monitoring/grafana/dashboards
    mkdir -p monitoring/grafana/datasources
    mkdir -p nginx/ssl
    mkdir -p api/migrations/versions
    
    print_success "Directories created!"
}

# Create environment files
create_env_files() {
    print_status "Creating environment configuration files..."
    
    # API Environment
    cat > api/.env << EOF
DATABASE_URL=postgresql+asyncpg://geoair_user:geoair_pass@localhost:5432/geoairquality
REDIS_URL=redis://localhost:6379/0
LOG_LEVEL=INFO
ENVIRONMENT=development
SECRET_KEY=dev-secret-key-change-in-production
CORS_ORIGINS=["http://localhost:3000", "http://localhost:8080"]
EOF

    # Pipeline Environment
    cat > data-pipeline/.env << EOF
DATABASE_URL=postgresql://geoair_user:geoair_pass@localhost:5432/geoairquality
REDIS_URL=redis://localhost:6379/0
DASK_WORKERS=4
DASK_MEMORY_LIMIT=2GB
DASK_DASHBOARD_PORT=8787
GRID_RESOLUTION=0.1
OUTPUT_FORMAT=parquet
EOF

    print_success "Environment files created!"
}

# Create monitoring configuration
create_monitoring_config() {
    print_status "Creating monitoring configuration..."
    
    # Prometheus configuration
    cat > monitoring/prometheus.yml << EOF
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  # - "first_rules.yml"
  # - "second_rules.yml"

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  - job_name: 'geoairquality-api'
    static_configs:
      - targets: ['api:8000']
    metrics_path: '/metrics'
    scrape_interval: 10s

  - job_name: 'postgres-exporter'
    static_configs:
      - targets: ['postgres-exporter:9187']

  - job_name: 'redis-exporter'
    static_configs:
      - targets: ['redis-exporter:9121']
EOF

    # Grafana datasource
    mkdir -p monitoring/grafana/datasources
    cat > monitoring/grafana/datasources/prometheus.yml << EOF
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    editable: true
EOF

    print_success "Monitoring configuration created!"
}

# Create nginx configuration
create_nginx_config() {
    print_status "Creating nginx configuration..."
    
    cat > nginx/nginx.conf << EOF
events {
    worker_connections 1024;
}

http {
    upstream api {
        server api:8000;
    }
    
    upstream grafana {
        server grafana:3000;
    }
    
    upstream dask {
        server pipeline:8787;
    }

    server {
        listen 80;
        server_name localhost;
        
        # API routes
        location /api/ {
            proxy_pass http://api/;
            proxy_set_header Host \$host;
            proxy_set_header X-Real-IP \$remote_addr;
            proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto \$scheme;
        }
        
        # Health check
        location /health {
            proxy_pass http://api/health;
            proxy_set_header Host \$host;
        }
        
        # API documentation
        location /docs {
            proxy_pass http://api/docs;
            proxy_set_header Host \$host;
        }
        
        # Grafana monitoring
        location /grafana/ {
            proxy_pass http://grafana/;
            proxy_set_header Host \$host;
            proxy_set_header X-Real-IP \$remote_addr;
        }
        
        # Dask dashboard
        location /dask/ {
            proxy_pass http://dask/;
            proxy_set_header Host \$host;
            proxy_set_header X-Real-IP \$remote_addr;
        }
        
        # Default route
        location / {
            return 200 'GeoAirQuality Development Environment';
            add_header Content-Type text/plain;
        }
    }
}
EOF

    print_success "Nginx configuration created!"
}

# Setup database initialization
create_db_init() {
    print_status "Creating database initialization script..."
    
    cat > api/migrations/init.sql << EOF
-- Enable PostGIS extension
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS postgis_topology;

-- Create application user with limited privileges
CREATE ROLE app_user WITH LOGIN PASSWORD 'app_pass';
GRANT CONNECT ON DATABASE geoairquality TO app_user;
GRANT USAGE ON SCHEMA public TO app_user;
GRANT CREATE ON SCHEMA public TO app_user;

-- Grant privileges for spatial operations
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO app_user;

-- Create initial indexes for performance
-- These will be expanded by Alembic migrations
EOF

    print_success "Database initialization script created!"
}

# Install Python dependencies
install_dependencies() {
    print_status "Installing Python dependencies..."
    
    # API dependencies
    if [ -f "api/requirements.txt" ]; then
        print_status "Installing API dependencies..."
        cd api
        python3 -m venv venv
        source venv/bin/activate
        pip install --upgrade pip
        pip install -r requirements.txt
        deactivate
        cd ..
        print_success "API dependencies installed!"
    else
        print_warning "api/requirements.txt not found. Skipping API dependencies."
    fi
    
    # Pipeline dependencies
    if [ -f "data-pipeline/requirements.txt" ]; then
        print_status "Installing pipeline dependencies..."
        cd data-pipeline
        python3 -m venv venv
        source venv/bin/activate
        pip install --upgrade pip
        pip install -r requirements.txt
        deactivate
        cd ..
        print_success "Pipeline dependencies installed!"
    else
        print_warning "data-pipeline/requirements.txt not found. Skipping pipeline dependencies."
    fi
}

# Download sample data
setup_sample_data() {
    print_status "Setting up sample data..."
    
    # Create sample air quality data if not exists
    if [ ! -f "data/raw/sample_air_quality.csv" ]; then
        cat > data/raw/sample_air_quality.csv << EOF
timestamp,latitude,longitude,pm25,pm10,no2,o3,co,so2,aqi
2024-01-01T00:00:00Z,40.7128,-74.0060,12.5,18.3,25.1,45.2,0.8,2.1,52
2024-01-01T01:00:00Z,40.7589,-73.9851,15.2,21.7,28.4,42.8,0.9,2.3,58
2024-01-01T02:00:00Z,40.6782,-73.9442,18.7,25.1,31.2,38.9,1.1,2.7,65
EOF
        print_success "Sample air quality data created!"
    fi
    
    # Create sample weather data if not exists
    if [ ! -f "data/raw/sample_weather.csv" ]; then
        cat > data/raw/sample_weather.csv << EOF
timestamp,latitude,longitude,temperature,humidity,pressure,wind_speed,wind_direction
2024-01-01T00:00:00Z,40.7128,-74.0060,15.2,68.5,1013.2,3.2,180
2024-01-01T01:00:00Z,40.7589,-73.9851,14.8,70.1,1013.5,2.8,165
2024-01-01T02:00:00Z,40.6782,-73.9442,14.1,72.3,1014.1,2.1,155
EOF
        print_success "Sample weather data created!"
    fi
}

# Start services
start_services() {
    print_status "Starting development services..."
    
    # Build and start services
    docker-compose build
    docker-compose up -d
    
    # Wait for services to be ready
    print_status "Waiting for services to be ready..."
    sleep 10
    
    # Check service health
    for i in {1..30}; do
        if curl -f http://localhost:8000/health &> /dev/null; then
            print_success "API service is ready!"
            break
        fi
        if [ $i -eq 30 ]; then
            print_error "API service failed to start within 30 seconds"
            exit 1
        fi
        sleep 1
    done
}

# Run database migrations
run_migrations() {
    print_status "Running database migrations..."
    
    # Wait a bit more for database to be fully ready
    sleep 5
    
    # Run Alembic migrations
    docker-compose exec api alembic upgrade head || {
        print_warning "Migrations failed or no migrations found. This is normal for first setup."
    }
    
    print_success "Database setup completed!"
}

# Display success message
show_success_message() {
    echo ""
    echo "=============================================="
    print_success "GeoAirQuality Development Environment Ready!"
    echo "=============================================="
    echo ""
    echo "🌐 Services Available:"
    echo "   • API: http://localhost:8000"
    echo "   • API Docs: http://localhost:8000/docs"
    echo "   • Grafana: http://localhost:3000 (admin/admin)"
    echo "   • Prometheus: http://localhost:9090"
    echo "   • Dask Dashboard: http://localhost:8787"
    echo "   • PgAdmin: http://localhost:5050 (admin@geoairquality.com/admin)"
    echo ""
    echo "🔧 Management Commands:"
    echo "   • View logs: docker-compose logs -f"
    echo "   • Stop services: docker-compose down"
    echo "   • Rebuild: docker-compose build"
    echo "   • Database shell: docker-compose exec postgres psql -U geoair_user -d geoairquality"
    echo ""
    echo "📊 Development Workflow:"
    echo "   1. API development: Edit files in ./api/"
    echo "   2. Pipeline development: Edit files in ./data-pipeline/"
    echo "   3. Check health: curl http://localhost:8000/health"
    echo "   4. Run tests: docker-compose exec api pytest"
    echo ""
    print_success "Happy coding! 🚀"
}

# Main execution
main() {
    check_prerequisites
    create_directories
    create_env_files
    create_monitoring_config
    create_nginx_config
    create_db_init
    install_dependencies
    setup_sample_data
    start_services
    run_migrations
    show_success_message
}

# Run main function
main "$@"
