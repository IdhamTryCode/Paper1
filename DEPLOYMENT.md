# Deployment Guide

This guide provides comprehensive instructions for deploying the Rust vs Python ML Benchmark System in various environments.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Local Development Setup](#local-development-setup)
3. [Docker Deployment](#docker-deployment)
4. [Cloud Deployment](#cloud-deployment)
5. [Production Deployment](#production-deployment)
6. [Monitoring and Maintenance](#monitoring-and-maintenance)
7. [Troubleshooting](#troubleshooting)

## Prerequisites

### System Requirements

- **CPU**: 8+ cores recommended
- **Memory**: 16+ GB RAM
- **Storage**: 50+ GB free space
- **GPU**: NVIDIA GPU with CUDA support (optional but recommended)
- **OS**: Linux (Ubuntu 20.04+), macOS (10.15+), or Windows 10+

### Software Requirements

- **Python**: 3.9+
- **Rust**: 1.70+
- **Docker**: 20.10+
- **Git**: 2.30+
- **Nextflow**: 22.10+

### GPU Requirements (Optional)

- **CUDA**: 11.8+
- **cuDNN**: 8.6+
- **NVIDIA Driver**: 470+

## Local Development Setup

### 1. Clone the Repository

```bash
git clone https://github.com/your-org/rust-ml-benchmark.git
cd rust-ml-benchmark
```

### 2. Install Python Dependencies

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Install Rust Dependencies

```bash
# Install Rust (if not already installed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env

# Install additional Rust tools
cargo install cargo-audit
cargo install cargo-tarpaulin
```

### 4. Build Rust Benchmarks

```bash
# Build all Rust benchmarks
find src/rust -name "Cargo.toml" -execdir cargo build --release \;
```

### 5. Verify Installation

```bash
# Run tests
python -m pytest tests/ -v

# Test individual benchmarks
python src/python/classical_ml/regression_benchmark.py --help
```

## Docker Deployment

### 1. Build Docker Images

```bash
# Build Python image
docker build -t rust-ml-benchmark:python containers/python/

# Build Rust image
docker build -t rust-ml-benchmark:rust containers/rust/
```

### 2. Run Benchmarks in Containers

```bash
# Run Python regression benchmark
docker run --rm -v $(pwd)/results:/app/results rust-ml-benchmark:python \
  python src/python/classical_ml/regression_benchmark.py \
  --dataset boston_housing --algorithm linear --mode training

# Run Rust regression benchmark
docker run --rm -v $(pwd)/results:/app/results rust-ml-benchmark:rust \
  cargo run --release --bin regression_benchmark \
  -- --dataset boston_housing --algorithm linear --mode training
```

### 3. Run Complete Pipeline

```bash
# Run Nextflow pipeline
nextflow run main.nf -profile docker
```

## Cloud Deployment

### AWS Deployment

#### 1. EC2 Setup

```bash
# Launch EC2 instance with GPU support
aws ec2 run-instances \
  --image-id ami-0c02fb55956c7d316 \
  --instance-type g4dn.xlarge \
  --key-name your-key-pair \
  --security-group-ids sg-xxxxxxxxx
```

#### 2. Install Dependencies

```bash
# Update system
sudo apt-get update && sudo apt-get upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Install Nextflow
curl -s https://get.nextflow.io | bash
sudo mv nextflow /usr/local/bin/
```

#### 3. Deploy Application

```bash
# Clone repository
git clone https://github.com/your-org/rust-ml-benchmark.git
cd rust-ml-benchmark

# Build and run
docker-compose up -d
```

### Google Cloud Platform

#### 1. Create Compute Instance

```bash
# Create instance with GPU
gcloud compute instances create benchmark-instance \
  --zone=us-central1-a \
  --machine-type=n1-standard-4 \
  --accelerator="type=nvidia-tesla-t4,count=1" \
  --image-family=debian-11 \
  --image-project=debian-cloud
```

#### 2. Setup Environment

```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Install NVIDIA Docker
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

### Azure Deployment

#### 1. Create VM with GPU

```bash
# Create resource group
az group create --name benchmark-rg --location eastus

# Create VM with GPU
az vm create \
  --resource-group benchmark-rg \
  --name benchmark-vm \
  --image UbuntuLTS \
  --size Standard_NC6s_v3 \
  --admin-username azureuser \
  --generate-ssh-keys
```

## Production Deployment

### 1. Environment Configuration

Create environment-specific configuration files:

```bash
# Production configuration
cp config/benchmarks.yaml config/benchmarks.prod.yaml
cp config/frameworks.yaml config/frameworks.prod.yaml
cp config/hardware.yaml config/hardware.prod.yaml
```

### 2. Security Setup

```bash
# Create service account
sudo useradd -r -s /bin/false benchmark-user

# Set up SSL certificates
sudo mkdir -p /etc/ssl/benchmark
sudo openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout /etc/ssl/benchmark/benchmark.key \
  -out /etc/ssl/benchmark/benchmark.crt
```

### 3. Database Setup

```bash
# Install PostgreSQL
sudo apt-get install postgresql postgresql-contrib

# Create database
sudo -u postgres createdb benchmark_db
sudo -u postgres createuser benchmark_user
```

### 4. Monitoring Setup

```bash
# Install Prometheus
wget https://github.com/prometheus/prometheus/releases/download/v2.40.0/prometheus-2.40.0.linux-amd64.tar.gz
tar xvf prometheus-*.tar.gz
cd prometheus-*

# Install Grafana
sudo apt-get install -y grafana
sudo systemctl enable grafana-server
sudo systemctl start grafana-server
```

### 5. Load Balancer Setup

```bash
# Install Nginx
sudo apt-get install nginx

# Configure Nginx
sudo tee /etc/nginx/sites-available/benchmark <<EOF
server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://localhost:8080;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
    }
}
EOF

sudo ln -s /etc/nginx/sites-available/benchmark /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

## Monitoring and Maintenance

### 1. Health Checks

Create health check scripts:

```bash
#!/bin/bash
# health_check.sh

# Check if benchmarks are running
if pgrep -f "regression_benchmark" > /dev/null; then
    echo "Benchmarks are running"
else
    echo "Benchmarks are not running"
    exit 1
fi

# Check disk space
DISK_USAGE=$(df / | awk 'NR==2 {print $5}' | sed 's/%//')
if [ $DISK_USAGE -gt 90 ]; then
    echo "Disk usage is high: ${DISK_USAGE}%"
    exit 1
fi

# Check memory usage
MEM_USAGE=$(free | awk 'NR==2{printf "%.0f", $3*100/$2}')
if [ $MEM_USAGE -gt 90 ]; then
    echo "Memory usage is high: ${MEM_USAGE}%"
    exit 1
fi
```

### 2. Log Management

```bash
# Configure log rotation
sudo tee /etc/logrotate.d/benchmark <<EOF
/var/log/benchmark/*.log {
    daily
    missingok
    rotate 52
    compress
    delaycompress
    notifempty
    create 644 benchmark-user benchmark-user
}
EOF
```

### 3. Backup Strategy

```bash
#!/bin/bash
# backup.sh

BACKUP_DIR="/backup/benchmark"
DATE=$(date +%Y%m%d_%H%M%S)

# Create backup directory
mkdir -p $BACKUP_DIR

# Backup results
tar -czf $BACKUP_DIR/results_$DATE.tar.gz results/

# Backup configuration
tar -czf $BACKUP_DIR/config_$DATE.tar.gz config/

# Backup database
pg_dump benchmark_db > $BACKUP_DIR/database_$DATE.sql

# Clean old backups (keep last 30 days)
find $BACKUP_DIR -name "*.tar.gz" -mtime +30 -delete
find $BACKUP_DIR -name "*.sql" -mtime +30 -delete
```

### 4. Performance Monitoring

```bash
#!/bin/bash
# monitor_performance.sh

# Monitor CPU usage
CPU_USAGE=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)
echo "CPU Usage: ${CPU_USAGE}%"

# Monitor memory usage
MEM_USAGE=$(free | awk 'NR==2{printf "%.0f", $3*100/$2}')
echo "Memory Usage: ${MEM_USAGE}%"

# Monitor disk I/O
DISK_IO=$(iostat -x 1 1 | awk 'NR==4 {print $2}')
echo "Disk I/O: ${DISK_IO}"

# Monitor network usage
NET_IO=$(netstat -i | awk 'NR==3 {print $3}')
echo "Network I/O: ${NET_IO}"
```

## Troubleshooting

### Common Issues

#### 1. GPU Not Detected

```bash
# Check GPU availability
nvidia-smi

# Check CUDA installation
nvcc --version

# Check PyTorch CUDA support
python -c "import torch; print(torch.cuda.is_available())"
```

#### 2. Memory Issues

```bash
# Check memory usage
free -h

# Check swap usage
swapon --show

# Increase swap if needed
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

#### 3. Docker Issues

```bash
# Check Docker status
sudo systemctl status docker

# Restart Docker
sudo systemctl restart docker

# Check Docker logs
sudo journalctl -u docker.service
```

#### 4. Nextflow Issues

```bash
# Check Nextflow installation
nextflow -version

# Clean Nextflow cache
rm -rf ~/.nextflow/cache

# Check Nextflow logs
tail -f .nextflow.log
```

### Performance Optimization

#### 1. System Tuning

```bash
# Optimize kernel parameters
echo 'vm.swappiness=10' | sudo tee -a /etc/sysctl.conf
echo 'vm.dirty_ratio=15' | sudo tee -a /etc/sysctl.conf
echo 'vm.dirty_background_ratio=5' | sudo tee -a /etc/sysctl.conf
sudo sysctl -p
```

#### 2. Docker Optimization

```bash
# Optimize Docker daemon
sudo tee /etc/docker/daemon.json <<EOF
{
  "storage-driver": "overlay2",
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "10m",
    "max-file": "3"
  }
}
EOF
sudo systemctl restart docker
```

#### 3. Network Optimization

```bash
# Optimize network settings
echo 'net.core.rmem_max=134217728' | sudo tee -a /etc/sysctl.conf
echo 'net.core.wmem_max=134217728' | sudo tee -a /etc/sysctl.conf
echo 'net.ipv4.tcp_rmem=4096 87380 134217728' | sudo tee -a /etc/sysctl.conf
echo 'net.ipv4.tcp_wmem=4096 65536 134217728' | sudo tee -a /etc/sysctl.conf
sudo sysctl -p
```

## Security Considerations

### 1. Access Control

```bash
# Create restricted user
sudo useradd -r -s /bin/false benchmark-runner

# Set up SSH key authentication
mkdir -p ~/.ssh
chmod 700 ~/.ssh
ssh-keygen -t rsa -b 4096 -f ~/.ssh/benchmark_key
```

### 2. Firewall Configuration

```bash
# Configure UFW firewall
sudo ufw allow ssh
sudo ufw allow 8080/tcp
sudo ufw allow 9090/tcp  # Prometheus
sudo ufw allow 3000/tcp  # Grafana
sudo ufw enable
```

### 3. SSL/TLS Setup

```bash
# Install Certbot
sudo apt-get install certbot python3-certbot-nginx

# Obtain SSL certificate
sudo certbot --nginx -d your-domain.com
```

## Scaling Considerations

### 1. Horizontal Scaling

```bash
# Set up load balancer
sudo apt-get install haproxy

# Configure HAProxy
sudo tee /etc/haproxy/haproxy.cfg <<EOF
global
    daemon

defaults
    mode http
    timeout connect 5000ms
    timeout client 50000ms
    timeout server 50000ms

frontend http_front
    bind *:80
    default_backend http_back

backend http_back
    balance roundrobin
    server benchmark1 192.168.1.10:8080 check
    server benchmark2 192.168.1.11:8080 check
    server benchmark3 192.168.1.12:8080 check
EOF
```

### 2. Vertical Scaling

```bash
# Monitor resource usage
htop
iotop
nethogs

# Scale based on usage patterns
# Add more CPU cores, RAM, or GPU resources as needed
```

## Conclusion

This deployment guide provides comprehensive instructions for deploying the Rust vs Python ML Benchmark System in various environments. Follow the security best practices and monitor the system regularly to ensure optimal performance and reliability.

For additional support, refer to the project documentation or create an issue in the repository. 