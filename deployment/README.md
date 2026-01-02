# Production Deployment Guide

This directory contains all the necessary configuration files and scripts for deploying the EcommerceAgents system to production.

## 🚀 Quick Start

1. **Prerequisites Setup**
   ```bash
   # Ensure you have Docker, kubectl, and helm installed
   docker --version
   kubectl version --client
   helm version
   ```

2. **Configure Environment**
   ```bash
   # Copy and edit the production environment file
   cp deployment/.env.production deployment/.env.local
   # Edit deployment/.env.local with your actual values
   ```

3. **Deploy to Production**
   ```bash
   # Run the deployment script
   ./deployment/deploy.sh
   ```

## 📁 Directory Structure

```
deployment/
├── docker-compose.prod.yml     # Production Docker Compose
├── .env.production            # Environment template
├── deploy.sh                  # Main deployment script
├── nginx/
│   └── nginx.conf            # NGINX load balancer config
├── kubernetes/
│   ├── namespace.yaml        # K8s namespace
│   ├── configmap.yaml        # Configuration maps
│   ├── secrets.yaml          # Secrets template
│   └── web-deployment.yaml   # Main application deployment
├── monitoring/
│   └── prometheus.yml        # Prometheus configuration
└── README.md                 # This file
```

## 🐳 Docker Deployment

For simple Docker-based deployment:

```bash
# Copy environment file
cp deployment/.env.production .env

# Edit .env with your actual values
nano .env

# Deploy with Docker Compose
docker-compose -f deployment/docker-compose.prod.yml up -d
```

### Services Included

- **Web Application** (3 replicas)
- **PostgreSQL Database** with backup
- **Redis Cache** with persistence
- **Elasticsearch** for search and analytics
- **RabbitMQ** for message queuing
- **Background Workers** for async processing
- **NGINX** load balancer
- **Prometheus + Grafana** for monitoring
- **Filebeat** for log aggregation
- **Jaeger** for distributed tracing

## ☸️ Kubernetes Deployment

For scalable Kubernetes deployment:

```bash
# Deploy to Kubernetes
./deployment/deploy.sh

# Or step by step:
kubectl apply -f deployment/kubernetes/namespace.yaml
kubectl apply -f deployment/kubernetes/secrets.yaml
kubectl apply -f deployment/kubernetes/configmap.yaml
kubectl apply -f deployment/kubernetes/web-deployment.yaml
```

### Features

- **Auto-scaling** based on CPU/memory usage
- **High availability** with multi-replica deployment
- **Health checks** and readiness probes
- **Rolling updates** with zero downtime
- **Resource management** with requests/limits
- **Security** with non-root containers

## 🔧 Configuration

### Environment Variables

Key environment variables to configure:

| Variable | Description | Default |
|----------|-------------|---------|
| `SECRET_KEY` | Application secret key | Required |
| `DB_PASSWORD` | Database password | Required |
| `OPENAI_API_KEY` | OpenAI API key | Required |
| `REDIS_PASSWORD` | Redis password | Required |
| `ENVIRONMENT` | Environment name | production |

### Secrets Management

1. **Kubernetes Secrets**
   ```bash
   # Create secrets from file
   kubectl create secret generic ecommerce-secrets \
     --from-env-file=deployment/.env.production \
     -n ecommerce-agents
   ```

2. **External Secret Management**
   - AWS Secrets Manager
   - HashiCorp Vault
   - Azure Key Vault
   - Google Secret Manager

### SSL/TLS Configuration

1. **Obtain SSL Certificates**
   ```bash
   # Using Let's Encrypt
   certbot certonly --nginx -d your-domain.com
   ```

2. **Update NGINX Configuration**
   ```nginx
   ssl_certificate /etc/nginx/ssl/your-domain.pem;
   ssl_certificate_key /etc/nginx/ssl/your-domain.key;
   ```

## 📊 Monitoring & Observability

### Prometheus Metrics

Access metrics at: `http://your-domain.com:9090`

Key metrics monitored:
- Application performance (response time, throughput)
- System resources (CPU, memory, disk)
- Business metrics (conversions, revenue)
- AI agent effectiveness
- Database performance

### Grafana Dashboards

Access dashboards at: `http://your-domain.com:3000`

Pre-configured dashboards:
- System Overview
- Application Performance
- Business Intelligence
- AI Agent Metrics
- Infrastructure Monitoring

### Log Aggregation

Logs are collected via Filebeat and can be viewed in:
- Elasticsearch/Kibana
- Grafana Loki
- External log management platforms

### Distributed Tracing

Jaeger tracing available at: `http://your-domain.com:16686`

## 🔒 Security

### Security Measures Implemented

1. **Container Security**
   - Non-root user execution
   - Minimal base images
   - Regular security updates

2. **Network Security**
   - TLS encryption for all communication
   - Network policies in Kubernetes
   - Rate limiting on API endpoints

3. **Authentication & Authorization**
   - JWT-based authentication
   - Role-based access control
   - API key management

4. **Data Protection**
   - Encryption at rest
   - Secure secret management
   - Regular backups

### Security Checklist

- [ ] Update all default passwords
- [ ] Configure SSL certificates
- [ ] Enable firewall rules
- [ ] Set up intrusion detection
- [ ] Configure backup encryption
- [ ] Implement monitoring alerts
- [ ] Regular security updates

## 🔄 Backup & Recovery

### Automated Backups

Database backups are automated via:
```bash
# Daily PostgreSQL backup
kubectl create cronjob postgres-backup \
  --image=postgres:15 \
  --schedule="0 2 * * *" \
  -- pg_dump -h postgres -U ecommerce ecommerce_prod
```

### Disaster Recovery

1. **Database Recovery**
   ```bash
   # Restore from backup
   psql -h postgres -U ecommerce -d ecommerce_prod < backup.sql
   ```

2. **Application Recovery**
   ```bash
   # Rollback deployment
   kubectl rollout undo deployment/ecommerce-web -n ecommerce-agents
   ```

## 🚀 Scaling

### Horizontal Pod Autoscaler

Automatic scaling based on metrics:
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ecommerce-web-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ecommerce-web
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

### Manual Scaling

```bash
# Scale web application
kubectl scale deployment ecommerce-web --replicas=5 -n ecommerce-agents

# Scale workers
kubectl scale deployment ecommerce-worker-recommendations --replicas=3 -n ecommerce-agents
```

## 🛠️ Maintenance

### Regular Maintenance Tasks

1. **Weekly**
   - Review monitoring dashboards
   - Check application logs
   - Verify backup integrity

2. **Monthly**
   - Update container images
   - Review resource usage
   - Performance optimization

3. **Quarterly**
   - Security audit
   - Disaster recovery testing
   - Architecture review

### Health Checks

Monitor application health:
```bash
# Application health
curl https://your-domain.com/health

# Database health
kubectl exec -it postgres-0 -n ecommerce-agents -- pg_isready

# Redis health
kubectl exec -it redis-0 -n ecommerce-agents -- redis-cli ping
```

## 🆘 Troubleshooting

### Common Issues

1. **Pod Startup Issues**
   ```bash
   kubectl describe pod <pod-name> -n ecommerce-agents
   kubectl logs <pod-name> -n ecommerce-agents
   ```

2. **Database Connection Issues**
   ```bash
   kubectl exec -it <web-pod> -n ecommerce-agents -- nc -zv postgres 5432
   ```

3. **Performance Issues**
   ```bash
   kubectl top pods -n ecommerce-agents
   kubectl top nodes
   ```

### Getting Help

- Check application logs
- Review monitoring dashboards
- Verify configuration files
- Test network connectivity

## 📞 Support

For deployment support:
- Create an issue in the repository
- Check the documentation
- Review monitoring dashboards
- Contact the development team

---

**Note**: This deployment is production-ready but should be customized based on your specific infrastructure and requirements.