#!/bin/bash

# Production Deployment Script for EcommerceAgents
# This script handles the complete deployment of the e-commerce multi-agent system

set -e  # Exit on any error

# Configuration
PROJECT_NAME="ecommerce-agents"
DOCKER_REGISTRY="your-registry.com"  # Replace with your Docker registry
IMAGE_NAME="${DOCKER_REGISTRY}/${PROJECT_NAME}"
VERSION=$(git rev-parse --short HEAD)
NAMESPACE="ecommerce-agents"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Function to print colored output
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if Docker is installed and running
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        log_error "Docker daemon is not running. Please start Docker."
        exit 1
    fi
    
    # Check if kubectl is installed
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl is not installed. Please install kubectl first."
        exit 1
    fi
    
    # Check if helm is installed
    if ! command -v helm &> /dev/null; then
        log_warning "Helm is not installed. Some features may not be available."
    fi
    
    # Check if git is available
    if ! command -v git &> /dev/null; then
        log_error "Git is not installed. Please install Git first."
        exit 1
    fi
    
    log_info "Prerequisites check completed successfully."
}

# Function to build Docker images
build_images() {
    log_info "Building Docker images..."
    
    # Build main application image
    docker build -t "${IMAGE_NAME}:${VERSION}" -t "${IMAGE_NAME}:latest" .
    
    # Build worker image
    docker build -f Dockerfile.worker -t "${IMAGE_NAME}-worker:${VERSION}" -t "${IMAGE_NAME}-worker:latest" .
    
    log_info "Docker images built successfully."
}

# Function to push Docker images
push_images() {
    log_info "Pushing Docker images to registry..."
    
    # Push main application image
    docker push "${IMAGE_NAME}:${VERSION}"
    docker push "${IMAGE_NAME}:latest"
    
    # Push worker image
    docker push "${IMAGE_NAME}-worker:${VERSION}"
    docker push "${IMAGE_NAME}-worker:latest"
    
    log_info "Docker images pushed successfully."
}

# Function to setup Kubernetes namespace and resources
setup_kubernetes() {
    log_info "Setting up Kubernetes resources..."
    
    # Create namespace
    kubectl apply -f deployment/kubernetes/namespace.yaml
    
    # Apply secrets (make sure these are properly configured)
    kubectl apply -f deployment/kubernetes/secrets.yaml
    
    # Apply config maps
    kubectl apply -f deployment/kubernetes/configmap.yaml
    
    # Apply persistent volume claims
    if [ -f "deployment/kubernetes/pvc.yaml" ]; then
        kubectl apply -f deployment/kubernetes/pvc.yaml
    fi
    
    log_info "Kubernetes namespace and basic resources created."
}

# Function to deploy database
deploy_database() {
    log_info "Deploying PostgreSQL database..."
    
    # Deploy PostgreSQL using Helm if available
    if command -v helm &> /dev/null; then
        helm repo add bitnami https://charts.bitnami.com/bitnami 2>/dev/null || true
        helm repo update
        
        helm upgrade --install postgres bitnami/postgresql \
            --namespace ${NAMESPACE} \
            --set auth.postgresPassword="$(kubectl get secret ecommerce-secrets -n ${NAMESPACE} -o jsonpath='{.data.DB_PASSWORD}' | base64 -d)" \
            --set auth.database=ecommerce_prod \
            --set auth.username=ecommerce \
            --set primary.persistence.enabled=true \
            --set primary.persistence.size=50Gi \
            --set primary.resources.requests.memory=1Gi \
            --set primary.resources.requests.cpu=500m \
            --set primary.resources.limits.memory=2Gi \
            --set primary.resources.limits.cpu=1000m \
            --wait
    else
        log_warning "Helm not available. Please deploy PostgreSQL manually."
    fi
    
    log_info "PostgreSQL deployment completed."
}

# Function to deploy Redis
deploy_redis() {
    log_info "Deploying Redis cache..."
    
    if command -v helm &> /dev/null; then
        helm upgrade --install redis bitnami/redis \
            --namespace ${NAMESPACE} \
            --set auth.password="$(kubectl get secret ecommerce-secrets -n ${NAMESPACE} -o jsonpath='{.data.REDIS_PASSWORD}' | base64 -d)" \
            --set master.persistence.enabled=true \
            --set master.persistence.size=10Gi \
            --set replica.replicaCount=2 \
            --set replica.persistence.enabled=true \
            --set replica.persistence.size=10Gi \
            --wait
    else
        log_warning "Helm not available. Please deploy Redis manually."
    fi
    
    log_info "Redis deployment completed."
}

# Function to deploy Elasticsearch
deploy_elasticsearch() {
    log_info "Deploying Elasticsearch..."
    
    if command -v helm &> /dev/null; then
        helm repo add elastic https://helm.elastic.co 2>/dev/null || true
        helm repo update
        
        helm upgrade --install elasticsearch elastic/elasticsearch \
            --namespace ${NAMESPACE} \
            --set replicas=3 \
            --set minimumMasterNodes=2 \
            --set resources.requests.cpu=1000m \
            --set resources.requests.memory=2Gi \
            --set resources.limits.cpu=2000m \
            --set resources.limits.memory=4Gi \
            --set volumeClaimTemplate.resources.requests.storage=50Gi \
            --set esConfig.elasticsearch\\.yml.xpack\\.security\\.enabled=false \
            --wait
    else
        log_warning "Helm not available. Please deploy Elasticsearch manually."
    fi
    
    log_info "Elasticsearch deployment completed."
}

# Function to deploy monitoring stack
deploy_monitoring() {
    log_info "Deploying monitoring stack (Prometheus + Grafana)..."
    
    if command -v helm &> /dev/null; then
        # Add Prometheus community repo
        helm repo add prometheus-community https://prometheus-community.github.io/helm-charts 2>/dev/null || true
        helm repo update
        
        # Deploy kube-prometheus-stack
        helm upgrade --install monitoring prometheus-community/kube-prometheus-stack \
            --namespace ${NAMESPACE} \
            --set prometheus.prometheusSpec.retention=30d \
            --set prometheus.prometheusSpec.storageSpec.volumeClaimTemplate.spec.resources.requests.storage=50Gi \
            --set grafana.adminPassword="$(kubectl get secret ecommerce-secrets -n ${NAMESPACE} -o jsonpath='{.data.GRAFANA_PASSWORD}' | base64 -d)" \
            --set grafana.persistence.enabled=true \
            --set grafana.persistence.size=10Gi \
            --wait
    else
        log_warning "Helm not available. Please deploy monitoring stack manually."
    fi
    
    log_info "Monitoring stack deployment completed."
}

# Function to deploy main application
deploy_application() {
    log_info "Deploying main application..."
    
    # Update image tag in deployment
    sed -i "s|image: ecommerce-agents:latest|image: ${IMAGE_NAME}:${VERSION}|g" deployment/kubernetes/web-deployment.yaml
    
    # Apply deployments
    kubectl apply -f deployment/kubernetes/web-deployment.yaml
    
    # Wait for rollout to complete
    kubectl rollout status deployment/ecommerce-web -n ${NAMESPACE} --timeout=600s
    
    log_info "Application deployment completed."
}

# Function to deploy workers
deploy_workers() {
    log_info "Deploying background workers..."
    
    if [ -f "deployment/kubernetes/workers-deployment.yaml" ]; then
        # Update image tag in worker deployment
        sed -i "s|image: ecommerce-agents-worker:latest|image: ${IMAGE_NAME}-worker:${VERSION}|g" deployment/kubernetes/workers-deployment.yaml
        
        kubectl apply -f deployment/kubernetes/workers-deployment.yaml
        
        # Wait for worker deployments
        kubectl rollout status deployment/ecommerce-worker-recommendations -n ${NAMESPACE} --timeout=300s
        kubectl rollout status deployment/ecommerce-worker-reviews -n ${NAMESPACE} --timeout=300s
        kubectl rollout status deployment/ecommerce-worker-descriptions -n ${NAMESPACE} --timeout=300s
    else
        log_warning "Worker deployment file not found. Skipping worker deployment."
    fi
    
    log_info "Worker deployment completed."
}

# Function to setup ingress
setup_ingress() {
    log_info "Setting up ingress controller..."
    
    if [ -f "deployment/kubernetes/ingress.yaml" ]; then
        kubectl apply -f deployment/kubernetes/ingress.yaml
        log_info "Ingress configured successfully."
    else
        log_warning "Ingress configuration file not found. Please configure ingress manually."
    fi
}

# Function to run database migrations
run_migrations() {
    log_info "Running database migrations..."
    
    # Create a temporary job to run migrations
    kubectl create job --from=cronjob/ecommerce-migration-job ecommerce-migration-$(date +%s) -n ${NAMESPACE} 2>/dev/null || \
    kubectl run migration-temp --rm -i --restart=Never --image=${IMAGE_NAME}:${VERSION} \
        --env="DATABASE_URL=postgresql://ecommerce:$(kubectl get secret ecommerce-secrets -n ${NAMESPACE} -o jsonpath='{.data.DB_PASSWORD}' | base64 -d)@postgres-postgresql:5432/ecommerce_prod" \
        -- alembic upgrade head
    
    log_info "Database migrations completed."
}

# Function to verify deployment
verify_deployment() {
    log_info "Verifying deployment..."
    
    # Check pod status
    kubectl get pods -n ${NAMESPACE}
    
    # Check service status
    kubectl get services -n ${NAMESPACE}
    
    # Wait for all pods to be ready
    kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=ecommerce-agents -n ${NAMESPACE} --timeout=300s
    
    # Test health endpoints
    log_info "Testing application health..."
    
    # Port forward for health check
    kubectl port-forward service/ecommerce-web-service 8080:80 -n ${NAMESPACE} &
    PORT_FORWARD_PID=$!
    
    sleep 10
    
    # Test health endpoint
    if curl -f http://localhost:8080/health > /dev/null 2>&1; then
        log_info "Application health check passed."
    else
        log_error "Application health check failed."
        kill $PORT_FORWARD_PID 2>/dev/null || true
        exit 1
    fi
    
    # Stop port forwarding
    kill $PORT_FORWARD_PID 2>/dev/null || true
    
    log_info "Deployment verification completed successfully."
}

# Function to display post-deployment information
post_deployment_info() {
    log_info "Post-deployment information:"
    
    echo "=================================="
    echo "Deployment Summary"
    echo "=================================="
    echo "Project: ${PROJECT_NAME}"
    echo "Version: ${VERSION}"
    echo "Namespace: ${NAMESPACE}"
    echo ""
    
    # Get external IPs/URLs
    echo "Access URLs:"
    kubectl get ingress -n ${NAMESPACE} 2>/dev/null || echo "No ingress configured"
    
    echo ""
    echo "Services:"
    kubectl get services -n ${NAMESPACE}
    
    echo ""
    echo "Pods:"
    kubectl get pods -n ${NAMESPACE}
    
    echo ""
    echo "Persistent Volumes:"
    kubectl get pvc -n ${NAMESPACE}
    
    echo ""
    echo "=================================="
    echo "Next Steps:"
    echo "1. Configure DNS to point to your ingress controller"
    echo "2. Set up SSL certificates if not already done"
    echo "3. Configure monitoring alerts"
    echo "4. Set up backup schedules"
    echo "5. Review and adjust resource limits based on usage"
    echo "=================================="
}

# Function to cleanup on failure
cleanup_on_failure() {
    log_error "Deployment failed. Cleaning up..."
    
    # Stop any port forwarding
    pkill -f "kubectl port-forward" 2>/dev/null || true
    
    # Optionally rollback deployments
    kubectl rollout undo deployment/ecommerce-web -n ${NAMESPACE} 2>/dev/null || true
    
    exit 1
}

# Trap cleanup on script exit
trap cleanup_on_failure ERR

# Main deployment function
main() {
    log_info "Starting production deployment of ${PROJECT_NAME}..."
    
    # Parse command line arguments
    SKIP_BUILD=false
    SKIP_PUSH=false
    DEPLOY_ONLY=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --skip-build)
                SKIP_BUILD=true
                shift
                ;;
            --skip-push)
                SKIP_PUSH=true
                shift
                ;;
            --deploy-only)
                DEPLOY_ONLY=true
                shift
                ;;
            --help)
                echo "Usage: $0 [--skip-build] [--skip-push] [--deploy-only] [--help]"
                echo "  --skip-build   Skip Docker image building"
                echo "  --skip-push    Skip pushing images to registry"
                echo "  --deploy-only  Skip build and push, only deploy"
                echo "  --help         Show this help message"
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    # Execute deployment steps
    check_prerequisites
    
    if [ "$DEPLOY_ONLY" = false ]; then
        if [ "$SKIP_BUILD" = false ]; then
            build_images
        fi
        
        if [ "$SKIP_PUSH" = false ]; then
            push_images
        fi
    fi
    
    setup_kubernetes
    deploy_database
    deploy_redis
    deploy_elasticsearch
    
    # Wait for infrastructure to be ready
    log_info "Waiting for infrastructure to be ready..."
    sleep 30
    
    run_migrations
    deploy_application
    deploy_workers
    deploy_monitoring
    setup_ingress
    
    verify_deployment
    post_deployment_info
    
    log_info "🎉 Production deployment completed successfully!"
}

# Run main function
main "$@"