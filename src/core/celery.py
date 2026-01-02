from celery import Celery
from kombu import Queue
from src.core.config import settings

# Create Celery app
celery_app = Celery(
    "ecommerce_agents",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=[
        "src.tasks.recommendation_tasks",
        "src.tasks.review_tasks", 
        "src.tasks.content_tasks",
        "src.tasks.analytics_tasks"
    ]
)

# Celery configuration
celery_app.conf.update(
    # Task routing
    task_routes={
        "src.tasks.recommendation_tasks.*": {"queue": "recommendations"},
        "src.tasks.review_tasks.*": {"queue": "reviews"},
        "src.tasks.content_tasks.*": {"queue": "content"},
        "src.tasks.analytics_tasks.*": {"queue": "analytics"},
    },
    
    # Queue configuration
    task_create_missing_queues=True,
    task_default_queue="default",
    task_queues=(
        Queue("default", routing_key="default"),
        Queue("recommendations", routing_key="recommendations"),
        Queue("reviews", routing_key="reviews"),
        Queue("content", routing_key="content"),
        Queue("analytics", routing_key="analytics"),
        Queue("priority", routing_key="priority"),
    ),
    
    # Task execution
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    
    # Worker configuration
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=1000,
    worker_disable_rate_limits=True,
    
    # Result backend
    result_expires=3600,  # 1 hour
    result_persistent=True,
    
    # Task configuration
    task_soft_time_limit=300,  # 5 minutes
    task_time_limit=600,  # 10 minutes
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    
    # Monitoring
    worker_send_task_events=True,
    task_send_sent_event=True,
)