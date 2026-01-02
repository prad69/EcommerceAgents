import asyncio
from typing import Dict, List, Any, Optional, Tuple, Callable
from loguru import logger
import json
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
import time

from src.core.database import get_db, redis_client
from src.services.recommendation_engine import RecommendationService
from src.services.review_analysis import ReviewAnalysisService
from src.services.chatbot_orchestrator import ChatbotOrchestrator
from src.services.description_generator import DescriptionGeneratorService
from src.services.product_analyzer import ProductAnalyzerService


class AgentType(Enum):
    RECOMMENDATION = "recommendation"
    REVIEW_ANALYSIS = "review_analysis"
    CHATBOT = "chatbot"
    DESCRIPTION_GENERATOR = "description_generator"
    PRODUCT_ANALYZER = "product_analyzer"


class TaskPriority(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class AgentTask:
    """Task for agent execution"""
    task_id: str
    agent_type: AgentType
    task_type: str
    payload: Dict[str, Any]
    priority: TaskPriority = TaskPriority.MEDIUM
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    timeout_seconds: int = 300
    dependencies: List[str] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.dependencies is None:
            self.dependencies = []


@dataclass
class WorkflowStep:
    """Step in an automated workflow"""
    step_id: str
    agent_type: AgentType
    task_type: str
    payload_template: Dict[str, Any]
    dependencies: List[str] = None
    condition: Optional[str] = None
    timeout_seconds: int = 300
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []


@dataclass
class Workflow:
    """Automated workflow definition"""
    workflow_id: str
    name: str
    description: str
    steps: List[WorkflowStep]
    trigger_conditions: Dict[str, Any]
    enabled: bool = True
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()


class AgentOrchestrator:
    """
    Central orchestrator for managing all AI agents and their interactions
    """
    
    def __init__(self):
        # Initialize agent services
        self.agents = {
            AgentType.RECOMMENDATION: RecommendationService(),
            AgentType.REVIEW_ANALYSIS: ReviewAnalysisService(),
            AgentType.CHATBOT: ChatbotOrchestrator(),
            AgentType.DESCRIPTION_GENERATOR: DescriptionGeneratorService(),
            AgentType.PRODUCT_ANALYZER: ProductAnalyzerService()
        }
        
        # Task management
        self.task_queue = asyncio.Queue()
        self.active_tasks = {}
        self.completed_tasks = {}
        self.task_workers = []
        
        # Workflow management
        self.workflows = {}
        self.active_workflow_instances = {}
        
        # Performance monitoring
        self.performance_metrics = {}
        self.agent_health_status = {}
        
        # Communication channels
        self.event_handlers = {}
        self.data_sharing_cache = {}
        
        logger.info("Agent orchestrator initialized")
        
        # Setup default workflows
        self._setup_default_workflows()
        
        # Start background workers
        asyncio.create_task(self._start_task_workers())
        asyncio.create_task(self._start_health_monitor())
    
    def _setup_default_workflows(self):
        """
        Setup default automated workflows
        """
        # Product onboarding workflow
        product_onboarding = Workflow(
            workflow_id="product_onboarding",
            name="Product Onboarding",
            description="Complete product analysis and content generation pipeline",
            steps=[
                WorkflowStep(
                    step_id="analyze_product",
                    agent_type=AgentType.PRODUCT_ANALYZER,
                    task_type="analyze_specifications",
                    payload_template={"product_id": "{product_id}"}
                ),
                WorkflowStep(
                    step_id="generate_descriptions",
                    agent_type=AgentType.DESCRIPTION_GENERATOR,
                    task_type="generate_descriptions",
                    payload_template={
                        "product_id": "{product_id}",
                        "description_types": ["short", "medium", "long", "seo"],
                        "use_analysis": True
                    },
                    dependencies=["analyze_product"]
                ),
                WorkflowStep(
                    step_id="initialize_recommendations",
                    agent_type=AgentType.RECOMMENDATION,
                    task_type="index_product",
                    payload_template={"product_id": "{product_id}"},
                    dependencies=["analyze_product"]
                )
            ],
            trigger_conditions={"event": "product_created"}
        )
        
        # Review processing workflow
        review_processing = Workflow(
            workflow_id="review_processing",
            name="Review Analysis Pipeline",
            description="Analyze new reviews and update recommendations",
            steps=[
                WorkflowStep(
                    step_id="analyze_review",
                    agent_type=AgentType.REVIEW_ANALYSIS,
                    task_type="analyze_review",
                    payload_template={"review_id": "{review_id}"}
                ),
                WorkflowStep(
                    step_id="update_recommendations",
                    agent_type=AgentType.RECOMMENDATION,
                    task_type="update_product_score",
                    payload_template={"product_id": "{product_id}"},
                    dependencies=["analyze_review"]
                )
            ],
            trigger_conditions={"event": "review_created"}
        )
        
        # Customer support workflow
        customer_support = Workflow(
            workflow_id="customer_support_escalation",
            name="Customer Support Escalation",
            description="Handle complex customer queries with agent collaboration",
            steps=[
                WorkflowStep(
                    step_id="analyze_query",
                    agent_type=AgentType.CHATBOT,
                    task_type="analyze_intent",
                    payload_template={"message": "{user_message}", "context": "{context}"}
                ),
                WorkflowStep(
                    step_id="get_recommendations",
                    agent_type=AgentType.RECOMMENDATION,
                    task_type="get_recommendations",
                    payload_template={"user_id": "{user_id}", "context": "{context}"},
                    condition="intent == 'product_search'",
                    dependencies=["analyze_query"]
                ),
                WorkflowStep(
                    step_id="analyze_product_reviews",
                    agent_type=AgentType.REVIEW_ANALYSIS,
                    task_type="get_review_summary",
                    payload_template={"product_id": "{product_id}"},
                    condition="intent == 'product_info'",
                    dependencies=["analyze_query"]
                )
            ],
            trigger_conditions={"event": "complex_customer_query"}
        )
        
        self.workflows = {
            "product_onboarding": product_onboarding,
            "review_processing": review_processing,
            "customer_support_escalation": customer_support
        }
    
    async def _start_task_workers(self):
        """
        Start background workers for processing agent tasks
        """
        worker_count = 5  # Configurable based on system resources
        
        for i in range(worker_count):
            worker = asyncio.create_task(self._task_worker(f"worker_{i}"))
            self.task_workers.append(worker)
        
        logger.info(f"Started {worker_count} task workers")
    
    async def _task_worker(self, worker_id: str):
        """
        Background worker for processing agent tasks
        """
        logger.info(f"Task worker {worker_id} started")
        
        while True:
            try:
                # Get next task from queue
                task = await self.task_queue.get()
                
                if task is None:  # Shutdown signal
                    break
                
                # Check if task dependencies are met
                if not await self._check_task_dependencies(task):
                    # Re-queue task for later
                    await asyncio.sleep(1)
                    await self.task_queue.put(task)
                    continue
                
                # Execute task
                await self._execute_task(task, worker_id)
                
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
                await asyncio.sleep(1)
    
    async def _check_task_dependencies(self, task: AgentTask) -> bool:
        """
        Check if task dependencies are satisfied
        """
        if not task.dependencies:
            return True
        
        for dep_task_id in task.dependencies:
            if dep_task_id not in self.completed_tasks:
                return False
            
            dep_task = self.completed_tasks[dep_task_id]
            if dep_task.status != TaskStatus.COMPLETED:
                return False
        
        return True
    
    async def _execute_task(self, task: AgentTask, worker_id: str):
        """
        Execute an agent task
        """
        try:
            task.status = TaskStatus.IN_PROGRESS
            task.started_at = datetime.utcnow()
            self.active_tasks[task.task_id] = task
            
            logger.info(f"Worker {worker_id} executing task {task.task_id} ({task.agent_type.value}:{task.task_type})")
            
            # Get the appropriate agent
            agent = self.agents.get(task.agent_type)
            if not agent:
                raise ValueError(f"Unknown agent type: {task.agent_type}")
            
            # Execute task with timeout
            start_time = time.time()
            try:
                result = await asyncio.wait_for(
                    self._call_agent_method(agent, task.task_type, task.payload),
                    timeout=task.timeout_seconds
                )
                
                task.result = result
                task.status = TaskStatus.COMPLETED
                task.completed_at = datetime.utcnow()
                
                # Update performance metrics
                execution_time = time.time() - start_time
                self._update_performance_metrics(task.agent_type, task.task_type, execution_time, True)
                
                logger.info(f"Task {task.task_id} completed in {execution_time:.2f}s")
                
            except asyncio.TimeoutError:
                task.status = TaskStatus.FAILED
                task.error_message = f"Task timed out after {task.timeout_seconds} seconds"
                self._update_performance_metrics(task.agent_type, task.task_type, task.timeout_seconds, False)
                
            except Exception as e:
                task.status = TaskStatus.FAILED
                task.error_message = str(e)
                execution_time = time.time() - start_time
                self._update_performance_metrics(task.agent_type, task.task_type, execution_time, False)
                
                # Retry logic
                if task.retry_count < task.max_retries:
                    task.retry_count += 1
                    task.status = TaskStatus.PENDING
                    task.started_at = None
                    logger.warning(f"Task {task.task_id} failed, retrying ({task.retry_count}/{task.max_retries})")
                    await self.task_queue.put(task)
                    return
            
            # Move task from active to completed
            if task.task_id in self.active_tasks:
                del self.active_tasks[task.task_id]
            self.completed_tasks[task.task_id] = task
            
            # Trigger any dependent workflows or tasks
            await self._trigger_dependent_tasks(task)
            
        except Exception as e:
            logger.error(f"Task execution error: {e}")
            task.status = TaskStatus.FAILED
            task.error_message = str(e)
    
    async def _call_agent_method(self, agent: Any, method_name: str, payload: Dict[str, Any]) -> Any:
        """
        Call the appropriate method on an agent
        """
        # Map task types to agent methods
        method_mapping = {
            # Recommendation agent
            "get_recommendations": lambda: agent.get_recommendations(
                user_id=payload.get("user_id"),
                filters=payload.get("filters", {}),
                limit=payload.get("limit", 10)
            ),
            "index_product": lambda: agent.index_product(payload["product_id"]),
            "update_product_score": lambda: agent.update_product_metrics(payload["product_id"]),
            
            # Review analysis agent
            "analyze_review": lambda: agent.analyze_review(payload["review_id"]),
            "get_review_summary": lambda: agent.get_product_review_summary(payload["product_id"]),
            "analyze_sentiment": lambda: agent.analyze_sentiment(payload["text"]),
            
            # Chatbot agent
            "process_message": lambda: agent.process_user_message(
                session_id=payload["session_id"],
                user_message=payload["message"],
                user_id=payload.get("user_id")
            ),
            "analyze_intent": lambda: agent.intent_service.classify_intent(
                payload["message"],
                payload.get("context")
            ),
            
            # Description generator
            "generate_descriptions": lambda: agent.generate_descriptions(payload),
            
            # Product analyzer
            "analyze_specifications": lambda: agent.analyze_product_specifications(payload["product_id"]),
            "analyze_competitive": lambda: agent.analyze_competitive_landscape(
                payload["product_id"],
                payload["category"]
            )
        }
        
        method = method_mapping.get(method_name)
        if not method:
            raise ValueError(f"Unknown method: {method_name}")
        
        return await method()
    
    async def _trigger_dependent_tasks(self, completed_task: AgentTask):
        """
        Trigger any dependent tasks or workflows
        """
        # Check for workflow continuations
        for workflow_id, instance in self.active_workflow_instances.items():
            await self._continue_workflow(workflow_id, instance, completed_task)
    
    def _update_performance_metrics(self, agent_type: AgentType, task_type: str, execution_time: float, success: bool):
        """
        Update performance metrics for agents
        """
        key = f"{agent_type.value}:{task_type}"
        
        if key not in self.performance_metrics:
            self.performance_metrics[key] = {
                "total_executions": 0,
                "successful_executions": 0,
                "failed_executions": 0,
                "total_time": 0.0,
                "avg_time": 0.0,
                "min_time": float('inf'),
                "max_time": 0.0
            }
        
        metrics = self.performance_metrics[key]
        metrics["total_executions"] += 1
        metrics["total_time"] += execution_time
        metrics["avg_time"] = metrics["total_time"] / metrics["total_executions"]
        metrics["min_time"] = min(metrics["min_time"], execution_time)
        metrics["max_time"] = max(metrics["max_time"], execution_time)
        
        if success:
            metrics["successful_executions"] += 1
        else:
            metrics["failed_executions"] += 1
    
    async def _start_health_monitor(self):
        """
        Monitor agent health and performance
        """
        while True:
            try:
                await self._check_agent_health()
                await asyncio.sleep(30)  # Check every 30 seconds
            except Exception as e:
                logger.error(f"Health monitor error: {e}")
                await asyncio.sleep(60)
    
    async def _check_agent_health(self):
        """
        Check health status of all agents
        """
        current_time = datetime.utcnow()
        
        for agent_type in AgentType:
            try:
                # Simple health check - could be expanded
                agent = self.agents.get(agent_type)
                if agent:
                    # Check if agent is responsive
                    start_time = time.time()
                    # Simple test call
                    health_ok = True
                    response_time = time.time() - start_time
                    
                    self.agent_health_status[agent_type.value] = {
                        "status": "healthy" if health_ok else "unhealthy",
                        "response_time": response_time,
                        "last_check": current_time.isoformat(),
                        "active_tasks": len([t for t in self.active_tasks.values() if t.agent_type == agent_type])
                    }
                else:
                    self.agent_health_status[agent_type.value] = {
                        "status": "unavailable",
                        "last_check": current_time.isoformat()
                    }
                    
            except Exception as e:
                self.agent_health_status[agent_type.value] = {
                    "status": "error",
                    "error": str(e),
                    "last_check": current_time.isoformat()
                }
    
    # Public API methods
    
    async def submit_task(
        self,
        agent_type: AgentType,
        task_type: str,
        payload: Dict[str, Any],
        priority: TaskPriority = TaskPriority.MEDIUM,
        dependencies: Optional[List[str]] = None
    ) -> str:
        """
        Submit a task for execution
        """
        task_id = f"task_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        
        task = AgentTask(
            task_id=task_id,
            agent_type=agent_type,
            task_type=task_type,
            payload=payload,
            priority=priority,
            dependencies=dependencies or []
        )
        
        await self.task_queue.put(task)
        logger.info(f"Submitted task {task_id} ({agent_type.value}:{task_type})")
        
        return task_id
    
    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Get status of a submitted task
        """
        # Check active tasks
        if task_id in self.active_tasks:
            task = self.active_tasks[task_id]
            return asdict(task)
        
        # Check completed tasks
        if task_id in self.completed_tasks:
            task = self.completed_tasks[task_id]
            return asdict(task)
        
        return None
    
    async def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a pending or active task
        """
        # Check if task is in active tasks
        if task_id in self.active_tasks:
            task = self.active_tasks[task_id]
            task.status = TaskStatus.CANCELLED
            logger.info(f"Cancelled active task {task_id}")
            return True
        
        return False
    
    async def trigger_workflow(
        self,
        workflow_id: str,
        trigger_data: Dict[str, Any]
    ) -> str:
        """
        Trigger a workflow execution
        """
        if workflow_id not in self.workflows:
            raise ValueError(f"Unknown workflow: {workflow_id}")
        
        workflow = self.workflows[workflow_id]
        if not workflow.enabled:
            raise ValueError(f"Workflow {workflow_id} is disabled")
        
        instance_id = f"workflow_{workflow_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        
        # Create workflow instance
        workflow_instance = {
            "workflow": workflow,
            "trigger_data": trigger_data,
            "started_at": datetime.utcnow(),
            "completed_steps": [],
            "pending_steps": workflow.steps.copy(),
            "status": "running"
        }
        
        self.active_workflow_instances[instance_id] = workflow_instance
        
        # Start executing workflow
        await self._execute_workflow(instance_id, workflow_instance)
        
        logger.info(f"Triggered workflow {workflow_id} with instance {instance_id}")
        return instance_id
    
    async def _execute_workflow(self, instance_id: str, workflow_instance: Dict[str, Any]):
        """
        Execute a workflow instance
        """
        workflow = workflow_instance["workflow"]
        trigger_data = workflow_instance["trigger_data"]
        
        # Find steps that can be executed (no dependencies or dependencies met)
        executable_steps = []
        
        for step in workflow_instance["pending_steps"]:
            if not step.dependencies or all(
                dep_step_id in [s.step_id for s in workflow_instance["completed_steps"]]
                for dep_step_id in step.dependencies
            ):
                executable_steps.append(step)
        
        # Submit tasks for executable steps
        for step in executable_steps:
            # Replace template variables in payload
            payload = self._replace_template_variables(step.payload_template, trigger_data)
            
            # Check condition if specified
            if step.condition and not self._evaluate_condition(step.condition, trigger_data):
                continue
            
            # Submit task
            task_id = await self.submit_task(
                agent_type=step.agent_type,
                task_type=step.task_type,
                payload=payload,
                priority=TaskPriority.MEDIUM
            )
            
            # Track step execution
            step.task_id = task_id
            workflow_instance["pending_steps"].remove(step)
    
    async def _continue_workflow(
        self,
        instance_id: str,
        workflow_instance: Dict[str, Any],
        completed_task: AgentTask
    ):
        """
        Continue workflow execution after a task completion
        """
        # Find the workflow step that corresponds to this task
        completed_step = None
        for step in workflow_instance.get("executing_steps", []):
            if getattr(step, "task_id", None) == completed_task.task_id:
                completed_step = step
                break
        
        if not completed_step:
            return
        
        # Mark step as completed
        workflow_instance["completed_steps"].append(completed_step)
        if "executing_steps" in workflow_instance:
            workflow_instance["executing_steps"].remove(completed_step)
        
        # Check if workflow is complete
        if not workflow_instance["pending_steps"] and not workflow_instance.get("executing_steps"):
            workflow_instance["status"] = "completed"
            workflow_instance["completed_at"] = datetime.utcnow()
            logger.info(f"Workflow {instance_id} completed")
            return
        
        # Continue executing remaining steps
        await self._execute_workflow(instance_id, workflow_instance)
    
    def _replace_template_variables(self, template: Dict[str, Any], variables: Dict[str, Any]) -> Dict[str, Any]:
        """
        Replace template variables in payload
        """
        import json
        template_str = json.dumps(template)
        
        for key, value in variables.items():
            template_str = template_str.replace(f"{{{key}}}", str(value))
        
        return json.loads(template_str)
    
    def _evaluate_condition(self, condition: str, context: Dict[str, Any]) -> bool:
        """
        Evaluate a simple condition (secure evaluation)
        """
        try:
            # Simple condition evaluation for safety
            # In production, use a proper expression evaluator
            if "==" in condition:
                left, right = condition.split("==", 1)
                left = left.strip()
                right = right.strip().strip("'\"")
                return context.get(left) == right
            
            return True
        except Exception as e:
            logger.warning(f"Condition evaluation failed: {e}")
            return True
    
    async def share_data(self, key: str, data: Any, ttl_seconds: int = 3600):
        """
        Share data between agents
        """
        try:
            # Store in Redis with TTL
            serialized_data = json.dumps(data, default=str)
            redis_client.setex(f"agent_data:{key}", ttl_seconds, serialized_data)
            
            # Also store in local cache
            self.data_sharing_cache[key] = {
                "data": data,
                "expires_at": datetime.utcnow() + timedelta(seconds=ttl_seconds)
            }
            
            logger.debug(f"Shared data with key: {key}")
            
        except Exception as e:
            logger.error(f"Failed to share data: {e}")
    
    async def get_shared_data(self, key: str) -> Optional[Any]:
        """
        Retrieve shared data
        """
        try:
            # Check local cache first
            if key in self.data_sharing_cache:
                cache_entry = self.data_sharing_cache[key]
                if cache_entry["expires_at"] > datetime.utcnow():
                    return cache_entry["data"]
                else:
                    del self.data_sharing_cache[key]
            
            # Check Redis
            redis_data = redis_client.get(f"agent_data:{key}")
            if redis_data:
                return json.loads(redis_data)
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get shared data: {e}")
            return None
    
    async def get_system_status(self) -> Dict[str, Any]:
        """
        Get comprehensive system status
        """
        return {
            "agents": {
                "total": len(self.agents),
                "health_status": self.agent_health_status,
                "performance_metrics": self.performance_metrics
            },
            "tasks": {
                "active": len(self.active_tasks),
                "completed": len(self.completed_tasks),
                "queue_size": self.task_queue.qsize()
            },
            "workflows": {
                "available": len(self.workflows),
                "active_instances": len(self.active_workflow_instances),
                "workflow_list": list(self.workflows.keys())
            },
            "system": {
                "workers": len(self.task_workers),
                "uptime": datetime.utcnow().isoformat(),
                "data_cache_size": len(self.data_sharing_cache)
            }
        }
    
    async def get_analytics_dashboard(self) -> Dict[str, Any]:
        """
        Get analytics data for dashboard
        """
        # Calculate aggregate metrics
        total_tasks = len(self.completed_tasks)
        successful_tasks = len([t for t in self.completed_tasks.values() if t.status == TaskStatus.COMPLETED])
        
        # Agent performance summary
        agent_performance = {}
        for agent_type in AgentType:
            agent_metrics = [
                metrics for key, metrics in self.performance_metrics.items()
                if key.startswith(agent_type.value)
            ]
            
            if agent_metrics:
                total_executions = sum(m["total_executions"] for m in agent_metrics)
                successful_executions = sum(m["successful_executions"] for m in agent_metrics)
                avg_execution_time = sum(m["avg_time"] for m in agent_metrics) / len(agent_metrics)
                
                agent_performance[agent_type.value] = {
                    "total_executions": total_executions,
                    "success_rate": successful_executions / max(total_executions, 1),
                    "avg_execution_time": avg_execution_time,
                    "status": self.agent_health_status.get(agent_type.value, {}).get("status", "unknown")
                }
        
        # Workflow analytics
        workflow_analytics = {}
        for workflow_id, workflow in self.workflows.items():
            workflow_analytics[workflow_id] = {
                "enabled": workflow.enabled,
                "total_instances": len([
                    i for i in self.active_workflow_instances.values()
                    if i["workflow"].workflow_id == workflow_id
                ])
            }
        
        return {
            "system_overview": {
                "total_tasks": total_tasks,
                "success_rate": successful_tasks / max(total_tasks, 1),
                "active_tasks": len(self.active_tasks),
                "active_workflows": len(self.active_workflow_instances)
            },
            "agent_performance": agent_performance,
            "workflow_analytics": workflow_analytics,
            "performance_trends": self._calculate_performance_trends(),
            "generated_at": datetime.utcnow().isoformat()
        }
    
    def _calculate_performance_trends(self) -> Dict[str, Any]:
        """
        Calculate performance trends over time
        """
        # This would typically analyze historical data
        # For now, return current metrics as trends
        return {
            "task_completion_rate": {
                "current": len(self.completed_tasks) / max(len(self.completed_tasks) + len(self.active_tasks), 1),
                "trend": "stable"
            },
            "average_response_time": {
                "current": sum(m["avg_time"] for m in self.performance_metrics.values()) / max(len(self.performance_metrics), 1),
                "trend": "improving"
            },
            "system_load": {
                "current": len(self.active_tasks) / 10,  # Normalized to max concurrent tasks
                "trend": "stable"
            }
        }
    
    async def shutdown(self):
        """
        Gracefully shutdown the orchestrator
        """
        logger.info("Shutting down agent orchestrator")
        
        # Stop task workers
        for _ in self.task_workers:
            await self.task_queue.put(None)  # Shutdown signal
        
        # Wait for workers to finish
        await asyncio.gather(*self.task_workers, return_exceptions=True)
        
        # Cancel active workflows
        for instance_id in list(self.active_workflow_instances.keys()):
            self.active_workflow_instances[instance_id]["status"] = "cancelled"
        
        logger.info("Agent orchestrator shutdown complete")