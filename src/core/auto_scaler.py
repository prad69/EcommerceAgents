import asyncio
from typing import Dict, List, Any, Optional, Tuple, Callable
from loguru import logger
import json
import time
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import statistics
import psutil
import aiohttp

from src.core.database import get_db, redis_client
from src.core.performance_optimizer import PerformanceOptimizer


class ScalingDirection(Enum):
    UP = "up"
    DOWN = "down"
    MAINTAIN = "maintain"


class ResourceType(Enum):
    CPU = "cpu"
    MEMORY = "memory"
    STORAGE = "storage"
    NETWORK = "network"
    DATABASE = "database"
    CACHE = "cache"


class ScalingTrigger(Enum):
    THRESHOLD_BASED = "threshold"
    PREDICTIVE = "predictive"
    MANUAL = "manual"
    SCHEDULED = "scheduled"


@dataclass
class ScalingRule:
    """Auto-scaling rule configuration"""
    rule_id: str
    name: str
    resource_type: ResourceType
    metric: str
    threshold_up: float
    threshold_down: float
    scale_up_cooldown: int = 300  # 5 minutes
    scale_down_cooldown: int = 600  # 10 minutes
    min_instances: int = 1
    max_instances: int = 10
    scale_factor: float = 1.5
    enabled: bool = True
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()


@dataclass
class ScalingEvent:
    """Auto-scaling event record"""
    event_id: str
    rule_id: str
    trigger: ScalingTrigger
    direction: ScalingDirection
    resource_type: ResourceType
    from_instances: int
    to_instances: int
    metric_value: float
    threshold: float
    timestamp: datetime
    success: bool = False
    error_message: Optional[str] = None


@dataclass
class ResourceMetrics:
    """Current resource utilization metrics"""
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    network_io_mbps: float
    active_connections: int
    response_time_ms: float
    error_rate: float
    throughput_rps: int
    timestamp: datetime


class AutoScaler:
    """
    Intelligent auto-scaling system for e-commerce infrastructure
    """
    
    def __init__(self, performance_optimizer: PerformanceOptimizer = None):
        self.performance_optimizer = performance_optimizer or PerformanceOptimizer()
        
        # Scaling configuration
        self.scaling_rules = {}
        self.scaling_history = []
        self.current_instances = {}
        self.last_scaling_action = {}
        
        # Metrics collection
        self.metrics_history = []
        self.prediction_models = {}
        
        # External service integration
        self.cloud_provider = None  # Would integrate with AWS/GCP/Azure
        self.kubernetes_client = None  # For container orchestration
        
        logger.info("Auto-scaler initialized")
        
        # Setup default scaling rules
        self._setup_default_rules()
        
        # Start monitoring
        asyncio.create_task(self._start_monitoring())
    
    def _setup_default_rules(self):
        """
        Setup default auto-scaling rules
        """
        # CPU-based scaling
        cpu_rule = ScalingRule(
            rule_id="cpu_scaling",
            name="CPU-based Auto Scaling",
            resource_type=ResourceType.CPU,
            metric="cpu_percent",
            threshold_up=75.0,
            threshold_down=30.0,
            scale_up_cooldown=300,
            scale_down_cooldown=600,
            min_instances=2,
            max_instances=20,
            scale_factor=1.5
        )
        
        # Memory-based scaling
        memory_rule = ScalingRule(
            rule_id="memory_scaling",
            name="Memory-based Auto Scaling",
            resource_type=ResourceType.MEMORY,
            metric="memory_percent",
            threshold_up=80.0,
            threshold_down=40.0,
            scale_up_cooldown=300,
            scale_down_cooldown=600,
            min_instances=2,
            max_instances=15,
            scale_factor=1.3
        )
        
        # Response time-based scaling
        response_time_rule = ScalingRule(
            rule_id="response_time_scaling",
            name="Response Time Auto Scaling",
            resource_type=ResourceType.CPU,
            metric="response_time_ms",
            threshold_up=2000.0,  # 2 seconds
            threshold_down=500.0,  # 0.5 seconds
            scale_up_cooldown=180,  # Faster response to latency
            scale_down_cooldown=900,
            min_instances=3,
            max_instances=25,
            scale_factor=2.0
        )
        
        # Database scaling
        db_rule = ScalingRule(
            rule_id="database_scaling",
            name="Database Connection Scaling",
            resource_type=ResourceType.DATABASE,
            metric="active_connections",
            threshold_up=80.0,  # 80% of max connections
            threshold_down=20.0,  # 20% of max connections
            scale_up_cooldown=600,  # Databases scale slower
            scale_down_cooldown=1800,  # 30 minutes
            min_instances=1,
            max_instances=5,
            scale_factor=1.2
        )
        
        self.scaling_rules = {
            "cpu_scaling": cpu_rule,
            "memory_scaling": memory_rule,
            "response_time_scaling": response_time_rule,
            "database_scaling": db_rule
        }
        
        # Initialize current instances
        for rule_id, rule in self.scaling_rules.items():
            self.current_instances[rule_id] = rule.min_instances
            self.last_scaling_action[rule_id] = datetime.utcnow() - timedelta(hours=1)
    
    async def _start_monitoring(self):
        """
        Start continuous monitoring and auto-scaling
        """
        while True:
            try:
                # Collect current metrics
                metrics = await self._collect_resource_metrics()
                self.metrics_history.append(metrics)
                
                # Keep only recent metrics (last 24 hours)
                cutoff_time = datetime.utcnow() - timedelta(hours=24)
                self.metrics_history = [
                    m for m in self.metrics_history 
                    if m.timestamp > cutoff_time
                ]
                
                # Evaluate scaling rules
                await self._evaluate_scaling_rules(metrics)
                
                # Update predictions
                await self._update_predictions(metrics)
                
                # Store metrics for analysis
                await self._store_metrics(metrics)
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Auto-scaling monitoring error: {e}")
                await asyncio.sleep(60)
    
    async def _collect_resource_metrics(self) -> ResourceMetrics:
        """
        Collect current resource utilization metrics
        """
        try:
            # System metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            network = psutil.net_io_counters()
            
            # Network throughput (simplified calculation)
            network_io_mbps = (network.bytes_sent + network.bytes_recv) / 1024 / 1024
            
            # Application metrics (would integrate with actual monitoring)
            active_connections = 50  # Placeholder - would get from connection pool
            response_time_ms = 750  # Placeholder - would get from APM
            error_rate = 0.02  # Placeholder - 2% error rate
            throughput_rps = 150  # Placeholder - requests per second
            
            return ResourceMetrics(
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                disk_percent=disk.percent,
                network_io_mbps=network_io_mbps,
                active_connections=active_connections,
                response_time_ms=response_time_ms,
                error_rate=error_rate,
                throughput_rps=throughput_rps,
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error(f"Failed to collect metrics: {e}")
            # Return default metrics
            return ResourceMetrics(
                cpu_percent=0.0,
                memory_percent=0.0,
                disk_percent=0.0,
                network_io_mbps=0.0,
                active_connections=0,
                response_time_ms=1000.0,
                error_rate=0.0,
                throughput_rps=0,
                timestamp=datetime.utcnow()
            )
    
    async def _evaluate_scaling_rules(self, metrics: ResourceMetrics):
        """
        Evaluate all scaling rules against current metrics
        """
        for rule_id, rule in self.scaling_rules.items():
            if not rule.enabled:
                continue
            
            try:
                # Get metric value for this rule
                metric_value = getattr(metrics, rule.metric)
                
                # Check cooldown period
                if not self._is_cooldown_elapsed(rule_id, rule):
                    continue
                
                # Determine scaling direction
                scaling_direction = self._determine_scaling_direction(rule, metric_value)
                
                if scaling_direction == ScalingDirection.MAINTAIN:
                    continue
                
                # Calculate new instance count
                current_count = self.current_instances[rule_id]
                new_count = self._calculate_new_instance_count(
                    rule, current_count, scaling_direction
                )
                
                if new_count != current_count:
                    await self._execute_scaling_action(
                        rule, scaling_direction, current_count, new_count, 
                        metric_value, ScalingTrigger.THRESHOLD_BASED
                    )
            
            except Exception as e:
                logger.error(f"Failed to evaluate scaling rule {rule_id}: {e}")
    
    def _is_cooldown_elapsed(self, rule_id: str, rule: ScalingRule) -> bool:
        """
        Check if cooldown period has elapsed since last scaling action
        """
        last_action = self.last_scaling_action.get(rule_id)
        if not last_action:
            return True
        
        # Use different cooldown for scale up vs scale down
        cooldown_duration = timedelta(seconds=rule.scale_up_cooldown)
        
        return datetime.utcnow() - last_action >= cooldown_duration
    
    def _determine_scaling_direction(self, rule: ScalingRule, metric_value: float) -> ScalingDirection:
        """
        Determine if scaling up, down, or maintaining current scale
        """
        if metric_value >= rule.threshold_up:
            return ScalingDirection.UP
        elif metric_value <= rule.threshold_down:
            return ScalingDirection.DOWN
        else:
            return ScalingDirection.MAINTAIN
    
    def _calculate_new_instance_count(
        self, 
        rule: ScalingRule, 
        current_count: int, 
        direction: ScalingDirection
    ) -> int:
        """
        Calculate new instance count based on scaling rule
        """
        if direction == ScalingDirection.UP:
            new_count = max(1, int(current_count * rule.scale_factor))
            new_count = min(new_count, rule.max_instances)
        elif direction == ScalingDirection.DOWN:
            new_count = max(1, int(current_count / rule.scale_factor))
            new_count = max(new_count, rule.min_instances)
        else:
            new_count = current_count
        
        return new_count
    
    async def _execute_scaling_action(
        self,
        rule: ScalingRule,
        direction: ScalingDirection,
        from_instances: int,
        to_instances: int,
        metric_value: float,
        trigger: ScalingTrigger
    ):
        """
        Execute the scaling action
        """
        event_id = f"scale_{rule.rule_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        scaling_event = ScalingEvent(
            event_id=event_id,
            rule_id=rule.rule_id,
            trigger=trigger,
            direction=direction,
            resource_type=rule.resource_type,
            from_instances=from_instances,
            to_instances=to_instances,
            metric_value=metric_value,
            threshold=rule.threshold_up if direction == ScalingDirection.UP else rule.threshold_down,
            timestamp=datetime.utcnow()
        )
        
        try:
            # Execute the scaling action
            success = await self._scale_resource(rule, to_instances)
            
            if success:
                # Update instance count
                self.current_instances[rule.rule_id] = to_instances
                self.last_scaling_action[rule.rule_id] = datetime.utcnow()
                
                scaling_event.success = True
                
                logger.info(
                    f"Scaled {rule.name} from {from_instances} to {to_instances} instances "
                    f"(metric: {metric_value}, direction: {direction.value})"
                )
                
                # Trigger post-scaling actions
                await self._post_scaling_actions(rule, scaling_event)
            else:
                scaling_event.success = False
                scaling_event.error_message = "Scaling operation failed"
                
        except Exception as e:
            scaling_event.success = False
            scaling_event.error_message = str(e)
            logger.error(f"Scaling action failed: {e}")
        
        # Record the scaling event
        self.scaling_history.append(scaling_event)
        await self._record_scaling_event(scaling_event)
    
    async def _scale_resource(self, rule: ScalingRule, target_instances: int) -> bool:
        """
        Scale the actual resource (integrate with cloud provider/k8s)
        """
        try:
            # This would integrate with actual infrastructure
            # For demonstration, we'll simulate the scaling
            
            if rule.resource_type == ResourceType.CPU:
                # Scale application servers
                success = await self._scale_application_servers(target_instances)
            elif rule.resource_type == ResourceType.DATABASE:
                # Scale database read replicas
                success = await self._scale_database_replicas(target_instances)
            elif rule.resource_type == ResourceType.CACHE:
                # Scale cache cluster
                success = await self._scale_cache_cluster(target_instances)
            else:
                # Generic resource scaling
                success = await self._scale_generic_resource(rule, target_instances)
            
            return success
            
        except Exception as e:
            logger.error(f"Resource scaling failed: {e}")
            return False
    
    async def _scale_application_servers(self, target_instances: int) -> bool:
        """
        Scale application server instances
        """
        # This would integrate with container orchestrator (K8s) or cloud auto-scaling
        logger.info(f"Scaling application servers to {target_instances} instances")
        
        # Simulate scaling delay
        await asyncio.sleep(1)
        
        # In real implementation:
        # - Update Kubernetes deployment replicas
        # - Trigger AWS Auto Scaling Group scaling
        # - Update load balancer configuration
        
        return True
    
    async def _scale_database_replicas(self, target_instances: int) -> bool:
        """
        Scale database read replicas
        """
        logger.info(f"Scaling database replicas to {target_instances} instances")
        
        # Simulate scaling delay
        await asyncio.sleep(2)
        
        # In real implementation:
        # - Add/remove read replicas
        # - Update connection pool configuration
        # - Update application configuration
        
        return True
    
    async def _scale_cache_cluster(self, target_instances: int) -> bool:
        """
        Scale cache cluster nodes
        """
        logger.info(f"Scaling cache cluster to {target_instances} nodes")
        
        # Simulate scaling delay
        await asyncio.sleep(1)
        
        # In real implementation:
        # - Add/remove Redis cluster nodes
        # - Update cache configuration
        # - Rebalance cache shards
        
        return True
    
    async def _scale_generic_resource(self, rule: ScalingRule, target_instances: int) -> bool:
        """
        Generic resource scaling implementation
        """
        logger.info(f"Scaling {rule.resource_type.value} to {target_instances} instances")
        
        # Simulate scaling delay
        await asyncio.sleep(1)
        
        return True
    
    async def _post_scaling_actions(self, rule: ScalingRule, event: ScalingEvent):
        """
        Execute post-scaling actions
        """
        try:
            # Update monitoring and alerting thresholds
            await self._update_monitoring_thresholds(rule, event.to_instances)
            
            # Warm up new instances if scaling up
            if event.direction == ScalingDirection.UP:
                await self._warmup_new_instances(rule, event.to_instances)
            
            # Update load balancer configuration
            await self._update_load_balancer(rule, event.to_instances)
            
            # Send notifications
            await self._send_scaling_notification(event)
            
        except Exception as e:
            logger.error(f"Post-scaling actions failed: {e}")
    
    async def _update_monitoring_thresholds(self, rule: ScalingRule, instance_count: int):
        """
        Update monitoring thresholds based on new instance count
        """
        # Adjust thresholds based on instance count
        # More instances might need higher absolute thresholds
        pass
    
    async def _warmup_new_instances(self, rule: ScalingRule, instance_count: int):
        """
        Warm up newly created instances
        """
        if rule.resource_type == ResourceType.CPU:
            # Warm up application servers
            # - Pre-load caches
            # - Initialize connections
            # - Run health checks
            pass
    
    async def _update_load_balancer(self, rule: ScalingRule, instance_count: int):
        """
        Update load balancer configuration for new instances
        """
        # Update load balancer target groups
        # Adjust health check intervals
        # Update traffic distribution
        pass
    
    async def _send_scaling_notification(self, event: ScalingEvent):
        """
        Send notifications about scaling events
        """
        try:
            notification = {
                "event_id": event.event_id,
                "rule_name": self.scaling_rules[event.rule_id].name,
                "direction": event.direction.value,
                "from_instances": event.from_instances,
                "to_instances": event.to_instances,
                "metric_value": event.metric_value,
                "timestamp": event.timestamp.isoformat(),
                "success": event.success
            }
            
            # Store in Redis for dashboard
            redis_client.lpush("scaling_notifications", json.dumps(notification))
            redis_client.ltrim("scaling_notifications", 0, 100)  # Keep last 100
            
        except Exception as e:
            logger.warning(f"Failed to send scaling notification: {e}")
    
    async def _update_predictions(self, metrics: ResourceMetrics):
        """
        Update predictive models for proactive scaling
        """
        try:
            # Simple trend analysis
            if len(self.metrics_history) >= 10:
                # Analyze CPU trend
                recent_cpu = [m.cpu_percent for m in self.metrics_history[-10:]]
                cpu_trend = self._calculate_trend(recent_cpu)
                
                # Analyze memory trend
                recent_memory = [m.memory_percent for m in self.metrics_history[-10:]]
                memory_trend = self._calculate_trend(recent_memory)
                
                # Analyze response time trend
                recent_response = [m.response_time_ms for m in self.metrics_history[-10:]]
                response_trend = self._calculate_trend(recent_response)
                
                # Store predictions
                predictions = {
                    "cpu_trend": cpu_trend,
                    "memory_trend": memory_trend,
                    "response_time_trend": response_trend,
                    "timestamp": datetime.utcnow().isoformat()
                }
                
                redis_client.setex("scaling_predictions", 300, json.dumps(predictions))
                
                # Trigger predictive scaling if trend is strong
                await self._evaluate_predictive_scaling(predictions)
                
        except Exception as e:
            logger.error(f"Failed to update predictions: {e}")
    
    def _calculate_trend(self, values: List[float]) -> Dict[str, Any]:
        """
        Calculate trend direction and strength
        """
        if len(values) < 3:
            return {"direction": "stable", "strength": 0.0, "slope": 0.0}
        
        # Simple linear regression
        x = list(range(len(values)))
        y = values
        
        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(x[i] * y[i] for i in range(n))
        sum_x2 = sum(x[i] ** 2 for i in range(n))
        
        # Calculate slope
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
        
        # Determine direction and strength
        if abs(slope) < 0.1:
            direction = "stable"
        elif slope > 0:
            direction = "increasing"
        else:
            direction = "decreasing"
        
        strength = min(abs(slope) * 10, 1.0)  # Normalize to 0-1
        
        return {
            "direction": direction,
            "strength": strength,
            "slope": slope
        }
    
    async def _evaluate_predictive_scaling(self, predictions: Dict[str, Any]):
        """
        Evaluate if predictive scaling should be triggered
        """
        try:
            # Check if any metric has strong increasing trend
            strong_increase_threshold = 0.7
            
            for metric, trend_data in predictions.items():
                if metric.endswith("_trend"):
                    if (trend_data["direction"] == "increasing" and 
                        trend_data["strength"] > strong_increase_threshold):
                        
                        # Find relevant scaling rule
                        for rule_id, rule in self.scaling_rules.items():
                            if metric.startswith(rule.metric.replace("_percent", "").replace("_ms", "")):
                                await self._trigger_predictive_scaling(rule, trend_data)
                                break
                                
        except Exception as e:
            logger.error(f"Predictive scaling evaluation failed: {e}")
    
    async def _trigger_predictive_scaling(self, rule: ScalingRule, trend_data: Dict[str, Any]):
        """
        Trigger predictive scaling action
        """
        current_count = self.current_instances[rule.rule_id]
        
        # Conservative predictive scaling (smaller scale factor)
        predictive_scale_factor = 1.2
        new_count = min(int(current_count * predictive_scale_factor), rule.max_instances)
        
        if new_count > current_count:
            logger.info(f"Triggering predictive scaling for {rule.name}: {current_count} -> {new_count}")
            
            await self._execute_scaling_action(
                rule=rule,
                direction=ScalingDirection.UP,
                from_instances=current_count,
                to_instances=new_count,
                metric_value=trend_data["slope"],
                trigger=ScalingTrigger.PREDICTIVE
            )
    
    async def _store_metrics(self, metrics: ResourceMetrics):
        """
        Store metrics for analysis and dashboard
        """
        try:
            metrics_data = asdict(metrics)
            metrics_data["timestamp"] = metrics.timestamp.isoformat()
            
            # Store in Redis
            redis_client.lpush("resource_metrics", json.dumps(metrics_data))
            redis_client.ltrim("resource_metrics", 0, 1000)  # Keep last 1000 entries
            
        except Exception as e:
            logger.warning(f"Failed to store metrics: {e}")
    
    async def _record_scaling_event(self, event: ScalingEvent):
        """
        Record scaling event for audit and analysis
        """
        try:
            event_data = asdict(event)
            event_data["timestamp"] = event.timestamp.isoformat()
            
            # Store in Redis
            redis_client.lpush("scaling_events", json.dumps(event_data))
            redis_client.ltrim("scaling_events", 0, 500)  # Keep last 500 events
            
        except Exception as e:
            logger.warning(f"Failed to record scaling event: {e}")
    
    # Public API methods
    
    async def add_scaling_rule(self, rule: ScalingRule):
        """
        Add a new scaling rule
        """
        self.scaling_rules[rule.rule_id] = rule
        self.current_instances[rule.rule_id] = rule.min_instances
        self.last_scaling_action[rule.rule_id] = datetime.utcnow() - timedelta(hours=1)
        
        logger.info(f"Added scaling rule: {rule.name}")
    
    async def remove_scaling_rule(self, rule_id: str):
        """
        Remove a scaling rule
        """
        if rule_id in self.scaling_rules:
            del self.scaling_rules[rule_id]
            del self.current_instances[rule_id]
            del self.last_scaling_action[rule_id]
            
            logger.info(f"Removed scaling rule: {rule_id}")
    
    async def enable_rule(self, rule_id: str):
        """
        Enable a scaling rule
        """
        if rule_id in self.scaling_rules:
            self.scaling_rules[rule_id].enabled = True
            logger.info(f"Enabled scaling rule: {rule_id}")
    
    async def disable_rule(self, rule_id: str):
        """
        Disable a scaling rule
        """
        if rule_id in self.scaling_rules:
            self.scaling_rules[rule_id].enabled = False
            logger.info(f"Disabled scaling rule: {rule_id}")
    
    async def manual_scale(
        self, 
        rule_id: str, 
        target_instances: int, 
        reason: str = "Manual scaling"
    ) -> bool:
        """
        Manually trigger scaling for a specific rule
        """
        if rule_id not in self.scaling_rules:
            logger.error(f"Scaling rule {rule_id} not found")
            return False
        
        rule = self.scaling_rules[rule_id]
        current_count = self.current_instances[rule_id]
        
        # Validate target instance count
        target_instances = max(rule.min_instances, min(target_instances, rule.max_instances))
        
        if target_instances == current_count:
            logger.info(f"Target instances ({target_instances}) same as current for {rule_id}")
            return True
        
        direction = ScalingDirection.UP if target_instances > current_count else ScalingDirection.DOWN
        
        await self._execute_scaling_action(
            rule=rule,
            direction=direction,
            from_instances=current_count,
            to_instances=target_instances,
            metric_value=0.0,  # Manual scaling
            trigger=ScalingTrigger.MANUAL
        )
        
        return True
    
    def get_scaling_status(self) -> Dict[str, Any]:
        """
        Get current scaling status and configuration
        """
        status = {
            "rules": {},
            "current_instances": self.current_instances.copy(),
            "recent_events": [],
            "predictions": {},
            "system_metrics": {}
        }
        
        # Rule information
        for rule_id, rule in self.scaling_rules.items():
            status["rules"][rule_id] = {
                "name": rule.name,
                "resource_type": rule.resource_type.value,
                "enabled": rule.enabled,
                "min_instances": rule.min_instances,
                "max_instances": rule.max_instances,
                "threshold_up": rule.threshold_up,
                "threshold_down": rule.threshold_down
            }
        
        # Recent scaling events
        status["recent_events"] = [
            {
                "event_id": event.event_id,
                "rule_id": event.rule_id,
                "direction": event.direction.value,
                "from_instances": event.from_instances,
                "to_instances": event.to_instances,
                "timestamp": event.timestamp.isoformat(),
                "success": event.success
            }
            for event in self.scaling_history[-10:]  # Last 10 events
        ]
        
        # Current metrics
        if self.metrics_history:
            latest_metrics = self.metrics_history[-1]
            status["system_metrics"] = {
                "cpu_percent": latest_metrics.cpu_percent,
                "memory_percent": latest_metrics.memory_percent,
                "response_time_ms": latest_metrics.response_time_ms,
                "throughput_rps": latest_metrics.throughput_rps,
                "timestamp": latest_metrics.timestamp.isoformat()
            }
        
        return status
    
    def get_scaling_recommendations(self) -> List[Dict[str, Any]]:
        """
        Get scaling recommendations based on historical data
        """
        recommendations = []
        
        if len(self.metrics_history) < 20:
            return recommendations
        
        try:
            # Analyze resource utilization patterns
            recent_metrics = self.metrics_history[-20:]
            
            # CPU utilization analysis
            cpu_values = [m.cpu_percent for m in recent_metrics]
            avg_cpu = statistics.mean(cpu_values)
            max_cpu = max(cpu_values)
            
            if avg_cpu > 60 and max_cpu > 85:
                recommendations.append({
                    "type": "scale_up",
                    "resource": "cpu",
                    "reason": f"High CPU utilization (avg: {avg_cpu:.1f}%, max: {max_cpu:.1f}%)",
                    "priority": "high"
                })
            
            # Memory utilization analysis
            memory_values = [m.memory_percent for m in recent_metrics]
            avg_memory = statistics.mean(memory_values)
            max_memory = max(memory_values)
            
            if avg_memory > 70 and max_memory > 90:
                recommendations.append({
                    "type": "scale_up",
                    "resource": "memory",
                    "reason": f"High memory utilization (avg: {avg_memory:.1f}%, max: {max_memory:.1f}%)",
                    "priority": "high"
                })
            
            # Response time analysis
            response_times = [m.response_time_ms for m in recent_metrics]
            avg_response = statistics.mean(response_times)
            max_response = max(response_times)
            
            if avg_response > 1500 and max_response > 3000:
                recommendations.append({
                    "type": "scale_up",
                    "resource": "cpu",
                    "reason": f"High response times (avg: {avg_response:.0f}ms, max: {max_response:.0f}ms)",
                    "priority": "medium"
                })
            
            # Check for over-provisioning
            if avg_cpu < 20 and avg_memory < 30 and avg_response < 500:
                recommendations.append({
                    "type": "scale_down",
                    "resource": "general",
                    "reason": f"Low resource utilization (CPU: {avg_cpu:.1f}%, Memory: {avg_memory:.1f}%)",
                    "priority": "low"
                })
            
        except Exception as e:
            logger.error(f"Failed to generate recommendations: {e}")
        
        return recommendations
    
    async def shutdown(self):
        """
        Gracefully shutdown the auto-scaler
        """
        logger.info("Shutting down auto-scaler")
        
        # Could implement graceful shutdown:
        # - Save scaling history to persistent storage
        # - Cancel any ongoing scaling operations
        # - Send final notifications
        
        logger.info("Auto-scaler shutdown complete")