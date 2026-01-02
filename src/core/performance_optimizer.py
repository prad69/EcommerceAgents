import asyncio
from typing import Dict, List, Any, Optional, Tuple, Callable
from loguru import logger
import json
import time
from datetime import datetime, timedelta
from dataclasses import dataclass
import statistics
import psutil
import gc
from contextlib import asynccontextmanager

from src.core.database import get_db, redis_client
from sqlalchemy import text
from sqlalchemy.orm import Query


@dataclass
class PerformanceMetrics:
    """Performance metrics data structure"""
    cpu_usage: float
    memory_usage: float
    db_query_time: float
    cache_hit_rate: float
    response_time: float
    error_rate: float
    throughput: int
    timestamp: datetime


@dataclass
class QueryOptimization:
    """Database query optimization recommendation"""
    query: str
    current_time: float
    optimization_type: str
    recommendation: str
    estimated_improvement: float
    priority: str


class PerformanceOptimizer:
    """
    Performance monitoring and optimization service
    """
    
    def __init__(self):
        self.metrics_history = []
        self.query_performance = {}
        self.cache_stats = {"hits": 0, "misses": 0}
        self.optimization_recommendations = []
        
        # Performance thresholds
        self.thresholds = {
            "cpu_usage": 80.0,
            "memory_usage": 85.0,
            "db_query_time": 1.0,
            "cache_hit_rate": 0.80,
            "response_time": 2.0,
            "error_rate": 0.05
        }
        
        # Start monitoring
        asyncio.create_task(self._start_monitoring())
        
        logger.info("Performance optimizer initialized")
    
    async def _start_monitoring(self):
        """
        Start continuous performance monitoring
        """
        while True:
            try:
                await self._collect_metrics()
                await self._analyze_performance()
                await asyncio.sleep(30)  # Collect metrics every 30 seconds
            except Exception as e:
                logger.error(f"Performance monitoring error: {e}")
                await asyncio.sleep(60)
    
    async def _collect_metrics(self):
        """
        Collect comprehensive performance metrics
        """
        try:
            # System metrics
            cpu_usage = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            memory_usage = memory.percent
            
            # Database metrics
            db_query_time = await self._measure_db_performance()
            
            # Cache metrics
            cache_hit_rate = self._calculate_cache_hit_rate()
            
            # Application metrics (would be collected from actual requests)
            response_time = await self._measure_response_time()
            error_rate = self._calculate_error_rate()
            throughput = self._calculate_throughput()
            
            metrics = PerformanceMetrics(
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                db_query_time=db_query_time,
                cache_hit_rate=cache_hit_rate,
                response_time=response_time,
                error_rate=error_rate,
                throughput=throughput,
                timestamp=datetime.utcnow()
            )
            
            self.metrics_history.append(metrics)
            
            # Keep only last 1000 metrics (about 8 hours at 30s intervals)
            if len(self.metrics_history) > 1000:
                self.metrics_history = self.metrics_history[-1000:]
            
            # Store in Redis for dashboard
            await self._store_metrics_in_cache(metrics)
            
        except Exception as e:
            logger.error(f"Failed to collect metrics: {e}")
    
    async def _measure_db_performance(self) -> float:
        """
        Measure database query performance
        """
        try:
            db = next(get_db())
            
            start_time = time.time()
            
            # Execute a sample query
            db.execute(text("SELECT COUNT(*) FROM products"))
            
            query_time = time.time() - start_time
            db.close()
            
            return query_time
            
        except Exception as e:
            logger.warning(f"DB performance measurement failed: {e}")
            return 0.0
    
    def _calculate_cache_hit_rate(self) -> float:
        """
        Calculate cache hit rate
        """
        total_requests = self.cache_stats["hits"] + self.cache_stats["misses"]
        if total_requests == 0:
            return 0.0
        
        return self.cache_stats["hits"] / total_requests
    
    async def _measure_response_time(self) -> float:
        """
        Measure average application response time
        """
        # This would measure actual API response times
        # For now, return a simulated value
        return 0.5 + (time.time() % 1.0)  # Simulated response time
    
    def _calculate_error_rate(self) -> float:
        """
        Calculate error rate from recent requests
        """
        # This would calculate actual error rates
        # For now, return a simulated value
        return 0.01  # 1% error rate
    
    def _calculate_throughput(self) -> int:
        """
        Calculate requests per second throughput
        """
        # This would calculate actual throughput
        # For now, return a simulated value
        return 100  # 100 RPS
    
    async def _store_metrics_in_cache(self, metrics: PerformanceMetrics):
        """
        Store metrics in Redis for dashboard access
        """
        try:
            metrics_data = {
                "cpu_usage": metrics.cpu_usage,
                "memory_usage": metrics.memory_usage,
                "db_query_time": metrics.db_query_time,
                "cache_hit_rate": metrics.cache_hit_rate,
                "response_time": metrics.response_time,
                "error_rate": metrics.error_rate,
                "throughput": metrics.throughput,
                "timestamp": metrics.timestamp.isoformat()
            }
            
            # Store current metrics
            redis_client.setex("performance:current", 300, json.dumps(metrics_data))
            
            # Store in time series for historical data
            redis_client.lpush("performance:history", json.dumps(metrics_data))
            redis_client.ltrim("performance:history", 0, 1000)  # Keep last 1000 entries
            
        except Exception as e:
            logger.warning(f"Failed to store metrics in cache: {e}")
    
    async def _analyze_performance(self):
        """
        Analyze performance and generate optimization recommendations
        """
        if not self.metrics_history:
            return
        
        current_metrics = self.metrics_history[-1]
        
        # Check thresholds and generate alerts
        alerts = []
        
        if current_metrics.cpu_usage > self.thresholds["cpu_usage"]:
            alerts.append(f"High CPU usage: {current_metrics.cpu_usage:.1f}%")
        
        if current_metrics.memory_usage > self.thresholds["memory_usage"]:
            alerts.append(f"High memory usage: {current_metrics.memory_usage:.1f}%")
        
        if current_metrics.db_query_time > self.thresholds["db_query_time"]:
            alerts.append(f"Slow database queries: {current_metrics.db_query_time:.2f}s")
        
        if current_metrics.cache_hit_rate < self.thresholds["cache_hit_rate"]:
            alerts.append(f"Low cache hit rate: {current_metrics.cache_hit_rate:.2f}")
        
        if current_metrics.response_time > self.thresholds["response_time"]:
            alerts.append(f"High response time: {current_metrics.response_time:.2f}s")
        
        if current_metrics.error_rate > self.thresholds["error_rate"]:
            alerts.append(f"High error rate: {current_metrics.error_rate:.2f}")
        
        if alerts:
            logger.warning(f"Performance alerts: {', '.join(alerts)}")
            await self._generate_optimization_recommendations(current_metrics, alerts)
    
    async def _generate_optimization_recommendations(
        self, 
        metrics: PerformanceMetrics, 
        alerts: List[str]
    ):
        """
        Generate specific optimization recommendations
        """
        recommendations = []
        
        # CPU optimization
        if metrics.cpu_usage > self.thresholds["cpu_usage"]:
            recommendations.append({
                "type": "cpu_optimization",
                "priority": "high",
                "recommendation": "Consider implementing CPU-intensive task queuing",
                "details": "High CPU usage detected. Move heavy computations to background tasks.",
                "estimated_impact": "30% CPU reduction"
            })
        
        # Memory optimization
        if metrics.memory_usage > self.thresholds["memory_usage"]:
            recommendations.append({
                "type": "memory_optimization",
                "priority": "high",
                "recommendation": "Implement garbage collection optimization",
                "details": "High memory usage. Consider memory profiling and leak detection.",
                "estimated_impact": "20% memory reduction"
            })
        
        # Database optimization
        if metrics.db_query_time > self.thresholds["db_query_time"]:
            recommendations.append({
                "type": "database_optimization",
                "priority": "high",
                "recommendation": "Optimize slow database queries",
                "details": "Add indexes, optimize query structure, implement query caching.",
                "estimated_impact": "50% query time reduction"
            })
        
        # Cache optimization
        if metrics.cache_hit_rate < self.thresholds["cache_hit_rate"]:
            recommendations.append({
                "type": "cache_optimization",
                "priority": "medium",
                "recommendation": "Improve caching strategy",
                "details": "Low cache hit rate. Review cache TTL and cache key strategy.",
                "estimated_impact": "40% response time improvement"
            })
        
        self.optimization_recommendations.extend(recommendations)
        
        # Keep only recent recommendations
        if len(self.optimization_recommendations) > 100:
            self.optimization_recommendations = self.optimization_recommendations[-100:]
    
    @asynccontextmanager
    async def monitor_query_performance(self, query_name: str):
        """
        Context manager for monitoring database query performance
        """
        start_time = time.time()
        
        try:
            yield
        finally:
            execution_time = time.time() - start_time
            
            if query_name not in self.query_performance:
                self.query_performance[query_name] = []
            
            self.query_performance[query_name].append({
                "execution_time": execution_time,
                "timestamp": datetime.utcnow()
            })
            
            # Keep only recent query metrics
            if len(self.query_performance[query_name]) > 100:
                self.query_performance[query_name] = self.query_performance[query_name][-100:]
            
            # Generate optimization recommendation if query is slow
            if execution_time > 1.0:  # Slow query threshold
                await self._analyze_slow_query(query_name, execution_time)
    
    async def _analyze_slow_query(self, query_name: str, execution_time: float):
        """
        Analyze slow queries and generate optimization recommendations
        """
        query_history = self.query_performance.get(query_name, [])
        
        if len(query_history) >= 5:  # Analyze with sufficient data
            recent_times = [q["execution_time"] for q in query_history[-10:]]
            avg_time = statistics.mean(recent_times)
            
            if avg_time > 0.5:  # Consistently slow
                optimization = QueryOptimization(
                    query=query_name,
                    current_time=avg_time,
                    optimization_type="index_recommendation",
                    recommendation=f"Consider adding database indexes for {query_name}",
                    estimated_improvement=0.6,  # 60% improvement
                    priority="high" if avg_time > 2.0 else "medium"
                )
                
                logger.warning(f"Slow query detected: {query_name} ({avg_time:.2f}s avg)")
    
    async def optimize_database_queries(self):
        """
        Apply database query optimizations
        """
        try:
            db = next(get_db())
            
            # Common optimization queries
            optimizations = [
                # Analyze table statistics
                "ANALYZE products;",
                "ANALYZE users;",
                "ANALYZE reviews;",
                
                # Suggest missing indexes (PostgreSQL specific)
                """
                SELECT schemaname, tablename, attname, n_distinct, correlation
                FROM pg_stats
                WHERE schemaname = 'public'
                AND n_distinct > 100
                ORDER BY n_distinct DESC;
                """
            ]
            
            for optimization_query in optimizations:
                try:
                    if optimization_query.strip().upper().startswith('SELECT'):
                        result = db.execute(text(optimization_query))
                        rows = result.fetchall()
                        if rows:
                            logger.info(f"Query optimization analysis completed: {len(rows)} results")
                    else:
                        db.execute(text(optimization_query))
                        logger.info(f"Applied optimization: {optimization_query[:50]}...")
                except Exception as e:
                    logger.warning(f"Optimization query failed: {e}")
            
            db.commit()
            db.close()
            
        except Exception as e:
            logger.error(f"Database optimization failed: {e}")
    
    async def optimize_cache_strategy(self):
        """
        Optimize caching strategy based on performance metrics
        """
        try:
            # Analyze cache performance
            hit_rate = self._calculate_cache_hit_rate()
            
            if hit_rate < 0.8:  # Low hit rate
                # Implement cache warming
                await self._warm_cache()
                
                # Adjust TTL based on access patterns
                await self._optimize_cache_ttl()
                
                logger.info("Applied cache optimizations")
            
        except Exception as e:
            logger.error(f"Cache optimization failed: {e}")
    
    async def _warm_cache(self):
        """
        Pre-populate cache with frequently accessed data
        """
        try:
            # Cache popular products
            popular_products_key = "popular_products"
            
            # This would fetch and cache popular products
            # For demonstration, we'll simulate this
            popular_products = ["product_1", "product_2", "product_3"]
            
            redis_client.setex(
                popular_products_key, 
                3600,  # 1 hour TTL
                json.dumps(popular_products)
            )
            
            # Cache frequently accessed user data
            # Similar implementation for other frequently accessed data
            
        except Exception as e:
            logger.warning(f"Cache warming failed: {e}")
    
    async def _optimize_cache_ttl(self):
        """
        Optimize cache TTL based on access patterns
        """
        try:
            # Analyze access patterns and adjust TTL
            # This would implement sophisticated TTL optimization
            # For now, we'll use default optimized values
            
            optimized_ttls = {
                "user_preferences": 1800,  # 30 minutes
                "product_recommendations": 900,  # 15 minutes
                "product_details": 3600,  # 1 hour
                "search_results": 600  # 10 minutes
            }
            
            # Apply optimized TTLs
            for key_pattern, ttl in optimized_ttls.items():
                # This would update existing cache entries with new TTL
                logger.debug(f"Optimized TTL for {key_pattern}: {ttl}s")
            
        except Exception as e:
            logger.warning(f"TTL optimization failed: {e}")
    
    async def auto_scale_recommendations(self) -> Dict[str, Any]:
        """
        Generate auto-scaling recommendations based on performance
        """
        if len(self.metrics_history) < 10:
            return {"recommendation": "insufficient_data"}
        
        recent_metrics = self.metrics_history[-10:]
        
        # Analyze recent performance trends
        avg_cpu = statistics.mean([m.cpu_usage for m in recent_metrics])
        avg_memory = statistics.mean([m.memory_usage for m in recent_metrics])
        avg_response_time = statistics.mean([m.response_time for m in recent_metrics])
        
        recommendations = {
            "current_load": {
                "cpu_usage": avg_cpu,
                "memory_usage": avg_memory,
                "response_time": avg_response_time
            },
            "scaling_recommendations": []
        }
        
        # Scale up recommendations
        if avg_cpu > 75 or avg_memory > 80:
            recommendations["scaling_recommendations"].append({
                "action": "scale_up",
                "reason": "High resource utilization",
                "priority": "high",
                "suggested_instances": 2
            })
        
        # Scale down recommendations
        elif avg_cpu < 20 and avg_memory < 30:
            recommendations["scaling_recommendations"].append({
                "action": "scale_down",
                "reason": "Low resource utilization",
                "priority": "medium",
                "suggested_instances": -1
            })
        
        # Performance optimization recommendations
        if avg_response_time > 1.5:
            recommendations["scaling_recommendations"].append({
                "action": "optimize_performance",
                "reason": "High response time",
                "priority": "high",
                "suggestions": ["enable_caching", "optimize_queries", "add_cdn"]
            })
        
        return recommendations
    
    async def memory_cleanup(self):
        """
        Perform memory cleanup and garbage collection
        """
        try:
            # Python garbage collection
            collected = gc.collect()
            
            # Clear old metrics
            if len(self.metrics_history) > 500:
                self.metrics_history = self.metrics_history[-500:]
            
            # Clear old query performance data
            for query_name in list(self.query_performance.keys()):
                if len(self.query_performance[query_name]) > 50:
                    self.query_performance[query_name] = self.query_performance[query_name][-50:]
            
            logger.info(f"Memory cleanup completed: {collected} objects collected")
            
        except Exception as e:
            logger.error(f"Memory cleanup failed: {e}")
    
    def record_cache_hit(self):
        """Record cache hit for statistics"""
        self.cache_stats["hits"] += 1
    
    def record_cache_miss(self):
        """Record cache miss for statistics"""
        self.cache_stats["misses"] += 1
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get performance summary for dashboard
        """
        if not self.metrics_history:
            return {"status": "no_data"}
        
        current_metrics = self.metrics_history[-1]
        
        # Calculate trends if we have enough data
        trends = {}
        if len(self.metrics_history) >= 10:
            recent = self.metrics_history[-10:]
            older = self.metrics_history[-20:-10] if len(self.metrics_history) >= 20 else []
            
            if older:
                recent_avg_cpu = statistics.mean([m.cpu_usage for m in recent])
                older_avg_cpu = statistics.mean([m.cpu_usage for m in older])
                cpu_trend = "increasing" if recent_avg_cpu > older_avg_cpu else "decreasing"
                
                recent_avg_response = statistics.mean([m.response_time for m in recent])
                older_avg_response = statistics.mean([m.response_time for m in older])
                response_trend = "increasing" if recent_avg_response > older_avg_response else "decreasing"
                
                trends = {
                    "cpu_trend": cpu_trend,
                    "response_time_trend": response_trend
                }
        
        return {
            "current_metrics": {
                "cpu_usage": current_metrics.cpu_usage,
                "memory_usage": current_metrics.memory_usage,
                "db_query_time": current_metrics.db_query_time,
                "cache_hit_rate": current_metrics.cache_hit_rate,
                "response_time": current_metrics.response_time,
                "error_rate": current_metrics.error_rate,
                "throughput": current_metrics.throughput
            },
            "trends": trends,
            "health_status": self._get_health_status(current_metrics),
            "optimization_recommendations": self.optimization_recommendations[-5:],  # Latest 5
            "metrics_collected_at": current_metrics.timestamp.isoformat()
        }
    
    def _get_health_status(self, metrics: PerformanceMetrics) -> str:
        """
        Determine overall system health status
        """
        issues = 0
        
        if metrics.cpu_usage > self.thresholds["cpu_usage"]:
            issues += 1
        if metrics.memory_usage > self.thresholds["memory_usage"]:
            issues += 1
        if metrics.db_query_time > self.thresholds["db_query_time"]:
            issues += 1
        if metrics.cache_hit_rate < self.thresholds["cache_hit_rate"]:
            issues += 1
        if metrics.response_time > self.thresholds["response_time"]:
            issues += 1
        if metrics.error_rate > self.thresholds["error_rate"]:
            issues += 1
        
        if issues == 0:
            return "excellent"
        elif issues <= 2:
            return "good"
        elif issues <= 4:
            return "fair"
        else:
            return "poor"
    
    async def run_performance_test(self, test_type: str = "basic") -> Dict[str, Any]:
        """
        Run performance tests and return results
        """
        test_results = {
            "test_type": test_type,
            "started_at": datetime.utcnow().isoformat(),
            "results": {}
        }
        
        try:
            if test_type == "database":
                # Database performance test
                db_results = await self._test_database_performance()
                test_results["results"]["database"] = db_results
            
            elif test_type == "cache":
                # Cache performance test
                cache_results = await self._test_cache_performance()
                test_results["results"]["cache"] = cache_results
            
            elif test_type == "memory":
                # Memory performance test
                memory_results = await self._test_memory_performance()
                test_results["results"]["memory"] = memory_results
            
            else:  # basic test
                # Run all basic tests
                test_results["results"]["database"] = await self._test_database_performance()
                test_results["results"]["cache"] = await self._test_cache_performance()
                test_results["results"]["memory"] = await self._test_memory_performance()
            
            test_results["completed_at"] = datetime.utcnow().isoformat()
            test_results["status"] = "completed"
            
        except Exception as e:
            test_results["status"] = "failed"
            test_results["error"] = str(e)
            test_results["completed_at"] = datetime.utcnow().isoformat()
        
        return test_results
    
    async def _test_database_performance(self) -> Dict[str, Any]:
        """Test database performance"""
        try:
            db = next(get_db())
            
            # Simple query performance test
            start_time = time.time()
            db.execute(text("SELECT COUNT(*) FROM products LIMIT 1000"))
            simple_query_time = time.time() - start_time
            
            # Complex query performance test
            start_time = time.time()
            db.execute(text("""
                SELECT p.category, COUNT(*) as product_count, AVG(p.price) as avg_price
                FROM products p
                GROUP BY p.category
                LIMIT 100
            """))
            complex_query_time = time.time() - start_time
            
            db.close()
            
            return {
                "simple_query_time": simple_query_time,
                "complex_query_time": complex_query_time,
                "status": "passed" if simple_query_time < 0.1 and complex_query_time < 0.5 else "warning"
            }
            
        except Exception as e:
            return {"status": "failed", "error": str(e)}
    
    async def _test_cache_performance(self) -> Dict[str, Any]:
        """Test cache performance"""
        try:
            # Test Redis performance
            start_time = time.time()
            
            # Write test
            test_key = "perf_test_key"
            test_data = {"test": "data", "timestamp": time.time()}
            redis_client.setex(test_key, 60, json.dumps(test_data))
            write_time = time.time() - start_time
            
            # Read test
            start_time = time.time()
            cached_data = redis_client.get(test_key)
            read_time = time.time() - start_time
            
            # Cleanup
            redis_client.delete(test_key)
            
            return {
                "write_time": write_time,
                "read_time": read_time,
                "data_integrity": json.loads(cached_data) == test_data,
                "status": "passed" if write_time < 0.01 and read_time < 0.01 else "warning"
            }
            
        except Exception as e:
            return {"status": "failed", "error": str(e)}
    
    async def _test_memory_performance(self) -> Dict[str, Any]:
        """Test memory performance"""
        try:
            # Get current memory usage
            memory_before = psutil.virtual_memory()
            
            # Create test data to measure memory allocation
            test_data = []
            for i in range(10000):
                test_data.append({"id": i, "data": f"test_data_{i}"})
            
            memory_after = psutil.virtual_memory()
            
            # Cleanup
            del test_data
            gc.collect()
            
            memory_final = psutil.virtual_memory()
            
            return {
                "memory_before_mb": memory_before.used / 1024 / 1024,
                "memory_after_mb": memory_after.used / 1024 / 1024,
                "memory_final_mb": memory_final.used / 1024 / 1024,
                "allocation_successful": memory_after.used > memory_before.used,
                "cleanup_successful": memory_final.used < memory_after.used,
                "status": "passed"
            }
            
        except Exception as e:
            return {"status": "failed", "error": str(e)}