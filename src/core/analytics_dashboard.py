import asyncio
from typing import Dict, List, Any, Optional, Tuple
from loguru import logger
import json
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import statistics
import pandas as pd
import numpy as np
from collections import defaultdict

from src.core.database import get_db, redis_client
from src.core.performance_optimizer import PerformanceOptimizer
from src.core.auto_scaler import AutoScaler
from sqlalchemy import text, func
from sqlalchemy.orm import Session


class MetricType(Enum):
    PERFORMANCE = "performance"
    BUSINESS = "business"
    OPERATIONAL = "operational"
    USER_BEHAVIOR = "user_behavior"


class TimeRange(Enum):
    HOUR = "1h"
    DAY = "1d"
    WEEK = "1w"
    MONTH = "1m"
    QUARTER = "3m"
    YEAR = "1y"


class AggregationType(Enum):
    SUM = "sum"
    AVERAGE = "avg"
    COUNT = "count"
    MIN = "min"
    MAX = "max"
    MEDIAN = "median"
    PERCENTILE = "percentile"


@dataclass
class DashboardWidget:
    """Dashboard widget configuration"""
    widget_id: str
    title: str
    widget_type: str  # chart, metric, table, heatmap
    data_source: str
    query: str
    refresh_interval: int = 60  # seconds
    visualization_config: Dict[str, Any] = None
    filters: Dict[str, Any] = None
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.visualization_config is None:
            self.visualization_config = {}
        if self.filters is None:
            self.filters = {}


@dataclass
class AnalyticsReport:
    """Analytics report data structure"""
    report_id: str
    title: str
    description: str
    time_range: TimeRange
    metrics: Dict[str, Any]
    charts: List[Dict[str, Any]]
    insights: List[str]
    recommendations: List[str]
    generated_at: datetime
    data_freshness: datetime


class AnalyticsDashboard:
    """
    Comprehensive analytics dashboard for e-commerce multi-agent system
    """
    
    def __init__(self, performance_optimizer: PerformanceOptimizer = None, auto_scaler: AutoScaler = None):
        self.performance_optimizer = performance_optimizer
        self.auto_scaler = auto_scaler
        
        # Dashboard configuration
        self.widgets = {}
        self.reports = {}
        self.cached_data = {}
        self.cache_ttl = {}
        
        # Analytics engines
        self.business_analytics = BusinessAnalyticsEngine()
        self.performance_analytics = PerformanceAnalyticsEngine()
        self.user_analytics = UserAnalyticsEngine()
        self.operational_analytics = OperationalAnalyticsEngine()
        
        logger.info("Analytics dashboard initialized")
        
        # Setup default widgets
        self._setup_default_widgets()
        
        # Start data refresh scheduler
        asyncio.create_task(self._start_data_refresh_scheduler())
    
    def _setup_default_widgets(self):
        """
        Setup default dashboard widgets
        """
        # System performance overview
        performance_overview = DashboardWidget(
            widget_id="performance_overview",
            title="System Performance Overview",
            widget_type="metric_grid",
            data_source="performance_metrics",
            query="SELECT cpu_usage, memory_usage, response_time, throughput FROM performance_metrics ORDER BY timestamp DESC LIMIT 1",
            refresh_interval=30,
            visualization_config={
                "metrics": [
                    {"name": "CPU Usage", "field": "cpu_usage", "unit": "%", "threshold": 80},
                    {"name": "Memory Usage", "field": "memory_usage", "unit": "%", "threshold": 85},
                    {"name": "Response Time", "field": "response_time", "unit": "ms", "threshold": 2000},
                    {"name": "Throughput", "field": "throughput", "unit": "RPS", "direction": "higher_better"}
                ]
            }
        )
        
        # Agent activity timeline
        agent_activity = DashboardWidget(
            widget_id="agent_activity",
            title="Agent Activity Timeline",
            widget_type="timeline_chart",
            data_source="agent_tasks",
            query="SELECT agent_type, task_type, status, created_at, completed_at FROM agent_tasks WHERE created_at >= NOW() - INTERVAL '24 HOURS'",
            refresh_interval=60,
            visualization_config={
                "x_axis": "created_at",
                "y_axis": "agent_type",
                "color_field": "status",
                "tooltip_fields": ["task_type", "status", "created_at"]
            }
        )
        
        # Business metrics
        business_metrics = DashboardWidget(
            widget_id="business_metrics",
            title="Business Metrics",
            widget_type="metric_cards",
            data_source="business_metrics",
            query="SELECT conversion_rate, revenue, user_engagement, recommendation_ctr FROM business_metrics ORDER BY date DESC LIMIT 7",
            refresh_interval=300,
            visualization_config={
                "cards": [
                    {"title": "Conversion Rate", "field": "conversion_rate", "format": "percentage"},
                    {"title": "Revenue", "field": "revenue", "format": "currency"},
                    {"title": "User Engagement", "field": "user_engagement", "format": "number"},
                    {"title": "Recommendation CTR", "field": "recommendation_ctr", "format": "percentage"}
                ]
            }
        )
        
        # Error rate trends
        error_trends = DashboardWidget(
            widget_id="error_trends",
            title="Error Rate Trends",
            widget_type="line_chart",
            data_source="error_logs",
            query="SELECT DATE_TRUNC('hour', timestamp) as hour, COUNT(*) as error_count, service FROM error_logs WHERE timestamp >= NOW() - INTERVAL '24 HOURS' GROUP BY hour, service ORDER BY hour",
            refresh_interval=120,
            visualization_config={
                "x_axis": "hour",
                "y_axis": "error_count",
                "group_by": "service",
                "chart_type": "line"
            }
        )
        
        # User behavior heatmap
        user_heatmap = DashboardWidget(
            widget_id="user_heatmap",
            title="User Activity Heatmap",
            widget_type="heatmap",
            data_source="user_activities",
            query="SELECT EXTRACT(hour from timestamp) as hour, EXTRACT(dow from timestamp) as day_of_week, COUNT(*) as activity_count FROM user_activities WHERE timestamp >= NOW() - INTERVAL '7 DAYS' GROUP BY hour, day_of_week",
            refresh_interval=600,
            visualization_config={
                "x_axis": "hour",
                "y_axis": "day_of_week",
                "value_field": "activity_count",
                "color_scale": "blues"
            }
        )
        
        # Recommendation performance
        recommendation_performance = DashboardWidget(
            widget_id="recommendation_performance",
            title="Recommendation Engine Performance",
            widget_type="gauge_chart",
            data_source="recommendation_metrics",
            query="SELECT accuracy, precision, recall, f1_score FROM recommendation_metrics ORDER BY timestamp DESC LIMIT 1",
            refresh_interval=300,
            visualization_config={
                "gauges": [
                    {"title": "Accuracy", "field": "accuracy", "min": 0, "max": 1, "threshold": 0.8},
                    {"title": "Precision", "field": "precision", "min": 0, "max": 1, "threshold": 0.75},
                    {"title": "Recall", "field": "recall", "min": 0, "max": 1, "threshold": 0.7},
                    {"title": "F1 Score", "field": "f1_score", "min": 0, "max": 1, "threshold": 0.75}
                ]
            }
        )
        
        self.widgets = {
            "performance_overview": performance_overview,
            "agent_activity": agent_activity,
            "business_metrics": business_metrics,
            "error_trends": error_trends,
            "user_heatmap": user_heatmap,
            "recommendation_performance": recommendation_performance
        }
    
    async def _start_data_refresh_scheduler(self):
        """
        Start scheduler for refreshing dashboard data
        """
        while True:
            try:
                current_time = datetime.utcnow()
                
                for widget_id, widget in self.widgets.items():
                    # Check if widget data needs refresh
                    if self._should_refresh_widget(widget_id, widget, current_time):
                        await self._refresh_widget_data(widget_id, widget)
                
                # Refresh analytics reports
                await self._refresh_analytics_reports()
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Dashboard refresh scheduler error: {e}")
                await asyncio.sleep(60)
    
    def _should_refresh_widget(self, widget_id: str, widget: DashboardWidget, current_time: datetime) -> bool:
        """
        Check if widget data should be refreshed
        """
        if widget_id not in self.cache_ttl:
            return True
        
        last_refresh = self.cache_ttl[widget_id]
        time_since_refresh = (current_time - last_refresh).total_seconds()
        
        return time_since_refresh >= widget.refresh_interval
    
    async def _refresh_widget_data(self, widget_id: str, widget: DashboardWidget):
        """
        Refresh data for a specific widget
        """
        try:
            logger.debug(f"Refreshing widget data: {widget_id}")
            
            # Execute widget query
            data = await self._execute_widget_query(widget)
            
            # Process and format data
            formatted_data = await self._format_widget_data(widget, data)
            
            # Cache the data
            self.cached_data[widget_id] = formatted_data
            self.cache_ttl[widget_id] = datetime.utcnow()
            
            # Store in Redis for real-time access
            redis_client.setex(
                f"widget_data:{widget_id}",
                widget.refresh_interval + 60,  # TTL with buffer
                json.dumps(formatted_data, default=str)
            )
            
        except Exception as e:
            logger.error(f"Failed to refresh widget {widget_id}: {e}")
    
    async def _execute_widget_query(self, widget: DashboardWidget) -> List[Dict[str, Any]]:
        """
        Execute query for widget data
        """
        try:
            if widget.data_source == "performance_metrics":
                return await self._get_performance_metrics_data(widget.query)
            elif widget.data_source == "agent_tasks":
                return await self._get_agent_tasks_data(widget.query)
            elif widget.data_source == "business_metrics":
                return await self._get_business_metrics_data(widget.query)
            elif widget.data_source == "error_logs":
                return await self._get_error_logs_data(widget.query)
            elif widget.data_source == "user_activities":
                return await self._get_user_activities_data(widget.query)
            elif widget.data_source == "recommendation_metrics":
                return await self._get_recommendation_metrics_data(widget.query)
            else:
                # Generic database query
                return await self._execute_database_query(widget.query)
                
        except Exception as e:
            logger.error(f"Widget query execution failed: {e}")
            return []
    
    async def _get_performance_metrics_data(self, query: str) -> List[Dict[str, Any]]:
        """
        Get performance metrics data from Redis/monitoring system
        """
        try:
            # Get latest performance metrics from Redis
            metrics_data = redis_client.get("performance:current")
            if metrics_data:
                metrics = json.loads(metrics_data)
                return [{
                    "cpu_usage": metrics.get("cpu_usage", 0),
                    "memory_usage": metrics.get("memory_usage", 0),
                    "response_time": metrics.get("response_time", 0),
                    "throughput": metrics.get("throughput", 0)
                }]
            
            return []
            
        except Exception as e:
            logger.error(f"Failed to get performance metrics: {e}")
            return []
    
    async def _get_agent_tasks_data(self, query: str) -> List[Dict[str, Any]]:
        """
        Get agent tasks data from Redis
        """
        try:
            # Get task history from Redis
            task_history = redis_client.lrange("task_history", 0, 100)
            
            tasks = []
            for task_data in task_history:
                task = json.loads(task_data)
                tasks.append({
                    "agent_type": task.get("agent_type"),
                    "task_type": task.get("task_type"),
                    "status": task.get("status"),
                    "created_at": task.get("created_at"),
                    "completed_at": task.get("completed_at")
                })
            
            return tasks
            
        except Exception as e:
            logger.error(f"Failed to get agent tasks data: {e}")
            return []
    
    async def _get_business_metrics_data(self, query: str) -> List[Dict[str, Any]]:
        """
        Get business metrics data
        """
        try:
            # Simulate business metrics (would come from actual business database)
            current_date = datetime.utcnow().date()
            
            metrics = []
            for i in range(7):
                date = current_date - timedelta(days=i)
                metrics.append({
                    "date": date.isoformat(),
                    "conversion_rate": 0.025 + (i * 0.001),  # 2.5-3.1%
                    "revenue": 50000 + (i * 2000),  # $50k-$62k
                    "user_engagement": 0.65 + (i * 0.02),  # 65-77%
                    "recommendation_ctr": 0.12 + (i * 0.005)  # 12-15%
                })
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to get business metrics: {e}")
            return []
    
    async def _get_error_logs_data(self, query: str) -> List[Dict[str, Any]]:
        """
        Get error logs data
        """
        try:
            # Get error logs from Redis
            error_logs = redis_client.lrange("error_logs", 0, 200)
            
            errors_by_hour = defaultdict(lambda: defaultdict(int))
            
            for error_data in error_logs:
                error = json.loads(error_data)
                timestamp = datetime.fromisoformat(error.get("timestamp"))
                hour = timestamp.replace(minute=0, second=0, microsecond=0)
                service = error.get("service", "unknown")
                
                errors_by_hour[hour][service] += 1
            
            # Convert to list format
            result = []
            for hour, services in errors_by_hour.items():
                for service, count in services.items():
                    result.append({
                        "hour": hour.isoformat(),
                        "error_count": count,
                        "service": service
                    })
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to get error logs data: {e}")
            return []
    
    async def _get_user_activities_data(self, query: str) -> List[Dict[str, Any]]:
        """
        Get user activities data
        """
        try:
            # Simulate user activity heatmap data
            activities = []
            
            for day in range(7):  # Days of week (0=Sunday)
                for hour in range(24):  # Hours of day
                    # Simulate activity patterns
                    base_activity = 100
                    
                    # Higher activity during business hours
                    if 9 <= hour <= 17:
                        base_activity *= 2
                    
                    # Lower activity on weekends
                    if day in [0, 6]:  # Sunday, Saturday
                        base_activity *= 0.7
                    
                    # Random variation
                    activity_count = int(base_activity * (0.8 + np.random.random() * 0.4))
                    
                    activities.append({
                        "hour": hour,
                        "day_of_week": day,
                        "activity_count": activity_count
                    })
            
            return activities
            
        except Exception as e:
            logger.error(f"Failed to get user activities data: {e}")
            return []
    
    async def _get_recommendation_metrics_data(self, query: str) -> List[Dict[str, Any]]:
        """
        Get recommendation engine metrics
        """
        try:
            # Get recommendation metrics from Redis
            metrics_data = redis_client.get("recommendation_metrics")
            if metrics_data:
                metrics = json.loads(metrics_data)
                return [metrics]
            
            # Default metrics if not available
            return [{
                "accuracy": 0.85,
                "precision": 0.78,
                "recall": 0.72,
                "f1_score": 0.75
            }]
            
        except Exception as e:
            logger.error(f"Failed to get recommendation metrics: {e}")
            return []
    
    async def _execute_database_query(self, query: str) -> List[Dict[str, Any]]:
        """
        Execute generic database query
        """
        try:
            db = next(get_db())
            result = db.execute(text(query))
            rows = result.fetchall()
            
            # Convert to dict format
            columns = result.keys()
            data = [dict(zip(columns, row)) for row in rows]
            
            db.close()
            return data
            
        except Exception as e:
            logger.error(f"Database query failed: {e}")
            return []
    
    async def _format_widget_data(self, widget: DashboardWidget, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Format widget data according to visualization configuration
        """
        try:
            formatted_data = {
                "widget_id": widget.widget_id,
                "title": widget.title,
                "widget_type": widget.widget_type,
                "data": data,
                "config": widget.visualization_config,
                "last_updated": datetime.utcnow().isoformat(),
                "data_count": len(data)
            }
            
            # Add widget-specific formatting
            if widget.widget_type == "metric_grid":
                formatted_data["summary"] = self._calculate_metric_summary(data, widget.visualization_config)
            elif widget.widget_type == "line_chart":
                formatted_data["chart_data"] = self._format_chart_data(data, widget.visualization_config)
            elif widget.widget_type == "heatmap":
                formatted_data["heatmap_data"] = self._format_heatmap_data(data, widget.visualization_config)
            
            return formatted_data
            
        except Exception as e:
            logger.error(f"Failed to format widget data: {e}")
            return {"error": str(e)}
    
    def _calculate_metric_summary(self, data: List[Dict[str, Any]], config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate summary statistics for metric widgets
        """
        if not data:
            return {}
        
        summary = {}
        metrics_config = config.get("metrics", [])
        
        for metric_config in metrics_config:
            field = metric_config["field"]
            if field in data[0]:
                value = data[0][field]
                threshold = metric_config.get("threshold")
                
                summary[field] = {
                    "value": value,
                    "unit": metric_config.get("unit", ""),
                    "status": "normal"
                }
                
                # Check threshold
                if threshold and value > threshold:
                    summary[field]["status"] = "warning"
        
        return summary
    
    def _format_chart_data(self, data: List[Dict[str, Any]], config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format data for chart visualizations
        """
        if not data:
            return {}
        
        x_axis = config.get("x_axis", "x")
        y_axis = config.get("y_axis", "y")
        group_by = config.get("group_by")
        
        chart_data = {
            "x_values": [],
            "series": {}
        }
        
        if group_by:
            # Group data by specified field
            grouped_data = defaultdict(list)
            for row in data:
                group_value = row.get(group_by, "default")
                grouped_data[group_value].append(row)
            
            for group, group_data in grouped_data.items():
                chart_data["series"][group] = {
                    "x_values": [row.get(x_axis) for row in group_data],
                    "y_values": [row.get(y_axis) for row in group_data]
                }
        else:
            chart_data["x_values"] = [row.get(x_axis) for row in data]
            chart_data["y_values"] = [row.get(y_axis) for row in data]
        
        return chart_data
    
    def _format_heatmap_data(self, data: List[Dict[str, Any]], config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format data for heatmap visualization
        """
        if not data:
            return {}
        
        x_axis = config.get("x_axis", "x")
        y_axis = config.get("y_axis", "y")
        value_field = config.get("value_field", "value")
        
        heatmap_data = {
            "matrix": [],
            "x_labels": [],
            "y_labels": []
        }
        
        # Create matrix from data
        matrix = {}
        x_values = set()
        y_values = set()
        
        for row in data:
            x_val = row.get(x_axis)
            y_val = row.get(y_axis)
            value = row.get(value_field, 0)
            
            matrix[(x_val, y_val)] = value
            x_values.add(x_val)
            y_values.add(y_val)
        
        x_sorted = sorted(list(x_values))
        y_sorted = sorted(list(y_values))
        
        # Build matrix
        for y_val in y_sorted:
            row = []
            for x_val in x_sorted:
                row.append(matrix.get((x_val, y_val), 0))
            heatmap_data["matrix"].append(row)
        
        heatmap_data["x_labels"] = x_sorted
        heatmap_data["y_labels"] = y_sorted
        
        return heatmap_data
    
    async def _refresh_analytics_reports(self):
        """
        Refresh analytics reports
        """
        try:
            # Generate daily report
            if "daily_report" not in self.reports or self._should_refresh_report("daily_report", hours=24):
                daily_report = await self._generate_daily_report()
                self.reports["daily_report"] = daily_report
                
                # Store in Redis
                redis_client.setex(
                    "analytics_report:daily",
                    86400,  # 24 hours
                    json.dumps(asdict(daily_report), default=str)
                )
            
            # Generate weekly report
            if "weekly_report" not in self.reports or self._should_refresh_report("weekly_report", hours=168):
                weekly_report = await self._generate_weekly_report()
                self.reports["weekly_report"] = weekly_report
                
                # Store in Redis
                redis_client.setex(
                    "analytics_report:weekly",
                    604800,  # 7 days
                    json.dumps(asdict(weekly_report), default=str)
                )
                
        except Exception as e:
            logger.error(f"Failed to refresh analytics reports: {e}")
    
    def _should_refresh_report(self, report_id: str, hours: int) -> bool:
        """
        Check if analytics report should be refreshed
        """
        if report_id not in self.reports:
            return True
        
        report = self.reports[report_id]
        time_since_generation = datetime.utcnow() - report.generated_at
        
        return time_since_generation > timedelta(hours=hours)
    
    async def _generate_daily_report(self) -> AnalyticsReport:
        """
        Generate daily analytics report
        """
        try:
            # Collect metrics for the last 24 hours
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=24)
            
            # Performance metrics
            performance_metrics = await self.performance_analytics.get_daily_summary(start_time, end_time)
            
            # Business metrics
            business_metrics = await self.business_analytics.get_daily_summary(start_time, end_time)
            
            # User behavior metrics
            user_metrics = await self.user_analytics.get_daily_summary(start_time, end_time)
            
            # Operational metrics
            operational_metrics = await self.operational_analytics.get_daily_summary(start_time, end_time)
            
            # Generate insights
            insights = await self._generate_insights({
                "performance": performance_metrics,
                "business": business_metrics,
                "user": user_metrics,
                "operational": operational_metrics
            })
            
            # Generate recommendations
            recommendations = await self._generate_recommendations(insights)
            
            report = AnalyticsReport(
                report_id="daily_report",
                title="Daily Analytics Report",
                description="Comprehensive daily performance and business metrics",
                time_range=TimeRange.DAY,
                metrics={
                    "performance": performance_metrics,
                    "business": business_metrics,
                    "user": user_metrics,
                    "operational": operational_metrics
                },
                charts=[],  # Would include chart configurations
                insights=insights,
                recommendations=recommendations,
                generated_at=datetime.utcnow(),
                data_freshness=datetime.utcnow()
            )
            
            return report
            
        except Exception as e:
            logger.error(f"Failed to generate daily report: {e}")
            return self._create_error_report("daily_report", str(e))
    
    async def _generate_weekly_report(self) -> AnalyticsReport:
        """
        Generate weekly analytics report
        """
        try:
            # Similar to daily report but with weekly data
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(weeks=1)
            
            # Performance trends
            performance_trends = await self.performance_analytics.get_weekly_trends(start_time, end_time)
            
            # Business growth metrics
            business_growth = await self.business_analytics.get_weekly_growth(start_time, end_time)
            
            # User engagement trends
            user_engagement = await self.user_analytics.get_weekly_engagement(start_time, end_time)
            
            # System reliability
            system_reliability = await self.operational_analytics.get_weekly_reliability(start_time, end_time)
            
            insights = await self._generate_weekly_insights({
                "performance_trends": performance_trends,
                "business_growth": business_growth,
                "user_engagement": user_engagement,
                "system_reliability": system_reliability
            })
            
            recommendations = await self._generate_strategic_recommendations(insights)
            
            report = AnalyticsReport(
                report_id="weekly_report",
                title="Weekly Analytics Report",
                description="Weekly trends and strategic insights",
                time_range=TimeRange.WEEK,
                metrics={
                    "performance_trends": performance_trends,
                    "business_growth": business_growth,
                    "user_engagement": user_engagement,
                    "system_reliability": system_reliability
                },
                charts=[],
                insights=insights,
                recommendations=recommendations,
                generated_at=datetime.utcnow(),
                data_freshness=datetime.utcnow()
            )
            
            return report
            
        except Exception as e:
            logger.error(f"Failed to generate weekly report: {e}")
            return self._create_error_report("weekly_report", str(e))
    
    async def _generate_insights(self, metrics: Dict[str, Any]) -> List[str]:
        """
        Generate insights from metrics data
        """
        insights = []
        
        try:
            performance = metrics.get("performance", {})
            business = metrics.get("business", {})
            
            # Performance insights
            if performance.get("avg_response_time", 0) > 1000:
                insights.append("Response times are above optimal threshold (>1s). Consider scaling or optimization.")
            
            if performance.get("error_rate", 0) > 0.05:
                insights.append("Error rate is elevated (>5%). Investigation recommended.")
            
            # Business insights
            if business.get("conversion_rate_change", 0) > 0.1:
                insights.append("Conversion rate improved by >10%. Positive trend in user experience.")
            
            if business.get("revenue_growth", 0) < -0.05:
                insights.append("Revenue decline detected. Review recommendation effectiveness.")
            
        except Exception as e:
            logger.error(f"Failed to generate insights: {e}")
            insights.append("Unable to generate insights due to data processing error.")
        
        return insights
    
    async def _generate_recommendations(self, insights: List[str]) -> List[str]:
        """
        Generate actionable recommendations
        """
        recommendations = []
        
        for insight in insights:
            if "response times" in insight.lower():
                recommendations.append("Enable auto-scaling for web servers")
                recommendations.append("Implement caching for frequently accessed data")
            
            elif "error rate" in insight.lower():
                recommendations.append("Review error logs and implement fixes")
                recommendations.append("Enhance monitoring and alerting")
            
            elif "conversion rate" in insight.lower():
                recommendations.append("Continue current optimization strategies")
                recommendations.append("A/B test new recommendation algorithms")
            
            elif "revenue decline" in insight.lower():
                recommendations.append("Audit recommendation model performance")
                recommendations.append("Review user engagement metrics")
        
        return recommendations
    
    async def _generate_weekly_insights(self, metrics: Dict[str, Any]) -> List[str]:
        """
        Generate weekly strategic insights
        """
        insights = []
        
        try:
            # Analyze trends over the week
            performance_trends = metrics.get("performance_trends", {})
            business_growth = metrics.get("business_growth", {})
            
            # Performance trend analysis
            if performance_trends.get("response_time_trend", "stable") == "improving":
                insights.append("System performance has been improving consistently over the week.")
            
            # Business growth analysis
            growth_rate = business_growth.get("weekly_growth_rate", 0)
            if growth_rate > 0.05:
                insights.append(f"Strong business growth of {growth_rate*100:.1f}% this week.")
            
        except Exception as e:
            logger.error(f"Failed to generate weekly insights: {e}")
        
        return insights
    
    async def _generate_strategic_recommendations(self, insights: List[str]) -> List[str]:
        """
        Generate strategic recommendations for weekly report
        """
        recommendations = []
        
        for insight in insights:
            if "improving" in insight.lower():
                recommendations.append("Document successful optimization practices")
                recommendations.append("Share performance improvements across teams")
            
            elif "growth" in insight.lower():
                recommendations.append("Scale infrastructure to support continued growth")
                recommendations.append("Invest in advanced analytics capabilities")
        
        return recommendations
    
    def _create_error_report(self, report_id: str, error_message: str) -> AnalyticsReport:
        """
        Create error report when generation fails
        """
        return AnalyticsReport(
            report_id=report_id,
            title=f"Error Report - {report_id}",
            description=f"Report generation failed: {error_message}",
            time_range=TimeRange.DAY,
            metrics={},
            charts=[],
            insights=[f"Report generation error: {error_message}"],
            recommendations=["Fix data collection issues", "Review system health"],
            generated_at=datetime.utcnow(),
            data_freshness=datetime.utcnow()
        )
    
    # Public API methods
    
    async def get_dashboard_data(self, widget_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Get current dashboard data for specified widgets
        """
        dashboard_data = {
            "widgets": {},
            "last_updated": datetime.utcnow().isoformat(),
            "status": "healthy"
        }
        
        widgets_to_fetch = widget_ids if widget_ids else list(self.widgets.keys())
        
        for widget_id in widgets_to_fetch:
            if widget_id in self.cached_data:
                dashboard_data["widgets"][widget_id] = self.cached_data[widget_id]
            else:
                # Try to get from Redis
                redis_data = redis_client.get(f"widget_data:{widget_id}")
                if redis_data:
                    dashboard_data["widgets"][widget_id] = json.loads(redis_data)
                else:
                    dashboard_data["widgets"][widget_id] = {"error": "No data available"}
        
        return dashboard_data
    
    async def get_analytics_report(self, report_id: str) -> Optional[AnalyticsReport]:
        """
        Get analytics report by ID
        """
        if report_id in self.reports:
            return self.reports[report_id]
        
        # Try to get from Redis
        redis_data = redis_client.get(f"analytics_report:{report_id}")
        if redis_data:
            report_data = json.loads(redis_data)
            return AnalyticsReport(**report_data)
        
        return None
    
    async def create_custom_widget(self, widget: DashboardWidget):
        """
        Create a custom dashboard widget
        """
        self.widgets[widget.widget_id] = widget
        logger.info(f"Created custom widget: {widget.widget_id}")
    
    async def remove_widget(self, widget_id: str):
        """
        Remove a dashboard widget
        """
        if widget_id in self.widgets:
            del self.widgets[widget_id]
            if widget_id in self.cached_data:
                del self.cached_data[widget_id]
            if widget_id in self.cache_ttl:
                del self.cache_ttl[widget_id]
            
            logger.info(f"Removed widget: {widget_id}")
    
    def get_available_widgets(self) -> List[Dict[str, Any]]:
        """
        Get list of available widgets
        """
        widget_list = []
        for widget_id, widget in self.widgets.items():
            widget_list.append({
                "widget_id": widget_id,
                "title": widget.title,
                "widget_type": widget.widget_type,
                "data_source": widget.data_source,
                "refresh_interval": widget.refresh_interval,
                "last_updated": self.cache_ttl.get(widget_id, "Never").isoformat() if isinstance(self.cache_ttl.get(widget_id), datetime) else "Never"
            })
        
        return widget_list


# Analytics Engines

class BusinessAnalyticsEngine:
    """Business metrics analytics engine"""
    
    async def get_daily_summary(self, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        # Implement business metrics collection
        return {
            "total_revenue": 125000.0,
            "conversion_rate": 0.028,
            "average_order_value": 89.50,
            "user_acquisition": 450,
            "recommendation_ctr": 0.14,
            "conversion_rate_change": 0.05
        }
    
    async def get_weekly_growth(self, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        return {
            "weekly_growth_rate": 0.07,
            "revenue_trend": "increasing",
            "user_growth": 0.12,
            "engagement_change": 0.08
        }


class PerformanceAnalyticsEngine:
    """Performance metrics analytics engine"""
    
    async def get_daily_summary(self, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        return {
            "avg_response_time": 750.0,
            "error_rate": 0.02,
            "uptime": 99.9,
            "throughput": 1250,
            "cpu_utilization": 65.0,
            "memory_utilization": 72.0
        }
    
    async def get_weekly_trends(self, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        return {
            "response_time_trend": "improving",
            "error_rate_trend": "stable",
            "resource_efficiency": "optimized"
        }


class UserAnalyticsEngine:
    """User behavior analytics engine"""
    
    async def get_daily_summary(self, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        return {
            "active_users": 12500,
            "session_duration": 8.5,
            "page_views": 75000,
            "bounce_rate": 0.32,
            "user_satisfaction": 4.2
        }
    
    async def get_weekly_engagement(self, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        return {
            "engagement_trend": "increasing",
            "retention_rate": 0.78,
            "feature_adoption": 0.65
        }


class OperationalAnalyticsEngine:
    """Operational metrics analytics engine"""
    
    async def get_daily_summary(self, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        return {
            "deployment_frequency": 2,
            "lead_time": 3.5,
            "mttr": 15.0,
            "change_failure_rate": 0.05
        }
    
    async def get_weekly_reliability(self, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        return {
            "availability": 99.95,
            "incident_count": 2,
            "resolved_incidents": 2,
            "security_events": 0
        }