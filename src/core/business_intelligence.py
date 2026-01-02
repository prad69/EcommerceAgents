import asyncio
from typing import Dict, List, Any, Optional, Tuple, Union
from loguru import logger
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict
import statistics
from sqlalchemy import text, func
from sqlalchemy.orm import Session

from src.core.database import get_db, redis_client
from src.core.analytics_dashboard import AnalyticsDashboard


class ReportType(Enum):
    EXECUTIVE_SUMMARY = "executive_summary"
    OPERATIONAL_METRICS = "operational_metrics"
    CUSTOMER_INSIGHTS = "customer_insights"
    REVENUE_ANALYSIS = "revenue_analysis"
    PRODUCT_PERFORMANCE = "product_performance"
    AGENT_EFFECTIVENESS = "agent_effectiveness"
    MARKET_INTELLIGENCE = "market_intelligence"
    PREDICTIVE_ANALYTICS = "predictive_analytics"


class ReportFrequency(Enum):
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUAL = "annual"
    AD_HOC = "ad_hoc"


class DataAggregation(Enum):
    SUM = "sum"
    AVERAGE = "average"
    COUNT = "count"
    MEDIAN = "median"
    PERCENTILE = "percentile"
    TREND = "trend"
    GROWTH_RATE = "growth_rate"


@dataclass
class BusinessMetric:
    """Business metric definition"""
    metric_id: str
    name: str
    description: str
    calculation_method: str
    data_sources: List[str]
    aggregation_type: DataAggregation
    target_value: Optional[float] = None
    warning_threshold: Optional[float] = None
    critical_threshold: Optional[float] = None
    unit: str = ""
    is_higher_better: bool = True


@dataclass
class KPI:
    """Key Performance Indicator"""
    kpi_id: str
    name: str
    current_value: float
    target_value: float
    previous_value: float
    trend: str  # "up", "down", "stable"
    performance_status: str  # "excellent", "good", "warning", "critical"
    change_percentage: float
    unit: str
    category: str


@dataclass
class BusinessInsight:
    """Business insight from data analysis"""
    insight_id: str
    title: str
    description: str
    impact_level: str  # "high", "medium", "low"
    confidence_score: float
    data_sources: List[str]
    recommendations: List[str]
    affected_metrics: List[str]
    generated_at: datetime
    insight_type: str  # "trend", "anomaly", "opportunity", "risk"


@dataclass
class BIReport:
    """Business Intelligence Report"""
    report_id: str
    title: str
    report_type: ReportType
    executive_summary: str
    kpis: List[KPI]
    insights: List[BusinessInsight]
    charts: List[Dict[str, Any]]
    data_tables: List[Dict[str, Any]]
    recommendations: List[str]
    generated_at: datetime
    period_start: datetime
    period_end: datetime
    data_freshness: datetime


class BusinessIntelligence:
    """
    Business Intelligence system for comprehensive reporting and insights
    """
    
    def __init__(self, analytics_dashboard: AnalyticsDashboard = None):
        self.analytics_dashboard = analytics_dashboard or AnalyticsDashboard()
        
        # BI Configuration
        self.business_metrics = {}
        self.kpi_definitions = {}
        self.scheduled_reports = {}
        self.generated_reports = {}
        
        # Data sources
        self.data_connectors = {}
        
        # Analytics engines
        self.revenue_analyzer = RevenueAnalyzer()
        self.customer_analyzer = CustomerAnalyzer()
        self.product_analyzer = ProductPerformanceAnalyzer()
        self.agent_analyzer = AgentEffectivenessAnalyzer()
        self.market_analyzer = MarketIntelligenceAnalyzer()
        self.predictive_analyzer = PredictiveAnalyzer()
        
        logger.info("Business Intelligence system initialized")
        
        # Setup default metrics and KPIs
        self._setup_default_metrics()
        self._setup_default_kpis()
        
        # Start report scheduler
        asyncio.create_task(self._start_report_scheduler())
    
    def _setup_default_metrics(self):
        """Setup default business metrics"""
        
        # Revenue metrics
        revenue_metric = BusinessMetric(
            metric_id="total_revenue",
            name="Total Revenue",
            description="Total revenue across all products and channels",
            calculation_method="SUM(order_amount)",
            data_sources=["orders", "payments"],
            aggregation_type=DataAggregation.SUM,
            target_value=1000000.0,
            warning_threshold=800000.0,
            critical_threshold=600000.0,
            unit="USD",
            is_higher_better=True
        )
        
        # Conversion rate metric
        conversion_metric = BusinessMetric(
            metric_id="conversion_rate",
            name="Conversion Rate",
            description="Percentage of visitors who make a purchase",
            calculation_method="(conversions / visits) * 100",
            data_sources=["user_sessions", "orders"],
            aggregation_type=DataAggregation.AVERAGE,
            target_value=3.0,
            warning_threshold=2.0,
            critical_threshold=1.5,
            unit="%",
            is_higher_better=True
        )
        
        # Customer satisfaction metric
        satisfaction_metric = BusinessMetric(
            metric_id="customer_satisfaction",
            name="Customer Satisfaction Score",
            description="Average customer satisfaction rating",
            calculation_method="AVG(satisfaction_rating)",
            data_sources=["reviews", "feedback"],
            aggregation_type=DataAggregation.AVERAGE,
            target_value=4.5,
            warning_threshold=4.0,
            critical_threshold=3.5,
            unit="/5",
            is_higher_better=True
        )
        
        # Agent effectiveness metric
        agent_effectiveness_metric = BusinessMetric(
            metric_id="agent_effectiveness",
            name="AI Agent Effectiveness",
            description="Overall effectiveness of AI agents in improving business metrics",
            calculation_method="Composite score based on recommendation accuracy, response time, and user satisfaction",
            data_sources=["agent_performance", "user_interactions"],
            aggregation_type=DataAggregation.AVERAGE,
            target_value=85.0,
            warning_threshold=70.0,
            critical_threshold=60.0,
            unit="%",
            is_higher_better=True
        )
        
        self.business_metrics = {
            "total_revenue": revenue_metric,
            "conversion_rate": conversion_metric,
            "customer_satisfaction": satisfaction_metric,
            "agent_effectiveness": agent_effectiveness_metric
        }
    
    def _setup_default_kpis(self):
        """Setup default KPI definitions"""
        
        self.kpi_definitions = {
            "revenue_growth": {
                "name": "Revenue Growth Rate",
                "calculation": "monthly_revenue_change_percentage",
                "target": 15.0,
                "unit": "%"
            },
            "customer_acquisition_cost": {
                "name": "Customer Acquisition Cost",
                "calculation": "marketing_spend / new_customers",
                "target": 25.0,
                "unit": "USD"
            },
            "customer_lifetime_value": {
                "name": "Customer Lifetime Value",
                "calculation": "average_order_value * purchase_frequency * customer_lifespan",
                "target": 500.0,
                "unit": "USD"
            },
            "recommendation_accuracy": {
                "name": "AI Recommendation Accuracy",
                "calculation": "successful_recommendations / total_recommendations",
                "target": 80.0,
                "unit": "%"
            },
            "system_uptime": {
                "name": "System Uptime",
                "calculation": "uptime_hours / total_hours",
                "target": 99.9,
                "unit": "%"
            }
        }
    
    async def _start_report_scheduler(self):
        """Start scheduler for automated report generation"""
        while True:
            try:
                current_time = datetime.utcnow()
                
                # Check for scheduled reports
                for schedule_id, schedule in self.scheduled_reports.items():
                    if self._should_generate_report(schedule, current_time):
                        await self._generate_scheduled_report(schedule)
                
                # Generate daily executive summary
                if current_time.hour == 8 and current_time.minute == 0:  # 8 AM daily
                    await self._generate_executive_summary()
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Report scheduler error: {e}")
                await asyncio.sleep(300)  # 5 minutes on error
    
    def _should_generate_report(self, schedule: Dict[str, Any], current_time: datetime) -> bool:
        """Check if a scheduled report should be generated"""
        frequency = schedule.get("frequency")
        last_generated = schedule.get("last_generated")
        
        if not last_generated:
            return True
        
        time_since_last = current_time - last_generated
        
        if frequency == ReportFrequency.DAILY and time_since_last >= timedelta(days=1):
            return True
        elif frequency == ReportFrequency.WEEKLY and time_since_last >= timedelta(weeks=1):
            return True
        elif frequency == ReportFrequency.MONTHLY and time_since_last >= timedelta(days=30):
            return True
        elif frequency == ReportFrequency.QUARTERLY and time_since_last >= timedelta(days=90):
            return True
        
        return False
    
    async def _generate_scheduled_report(self, schedule: Dict[str, Any]):
        """Generate a scheduled report"""
        try:
            report_type = ReportType(schedule["report_type"])
            
            if report_type == ReportType.EXECUTIVE_SUMMARY:
                report = await self._generate_executive_summary()
            elif report_type == ReportType.REVENUE_ANALYSIS:
                report = await self._generate_revenue_analysis_report()
            elif report_type == ReportType.CUSTOMER_INSIGHTS:
                report = await self._generate_customer_insights_report()
            elif report_type == ReportType.PRODUCT_PERFORMANCE:
                report = await self._generate_product_performance_report()
            elif report_type == ReportType.AGENT_EFFECTIVENESS:
                report = await self._generate_agent_effectiveness_report()
            else:
                logger.warning(f"Unknown report type: {report_type}")
                return
            
            # Update schedule
            schedule["last_generated"] = datetime.utcnow()
            
            # Store report
            self.generated_reports[report.report_id] = report
            
            # Cache in Redis
            await self._cache_report(report)
            
            logger.info(f"Generated scheduled report: {report.report_id}")
            
        except Exception as e:
            logger.error(f"Failed to generate scheduled report: {e}")
    
    async def _generate_executive_summary(self) -> BIReport:
        """Generate executive summary report"""
        try:
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=30)  # Last 30 days
            
            # Collect KPIs
            kpis = await self._calculate_kpis(start_date, end_date)
            
            # Generate insights
            insights = await self._generate_executive_insights(kpis)
            
            # Create executive summary
            executive_summary = await self._create_executive_summary_text(kpis, insights)
            
            # Generate charts
            charts = await self._create_executive_charts(start_date, end_date)
            
            # Generate recommendations
            recommendations = await self._generate_executive_recommendations(kpis, insights)
            
            report = BIReport(
                report_id=f"executive_summary_{end_date.strftime('%Y%m%d')}",
                title="Executive Summary Report",
                report_type=ReportType.EXECUTIVE_SUMMARY,
                executive_summary=executive_summary,
                kpis=kpis,
                insights=insights,
                charts=charts,
                data_tables=[],
                recommendations=recommendations,
                generated_at=datetime.utcnow(),
                period_start=start_date,
                period_end=end_date,
                data_freshness=datetime.utcnow()
            )
            
            return report
            
        except Exception as e:
            logger.error(f"Failed to generate executive summary: {e}")
            raise
    
    async def _calculate_kpis(self, start_date: datetime, end_date: datetime) -> List[KPI]:
        """Calculate current KPIs"""
        kpis = []
        
        try:
            # Revenue KPIs
            current_revenue = await self.revenue_analyzer.get_total_revenue(start_date, end_date)
            previous_revenue = await self.revenue_analyzer.get_total_revenue(
                start_date - timedelta(days=30), 
                start_date
            )
            
            revenue_change = ((current_revenue - previous_revenue) / previous_revenue * 100) if previous_revenue > 0 else 0
            
            revenue_kpi = KPI(
                kpi_id="total_revenue",
                name="Total Revenue",
                current_value=current_revenue,
                target_value=1000000.0,
                previous_value=previous_revenue,
                trend="up" if revenue_change > 0 else "down" if revenue_change < 0 else "stable",
                performance_status=self._calculate_performance_status(current_revenue, 1000000.0, 800000.0, 600000.0),
                change_percentage=revenue_change,
                unit="USD",
                category="Financial"
            )
            kpis.append(revenue_kpi)
            
            # Conversion Rate KPI
            current_conversion = await self.customer_analyzer.get_conversion_rate(start_date, end_date)
            previous_conversion = await self.customer_analyzer.get_conversion_rate(
                start_date - timedelta(days=30),
                start_date
            )
            
            conversion_change = ((current_conversion - previous_conversion) / previous_conversion * 100) if previous_conversion > 0 else 0
            
            conversion_kpi = KPI(
                kpi_id="conversion_rate",
                name="Conversion Rate",
                current_value=current_conversion,
                target_value=3.0,
                previous_value=previous_conversion,
                trend="up" if conversion_change > 0 else "down" if conversion_change < 0 else "stable",
                performance_status=self._calculate_performance_status(current_conversion, 3.0, 2.0, 1.5),
                change_percentage=conversion_change,
                unit="%",
                category="Customer"
            )
            kpis.append(conversion_kpi)
            
            # Agent Effectiveness KPI
            agent_effectiveness = await self.agent_analyzer.get_overall_effectiveness(start_date, end_date)
            previous_effectiveness = await self.agent_analyzer.get_overall_effectiveness(
                start_date - timedelta(days=30),
                start_date
            )
            
            effectiveness_change = ((agent_effectiveness - previous_effectiveness) / previous_effectiveness * 100) if previous_effectiveness > 0 else 0
            
            effectiveness_kpi = KPI(
                kpi_id="agent_effectiveness",
                name="AI Agent Effectiveness",
                current_value=agent_effectiveness,
                target_value=85.0,
                previous_value=previous_effectiveness,
                trend="up" if effectiveness_change > 0 else "down" if effectiveness_change < 0 else "stable",
                performance_status=self._calculate_performance_status(agent_effectiveness, 85.0, 70.0, 60.0),
                change_percentage=effectiveness_change,
                unit="%",
                category="Operational"
            )
            kpis.append(effectiveness_kpi)
            
            # Customer Satisfaction KPI
            satisfaction_score = await self.customer_analyzer.get_satisfaction_score(start_date, end_date)
            previous_satisfaction = await self.customer_analyzer.get_satisfaction_score(
                start_date - timedelta(days=30),
                start_date
            )
            
            satisfaction_change = ((satisfaction_score - previous_satisfaction) / previous_satisfaction * 100) if previous_satisfaction > 0 else 0
            
            satisfaction_kpi = KPI(
                kpi_id="customer_satisfaction",
                name="Customer Satisfaction",
                current_value=satisfaction_score,
                target_value=4.5,
                previous_value=previous_satisfaction,
                trend="up" if satisfaction_change > 0 else "down" if satisfaction_change < 0 else "stable",
                performance_status=self._calculate_performance_status(satisfaction_score, 4.5, 4.0, 3.5),
                change_percentage=satisfaction_change,
                unit="/5",
                category="Customer"
            )
            kpis.append(satisfaction_kpi)
            
        except Exception as e:
            logger.error(f"Failed to calculate KPIs: {e}")
        
        return kpis
    
    def _calculate_performance_status(self, current: float, target: float, warning: float, critical: float) -> str:
        """Calculate performance status based on thresholds"""
        if current >= target:
            return "excellent"
        elif current >= warning:
            return "good"
        elif current >= critical:
            return "warning"
        else:
            return "critical"
    
    async def _generate_executive_insights(self, kpis: List[KPI]) -> List[BusinessInsight]:
        """Generate insights for executive summary"""
        insights = []
        
        try:
            # Revenue insights
            revenue_kpi = next((kpi for kpi in kpis if kpi.kpi_id == "total_revenue"), None)
            if revenue_kpi:
                if revenue_kpi.change_percentage > 10:
                    insights.append(BusinessInsight(
                        insight_id="revenue_growth_strong",
                        title="Strong Revenue Growth",
                        description=f"Revenue has increased by {revenue_kpi.change_percentage:.1f}% compared to the previous period, indicating strong business performance.",
                        impact_level="high",
                        confidence_score=0.9,
                        data_sources=["revenue_data", "orders"],
                        recommendations=[
                            "Continue current growth strategies",
                            "Consider scaling marketing efforts",
                            "Invest in infrastructure to support growth"
                        ],
                        affected_metrics=["total_revenue", "customer_acquisition"],
                        generated_at=datetime.utcnow(),
                        insight_type="opportunity"
                    ))
                elif revenue_kpi.change_percentage < -5:
                    insights.append(BusinessInsight(
                        insight_id="revenue_decline_concern",
                        title="Revenue Decline Concern",
                        description=f"Revenue has decreased by {abs(revenue_kpi.change_percentage):.1f}% compared to the previous period, requiring immediate attention.",
                        impact_level="high",
                        confidence_score=0.95,
                        data_sources=["revenue_data", "orders"],
                        recommendations=[
                            "Investigate root causes of revenue decline",
                            "Review marketing and sales strategies",
                            "Analyze competitor activities"
                        ],
                        affected_metrics=["total_revenue", "conversion_rate"],
                        generated_at=datetime.utcnow(),
                        insight_type="risk"
                    ))
            
            # Conversion rate insights
            conversion_kpi = next((kpi for kpi in kpis if kpi.kpi_id == "conversion_rate"), None)
            if conversion_kpi and conversion_kpi.change_percentage > 5:
                insights.append(BusinessInsight(
                    insight_id="conversion_improvement",
                    title="Conversion Rate Improvement",
                    description=f"Conversion rate has improved by {conversion_kpi.change_percentage:.1f}%, indicating better user experience and product-market fit.",
                    impact_level="medium",
                    confidence_score=0.85,
                    data_sources=["user_sessions", "orders"],
                    recommendations=[
                        "Analyze successful conversion factors",
                        "Replicate successful strategies",
                        "Continue optimizing user experience"
                    ],
                    affected_metrics=["conversion_rate", "customer_satisfaction"],
                    generated_at=datetime.utcnow(),
                    insight_type="opportunity"
                ))
            
            # Agent effectiveness insights
            agent_kpi = next((kpi for kpi in kpis if kpi.kpi_id == "agent_effectiveness"), None)
            if agent_kpi:
                if agent_kpi.current_value > 80:
                    insights.append(BusinessInsight(
                        insight_id="agent_high_performance",
                        title="High AI Agent Performance",
                        description=f"AI agents are performing exceptionally well with {agent_kpi.current_value:.1f}% effectiveness, contributing to improved customer experience.",
                        impact_level="medium",
                        confidence_score=0.9,
                        data_sources=["agent_performance", "user_feedback"],
                        recommendations=[
                            "Document best practices",
                            "Consider expanding AI agent capabilities",
                            "Share learnings across teams"
                        ],
                        affected_metrics=["agent_effectiveness", "customer_satisfaction"],
                        generated_at=datetime.utcnow(),
                        insight_type="opportunity"
                    ))
                elif agent_kpi.current_value < 65:
                    insights.append(BusinessInsight(
                        insight_id="agent_underperformance",
                        title="AI Agent Performance Concerns",
                        description=f"AI agents are underperforming with {agent_kpi.current_value:.1f}% effectiveness, requiring optimization.",
                        impact_level="high",
                        confidence_score=0.85,
                        data_sources=["agent_performance", "error_logs"],
                        recommendations=[
                            "Review and retrain AI models",
                            "Investigate performance bottlenecks",
                            "Improve agent coordination"
                        ],
                        affected_metrics=["agent_effectiveness", "user_satisfaction"],
                        generated_at=datetime.utcnow(),
                        insight_type="risk"
                    ))
            
        except Exception as e:
            logger.error(f"Failed to generate executive insights: {e}")
        
        return insights
    
    async def _create_executive_summary_text(self, kpis: List[KPI], insights: List[BusinessInsight]) -> str:
        """Create executive summary text"""
        try:
            summary_parts = []
            
            # Overall performance
            excellent_kpis = [kpi for kpi in kpis if kpi.performance_status == "excellent"]
            warning_kpis = [kpi for kpi in kpis if kpi.performance_status in ["warning", "critical"]]
            
            if len(excellent_kpis) > len(warning_kpis):
                summary_parts.append("Overall business performance is strong with most KPIs meeting or exceeding targets.")
            else:
                summary_parts.append("Business performance shows mixed results with several KPIs requiring attention.")
            
            # Key highlights
            summary_parts.append("\n\nKey Highlights:")
            for kpi in kpis[:3]:  # Top 3 KPIs
                trend_description = "increased" if kpi.trend == "up" else "decreased" if kpi.trend == "down" else "remained stable"
                summary_parts.append(f"• {kpi.name}: {kpi.current_value:.1f}{kpi.unit}, {trend_description} by {abs(kpi.change_percentage):.1f}%")
            
            # High-impact insights
            high_impact_insights = [insight for insight in insights if insight.impact_level == "high"]
            if high_impact_insights:
                summary_parts.append("\n\nCritical Insights:")
                for insight in high_impact_insights[:2]:  # Top 2 high-impact insights
                    summary_parts.append(f"• {insight.title}: {insight.description}")
            
            return "\n".join(summary_parts)
            
        except Exception as e:
            logger.error(f"Failed to create executive summary text: {e}")
            return "Executive summary generation failed due to data processing error."
    
    async def _create_executive_charts(self, start_date: datetime, end_date: datetime) -> List[Dict[str, Any]]:
        """Create charts for executive summary"""
        charts = []
        
        try:
            # Revenue trend chart
            revenue_chart = {
                "chart_id": "revenue_trend",
                "title": "Revenue Trend (30 Days)",
                "type": "line_chart",
                "data": await self.revenue_analyzer.get_daily_revenue_trend(start_date, end_date),
                "config": {
                    "x_axis": "date",
                    "y_axis": "revenue",
                    "color": "#2E7D32"
                }
            }
            charts.append(revenue_chart)
            
            # KPI dashboard chart
            kpi_chart = {
                "chart_id": "kpi_dashboard",
                "title": "Key Performance Indicators",
                "type": "gauge_chart",
                "data": await self._get_kpi_chart_data(),
                "config": {
                    "thresholds": {
                        "excellent": 90,
                        "good": 70,
                        "warning": 50,
                        "critical": 30
                    }
                }
            }
            charts.append(kpi_chart)
            
            # Customer metrics pie chart
            customer_chart = {
                "chart_id": "customer_metrics",
                "title": "Customer Acquisition Channels",
                "type": "pie_chart",
                "data": await self.customer_analyzer.get_acquisition_channels_data(start_date, end_date),
                "config": {
                    "value_field": "customers",
                    "label_field": "channel"
                }
            }
            charts.append(customer_chart)
            
        except Exception as e:
            logger.error(f"Failed to create executive charts: {e}")
        
        return charts
    
    async def _get_kpi_chart_data(self) -> List[Dict[str, Any]]:
        """Get KPI data for chart visualization"""
        # Simplified KPI data for gauge chart
        return [
            {"name": "Revenue Growth", "value": 12.5, "target": 15.0},
            {"name": "Conversion Rate", "value": 2.8, "target": 3.0},
            {"name": "Customer Satisfaction", "value": 4.2, "target": 4.5},
            {"name": "Agent Effectiveness", "value": 78.5, "target": 85.0}
        ]
    
    async def _generate_executive_recommendations(self, kpis: List[KPI], insights: List[BusinessInsight]) -> List[str]:
        """Generate executive recommendations"""
        recommendations = []
        
        try:
            # Analyze KPI performance
            underperforming_kpis = [kpi for kpi in kpis if kpi.performance_status in ["warning", "critical"]]
            
            if underperforming_kpis:
                recommendations.append("Focus on improving underperforming KPIs to drive overall business growth")
                
                for kpi in underperforming_kpis:
                    if kpi.kpi_id == "total_revenue":
                        recommendations.append("Implement revenue optimization strategies including pricing optimization and upselling")
                    elif kpi.kpi_id == "conversion_rate":
                        recommendations.append("Optimize conversion funnel and improve product recommendations")
                    elif kpi.kpi_id == "agent_effectiveness":
                        recommendations.append("Invest in AI model improvements and agent optimization")
            
            # Add insight-based recommendations
            for insight in insights:
                if insight.impact_level == "high":
                    recommendations.extend(insight.recommendations[:2])  # Top 2 recommendations per insight
            
            # Strategic recommendations
            recommendations.append("Continue investing in AI-driven personalization to improve customer experience")
            recommendations.append("Monitor competitive landscape and adjust strategies accordingly")
            
        except Exception as e:
            logger.error(f"Failed to generate executive recommendations: {e}")
        
        return recommendations
    
    async def _generate_revenue_analysis_report(self) -> BIReport:
        """Generate detailed revenue analysis report"""
        try:
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=90)  # Last 90 days
            
            # Revenue analytics
            revenue_metrics = await self.revenue_analyzer.get_comprehensive_metrics(start_date, end_date)
            
            # Revenue insights
            revenue_insights = await self.revenue_analyzer.generate_insights(revenue_metrics)
            
            # Revenue charts
            revenue_charts = await self.revenue_analyzer.create_charts(start_date, end_date)
            
            # Revenue data tables
            revenue_tables = await self.revenue_analyzer.create_data_tables(start_date, end_date)
            
            report = BIReport(
                report_id=f"revenue_analysis_{end_date.strftime('%Y%m%d')}",
                title="Revenue Analysis Report",
                report_type=ReportType.REVENUE_ANALYSIS,
                executive_summary="Detailed analysis of revenue performance, trends, and opportunities.",
                kpis=await self._get_revenue_kpis(start_date, end_date),
                insights=revenue_insights,
                charts=revenue_charts,
                data_tables=revenue_tables,
                recommendations=await self.revenue_analyzer.generate_recommendations(revenue_metrics),
                generated_at=datetime.utcnow(),
                period_start=start_date,
                period_end=end_date,
                data_freshness=datetime.utcnow()
            )
            
            return report
            
        except Exception as e:
            logger.error(f"Failed to generate revenue analysis report: {e}")
            raise
    
    async def _get_revenue_kpis(self, start_date: datetime, end_date: datetime) -> List[KPI]:
        """Get revenue-specific KPIs"""
        # Implementation would return revenue-focused KPIs
        return await self._calculate_kpis(start_date, end_date)
    
    async def _cache_report(self, report: BIReport):
        """Cache report in Redis"""
        try:
            report_data = asdict(report)
            
            # Convert datetime objects to ISO format for JSON serialization
            report_data = self._serialize_datetime_fields(report_data)
            
            redis_client.setex(
                f"bi_report:{report.report_id}",
                86400 * 7,  # 7 days
                json.dumps(report_data, default=str)
            )
            
        except Exception as e:
            logger.error(f"Failed to cache report: {e}")
    
    def _serialize_datetime_fields(self, data: Any) -> Any:
        """Recursively serialize datetime fields to ISO format"""
        if isinstance(data, datetime):
            return data.isoformat()
        elif isinstance(data, dict):
            return {key: self._serialize_datetime_fields(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self._serialize_datetime_fields(item) for item in data]
        else:
            return data
    
    # Public API methods
    
    async def generate_report(self, report_type: ReportType, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None) -> BIReport:
        """Generate a business intelligence report"""
        try:
            if not end_date:
                end_date = datetime.utcnow()
            if not start_date:
                start_date = end_date - timedelta(days=30)
            
            if report_type == ReportType.EXECUTIVE_SUMMARY:
                return await self._generate_executive_summary()
            elif report_type == ReportType.REVENUE_ANALYSIS:
                return await self._generate_revenue_analysis_report()
            elif report_type == ReportType.CUSTOMER_INSIGHTS:
                return await self._generate_customer_insights_report()
            elif report_type == ReportType.PRODUCT_PERFORMANCE:
                return await self._generate_product_performance_report()
            elif report_type == ReportType.AGENT_EFFECTIVENESS:
                return await self._generate_agent_effectiveness_report()
            else:
                raise ValueError(f"Unsupported report type: {report_type}")
                
        except Exception as e:
            logger.error(f"Failed to generate report {report_type}: {e}")
            raise
    
    async def get_report(self, report_id: str) -> Optional[BIReport]:
        """Get a generated report by ID"""
        # Check in-memory cache first
        if report_id in self.generated_reports:
            return self.generated_reports[report_id]
        
        # Check Redis cache
        cached_data = redis_client.get(f"bi_report:{report_id}")
        if cached_data:
            report_data = json.loads(cached_data)
            return BIReport(**report_data)
        
        return None
    
    def schedule_report(self, report_type: ReportType, frequency: ReportFrequency, schedule_id: str):
        """Schedule automatic report generation"""
        self.scheduled_reports[schedule_id] = {
            "report_type": report_type.value,
            "frequency": frequency,
            "last_generated": None,
            "created_at": datetime.utcnow()
        }
        
        logger.info(f"Scheduled {report_type.value} report with frequency {frequency.value}")
    
    def get_available_reports(self) -> List[Dict[str, Any]]:
        """Get list of available reports"""
        reports = []
        
        for report_id, report in self.generated_reports.items():
            reports.append({
                "report_id": report_id,
                "title": report.title,
                "report_type": report.report_type.value,
                "generated_at": report.generated_at.isoformat(),
                "period_start": report.period_start.isoformat(),
                "period_end": report.period_end.isoformat()
            })
        
        return reports
    
    async def get_business_metrics_summary(self) -> Dict[str, Any]:
        """Get current business metrics summary"""
        try:
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=7)  # Last week
            
            kpis = await self._calculate_kpis(start_date, end_date)
            
            summary = {
                "kpis": [asdict(kpi) for kpi in kpis],
                "overall_performance": self._calculate_overall_performance(kpis),
                "trends": self._analyze_kpi_trends(kpis),
                "alerts": self._generate_alerts(kpis),
                "last_updated": datetime.utcnow().isoformat()
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to get business metrics summary: {e}")
            return {"error": str(e)}
    
    def _calculate_overall_performance(self, kpis: List[KPI]) -> Dict[str, Any]:
        """Calculate overall business performance score"""
        if not kpis:
            return {"score": 0, "status": "unknown"}
        
        status_scores = {
            "excellent": 100,
            "good": 75,
            "warning": 50,
            "critical": 25
        }
        
        total_score = sum(status_scores.get(kpi.performance_status, 0) for kpi in kpis)
        average_score = total_score / len(kpis)
        
        if average_score >= 90:
            status = "excellent"
        elif average_score >= 70:
            status = "good"
        elif average_score >= 50:
            status = "warning"
        else:
            status = "critical"
        
        return {"score": average_score, "status": status}
    
    def _analyze_kpi_trends(self, kpis: List[KPI]) -> Dict[str, int]:
        """Analyze KPI trends"""
        trends = {"up": 0, "down": 0, "stable": 0}
        
        for kpi in kpis:
            trends[kpi.trend] += 1
        
        return trends
    
    def _generate_alerts(self, kpis: List[KPI]) -> List[Dict[str, Any]]:
        """Generate alerts for critical KPIs"""
        alerts = []
        
        for kpi in kpis:
            if kpi.performance_status == "critical":
                alerts.append({
                    "type": "critical",
                    "kpi": kpi.name,
                    "message": f"{kpi.name} is critically low at {kpi.current_value:.1f}{kpi.unit}",
                    "priority": "high"
                })
            elif kpi.performance_status == "warning":
                alerts.append({
                    "type": "warning",
                    "kpi": kpi.name,
                    "message": f"{kpi.name} is below target at {kpi.current_value:.1f}{kpi.unit}",
                    "priority": "medium"
                })
        
        return alerts
    
    # Placeholder methods for missing report generators
    async def _generate_customer_insights_report(self) -> BIReport:
        """Generate customer insights report - placeholder"""
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=30)
        
        return BIReport(
            report_id=f"customer_insights_{end_date.strftime('%Y%m%d')}",
            title="Customer Insights Report",
            report_type=ReportType.CUSTOMER_INSIGHTS,
            executive_summary="Customer behavior analysis and insights",
            kpis=[],
            insights=[],
            charts=[],
            data_tables=[],
            recommendations=[],
            generated_at=datetime.utcnow(),
            period_start=start_date,
            period_end=end_date,
            data_freshness=datetime.utcnow()
        )
    
    async def _generate_product_performance_report(self) -> BIReport:
        """Generate product performance report - placeholder"""
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=30)
        
        return BIReport(
            report_id=f"product_performance_{end_date.strftime('%Y%m%d')}",
            title="Product Performance Report",
            report_type=ReportType.PRODUCT_PERFORMANCE,
            executive_summary="Product sales and performance analysis",
            kpis=[],
            insights=[],
            charts=[],
            data_tables=[],
            recommendations=[],
            generated_at=datetime.utcnow(),
            period_start=start_date,
            period_end=end_date,
            data_freshness=datetime.utcnow()
        )
    
    async def _generate_agent_effectiveness_report(self) -> BIReport:
        """Generate agent effectiveness report - placeholder"""
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=30)
        
        return BIReport(
            report_id=f"agent_effectiveness_{end_date.strftime('%Y%m%d')}",
            title="AI Agent Effectiveness Report",
            report_type=ReportType.AGENT_EFFECTIVENESS,
            executive_summary="AI agent performance and effectiveness analysis",
            kpis=[],
            insights=[],
            charts=[],
            data_tables=[],
            recommendations=[],
            generated_at=datetime.utcnow(),
            period_start=start_date,
            period_end=end_date,
            data_freshness=datetime.utcnow()
        )


# Analytics analyzer classes (simplified implementations)

class RevenueAnalyzer:
    async def get_total_revenue(self, start_date: datetime, end_date: datetime) -> float:
        # Simulate revenue calculation
        days = (end_date - start_date).days
        return 50000 * days * (1 + np.random.random() * 0.2)
    
    async def get_daily_revenue_trend(self, start_date: datetime, end_date: datetime) -> List[Dict[str, Any]]:
        # Simulate daily revenue data
        data = []
        current_date = start_date
        base_revenue = 50000
        
        while current_date <= end_date:
            revenue = base_revenue * (1 + np.random.random() * 0.3)
            data.append({
                "date": current_date.strftime("%Y-%m-%d"),
                "revenue": round(revenue, 2)
            })
            current_date += timedelta(days=1)
        
        return data
    
    async def get_comprehensive_metrics(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        return {"total_revenue": await self.get_total_revenue(start_date, end_date)}
    
    async def generate_insights(self, metrics: Dict[str, Any]) -> List[BusinessInsight]:
        return []
    
    async def create_charts(self, start_date: datetime, end_date: datetime) -> List[Dict[str, Any]]:
        return []
    
    async def create_data_tables(self, start_date: datetime, end_date: datetime) -> List[Dict[str, Any]]:
        return []
    
    async def generate_recommendations(self, metrics: Dict[str, Any]) -> List[str]:
        return ["Optimize pricing strategies", "Expand high-performing product lines"]


class CustomerAnalyzer:
    async def get_conversion_rate(self, start_date: datetime, end_date: datetime) -> float:
        return 2.5 + np.random.random() * 1.0  # 2.5-3.5%
    
    async def get_satisfaction_score(self, start_date: datetime, end_date: datetime) -> float:
        return 4.0 + np.random.random() * 0.8  # 4.0-4.8/5
    
    async def get_acquisition_channels_data(self, start_date: datetime, end_date: datetime) -> List[Dict[str, Any]]:
        return [
            {"channel": "Organic Search", "customers": 1200},
            {"channel": "Social Media", "customers": 800},
            {"channel": "Email Marketing", "customers": 600},
            {"channel": "Direct Traffic", "customers": 400}
        ]


class ProductPerformanceAnalyzer:
    pass


class AgentEffectivenessAnalyzer:
    async def get_overall_effectiveness(self, start_date: datetime, end_date: datetime) -> float:
        return 70 + np.random.random() * 20  # 70-90%


class MarketIntelligenceAnalyzer:
    pass


class PredictiveAnalyzer:
    pass