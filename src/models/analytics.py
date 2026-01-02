from sqlalchemy import Column, Integer, String, Boolean, DateTime, Text, JSON, Float, ForeignKey
from sqlalchemy.sql import func
from sqlalchemy.dialects.postgresql import UUID, ARRAY
from sqlalchemy.orm import relationship
import uuid

from src.core.database import Base


class UserInteraction(Base):
    __tablename__ = "user_interactions"
    __table_args__ = {'schema': 'analytics'}
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), nullable=True, index=True)  # Can be null for anonymous
    session_id = Column(String, nullable=False, index=True)
    
    # Interaction details
    interaction_type = Column(String, nullable=False, index=True)  # view, click, cart, purchase
    product_id = Column(UUID(as_uuid=True), nullable=True, index=True)
    category_id = Column(UUID(as_uuid=True), nullable=True, index=True)
    
    # Context
    page_url = Column(String, nullable=True)
    referrer_url = Column(String, nullable=True)
    user_agent = Column(Text, nullable=True)
    ip_address = Column(String, nullable=True)
    
    # Additional data
    metadata = Column(JSON, nullable=True, default={})
    duration = Column(Float, nullable=True)  # Time spent in seconds
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    def __repr__(self):
        return f"<UserInteraction(user_id='{self.user_id}', type='{self.interaction_type}')>"


class SearchQuery(Base):
    __tablename__ = "search_queries"
    __table_args__ = {'schema': 'analytics'}
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), nullable=True, index=True)
    session_id = Column(String, nullable=False, index=True)
    
    # Search details
    query_text = Column(Text, nullable=False, index=True)
    query_vector = Column(String, nullable=True)  # Reference to vector representation
    filters_applied = Column(JSON, nullable=True, default={})
    
    # Results and interactions
    results_count = Column(Integer, default=0)
    clicked_results = Column(ARRAY(String), nullable=True)  # Product IDs clicked
    conversion = Column(Boolean, default=False)  # Did search lead to purchase
    
    # Performance
    response_time = Column(Float, nullable=True)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    def __repr__(self):
        return f"<SearchQuery(query='{self.query_text[:50]}...', results={self.results_count})>"


class RecommendationEvent(Base):
    __tablename__ = "recommendation_events"
    __table_args__ = {'schema': 'analytics'}
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), nullable=True, index=True)
    session_id = Column(String, nullable=False, index=True)
    
    # Recommendation details
    recommendation_type = Column(String, nullable=False)  # content_based, collaborative, hybrid
    context = Column(String, nullable=False)  # homepage, product_page, cart, etc.
    algorithm_version = Column(String, nullable=True)
    
    # Products
    recommended_products = Column(ARRAY(String), nullable=False)  # Product IDs
    clicked_products = Column(ARRAY(String), nullable=True)
    purchased_products = Column(ARRAY(String), nullable=True)
    
    # Performance metrics
    click_through_rate = Column(Float, nullable=True)
    conversion_rate = Column(Float, nullable=True)
    response_time = Column(Float, nullable=True)
    
    # A/B testing
    experiment_id = Column(String, nullable=True)
    variant = Column(String, nullable=True)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    def __repr__(self):
        return f"<RecommendationEvent(type='{self.recommendation_type}', context='{self.context}')>"


class AgentMetrics(Base):
    __tablename__ = "agent_metrics"
    __table_args__ = {'schema': 'analytics'}
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Agent identification
    agent_type = Column(String, nullable=False, index=True)  # recommendation, review, chatbot, etc.
    agent_version = Column(String, nullable=True)
    
    # Performance metrics
    request_count = Column(Integer, default=0)
    success_count = Column(Integer, default=0)
    error_count = Column(Integer, default=0)
    avg_response_time = Column(Float, nullable=True)
    
    # Quality metrics (specific to agent type)
    quality_metrics = Column(JSON, nullable=True, default={})
    
    # Resource usage
    cpu_usage = Column(Float, nullable=True)
    memory_usage = Column(Float, nullable=True)
    
    # Time period
    period_start = Column(DateTime(timezone=True), nullable=False)
    period_end = Column(DateTime(timezone=True), nullable=False)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    def __repr__(self):
        return f"<AgentMetrics(agent='{self.agent_type}', period='{self.period_start}')>"


class BusinessMetrics(Base):
    __tablename__ = "business_metrics"
    __table_args__ = {'schema': 'analytics'}
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Metric details
    metric_name = Column(String, nullable=False, index=True)
    metric_value = Column(Float, nullable=False)
    metric_unit = Column(String, nullable=True)
    
    # Dimensions
    dimensions = Column(JSON, nullable=True, default={})  # category, region, etc.
    
    # Time period
    date = Column(DateTime(timezone=True), nullable=False, index=True)
    granularity = Column(String, nullable=False)  # hour, day, week, month
    
    # Metadata
    calculation_method = Column(String, nullable=True)
    data_sources = Column(ARRAY(String), nullable=True)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    def __repr__(self):
        return f"<BusinessMetrics(metric='{self.metric_name}', value={self.metric_value})>"