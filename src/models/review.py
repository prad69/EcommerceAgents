from sqlalchemy import Column, Integer, String, Text, Float, DateTime, Boolean, ForeignKey, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID
from datetime import datetime
import uuid

from src.core.database import Base


class Review(Base):
    """
    Customer review model for storing product reviews and ratings
    """
    __tablename__ = "reviews"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    product_id = Column(UUID(as_uuid=True), ForeignKey("products.id"), nullable=False, index=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True, index=True)
    
    # Review content
    title = Column(String(500))
    content = Column(Text, nullable=False)
    rating = Column(Float, nullable=False)  # 1-5 star rating
    
    # Review metadata
    reviewer_name = Column(String(200))
    verified_purchase = Column(Boolean, default=False)
    helpful_votes = Column(Integer, default=0)
    total_votes = Column(Integer, default=0)
    
    # External source data
    external_id = Column(String(200), unique=True, index=True)  # Amazon review ID, etc.
    data_source = Column(String(100), default="manual")  # amazon, manual, etc.
    review_date = Column(DateTime, nullable=False)
    
    # Processed data
    is_processed = Column(Boolean, default=False)
    language = Column(String(10), default="en")
    word_count = Column(Integer)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    product = relationship("Product", back_populates="reviews")
    analysis = relationship("ReviewAnalysis", back_populates="review", uselist=False)
    sentiment_history = relationship("SentimentHistory", back_populates="review")


class ReviewAnalysis(Base):
    """
    Stores the AI analysis results for each review
    """
    __tablename__ = "review_analysis"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    review_id = Column(UUID(as_uuid=True), ForeignKey("reviews.id"), nullable=False, unique=True)
    
    # Sentiment analysis
    overall_sentiment = Column(String(20))  # positive, negative, neutral
    sentiment_score = Column(Float)  # -1 to 1
    sentiment_confidence = Column(Float)  # 0 to 1
    
    # Aspect-based sentiment
    aspects = Column(JSON)  # {"quality": {"sentiment": "positive", "score": 0.8}, ...}
    
    # Content analysis
    themes = Column(JSON)  # ["durability", "price", "design", ...]
    keywords = Column(JSON)  # {"positive": [...], "negative": [...]}
    
    # Extracted insights
    pros = Column(JSON)  # ["good quality", "fast shipping", ...]
    cons = Column(JSON)  # ["too expensive", "poor packaging", ...]
    
    # Quality metrics
    readability_score = Column(Float)
    authenticity_score = Column(Float)  # Fake review detection
    helpfulness_predicted = Column(Float)
    
    # Processing metadata
    model_version = Column(String(50))
    processing_time = Column(Float)  # seconds
    processed_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    review = relationship("Review", back_populates="analysis")


class ReviewSummary(Base):
    """
    Aggregated review summaries for products
    """
    __tablename__ = "review_summaries"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    product_id = Column(UUID(as_uuid=True), ForeignKey("products.id"), nullable=False, unique=True)
    
    # Summary statistics
    total_reviews = Column(Integer, default=0)
    average_rating = Column(Float)
    rating_distribution = Column(JSON)  # {"1": 5, "2": 10, "3": 20, "4": 30, "5": 35}
    
    # Sentiment summary
    sentiment_distribution = Column(JSON)  # {"positive": 60, "neutral": 30, "negative": 10}
    overall_sentiment = Column(String(20))
    sentiment_trend = Column(String(20))  # improving, declining, stable
    
    # Content summaries
    summary_text = Column(Text)  # AI-generated summary
    top_themes = Column(JSON)  # Most mentioned themes
    common_pros = Column(JSON)  # Most mentioned pros
    common_cons = Column(JSON)  # Most mentioned cons
    
    # Aspect ratings
    aspect_ratings = Column(JSON)  # {"quality": 4.2, "price": 3.8, "shipping": 4.5, ...}
    
    # Quality indicators
    verified_purchase_ratio = Column(Float)
    average_helpfulness = Column(Float)
    fake_review_ratio = Column(Float)
    
    # Timestamps
    last_updated = Column(DateTime, default=datetime.utcnow)
    last_review_date = Column(DateTime)
    
    # Relationships
    product = relationship("Product", back_populates="review_summary")


class SentimentHistory(Base):
    """
    Tracks sentiment changes over time
    """
    __tablename__ = "sentiment_history"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    product_id = Column(UUID(as_uuid=True), ForeignKey("products.id"), nullable=False, index=True)
    review_id = Column(UUID(as_uuid=True), ForeignKey("reviews.id"), nullable=True)
    
    # Sentiment data
    sentiment = Column(String(20), nullable=False)
    sentiment_score = Column(Float, nullable=False)
    rating = Column(Float)
    
    # Time period
    date = Column(DateTime, nullable=False, index=True)
    week_start = Column(DateTime, index=True)
    month_start = Column(DateTime, index=True)
    
    # Relationships
    review = relationship("Review", back_populates="sentiment_history")


class ReviewAlert(Base):
    """
    Alert system for significant sentiment changes
    """
    __tablename__ = "review_alerts"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    product_id = Column(UUID(as_uuid=True), ForeignKey("products.id"), nullable=False, index=True)
    
    # Alert details
    alert_type = Column(String(50), nullable=False)  # sentiment_drop, fake_reviews, etc.
    severity = Column(String(20), default="medium")  # low, medium, high, critical
    title = Column(String(200), nullable=False)
    description = Column(Text)
    
    # Alert data
    current_value = Column(Float)
    previous_value = Column(Float)
    threshold = Column(Float)
    
    # Status
    is_active = Column(Boolean, default=True)
    is_acknowledged = Column(Boolean, default=False)
    acknowledged_by = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    acknowledged_at = Column(DateTime)
    
    # Timestamps
    triggered_at = Column(DateTime, default=datetime.utcnow)
    resolved_at = Column(DateTime)
    
    # Relationships
    product = relationship("Product")