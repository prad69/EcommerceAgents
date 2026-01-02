from sqlalchemy import Column, Integer, String, Text, Float, DateTime, Boolean, ForeignKey, JSON, Enum
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID
from datetime import datetime
import uuid
import enum

from src.core.database import Base


class DescriptionType(enum.Enum):
    SHORT = "short"           # Brief 1-2 sentence description
    MEDIUM = "medium"         # Standard paragraph description  
    LONG = "long"            # Detailed multi-paragraph description
    BULLETS = "bullets"       # Bullet point format
    SEO = "seo"              # SEO-optimized version
    SOCIAL = "social"        # Social media friendly
    TECHNICAL = "technical"   # Technical specifications focus


class DescriptionStatus(enum.Enum):
    DRAFT = "draft"
    REVIEW = "review" 
    APPROVED = "approved"
    PUBLISHED = "published"
    ARCHIVED = "archived"


class GenerationMethod(enum.Enum):
    TEMPLATE = "template"
    AI_GENERATED = "ai_generated"
    HUMAN_WRITTEN = "human_written"
    HYBRID = "hybrid"


class ProductDescription(Base):
    """
    Product descriptions with multiple formats and versions
    """
    __tablename__ = "product_descriptions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    product_id = Column(UUID(as_uuid=True), ForeignKey("products.id"), nullable=False, index=True)
    
    # Description content
    description_type = Column(Enum(DescriptionType), nullable=False)
    title = Column(String(200), nullable=False)
    content = Column(Text, nullable=False)
    formatted_content = Column(JSON)  # HTML, Markdown, or structured data
    
    # Metadata
    version = Column(Integer, default=1)
    status = Column(Enum(DescriptionStatus), default=DescriptionStatus.DRAFT)
    generation_method = Column(Enum(GenerationMethod), default=GenerationMethod.TEMPLATE)
    
    # SEO attributes
    meta_title = Column(String(60))  # SEO title (max 60 chars)
    meta_description = Column(String(160))  # Meta description (max 160 chars)
    keywords = Column(JSON)  # List of target keywords
    seo_score = Column(Float, default=0.0)  # SEO optimization score (0-1)
    
    # Quality metrics
    quality_score = Column(Float, default=0.0)  # Overall quality score (0-1)
    readability_score = Column(Float, default=0.0)  # Reading ease score
    uniqueness_score = Column(Float, default=0.0)  # Content uniqueness
    conversion_score = Column(Float, default=0.0)  # Conversion effectiveness
    
    # Performance tracking
    view_count = Column(Integer, default=0)
    click_through_rate = Column(Float, default=0.0)
    conversion_rate = Column(Float, default=0.0)
    bounce_rate = Column(Float, default=0.0)
    
    # Generation context
    template_used = Column(String(100))
    generation_prompt = Column(Text)
    ai_model_version = Column(String(50))
    generation_parameters = Column(JSON)
    
    # Review and approval
    created_by = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    reviewed_by = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    approved_by = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    reviewed_at = Column(DateTime, nullable=True)
    approved_at = Column(DateTime, nullable=True)
    published_at = Column(DateTime, nullable=True)
    
    # Relationships
    product = relationship("Product", back_populates="descriptions")
    analytics = relationship("DescriptionAnalytics", back_populates="description", uselist=False)
    a_b_tests = relationship("DescriptionABTest", back_populates="description")


class DescriptionTemplate(Base):
    """
    Templates for generating product descriptions
    """
    __tablename__ = "description_templates"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    
    # Template identification
    name = Column(String(100), nullable=False, unique=True)
    description_type = Column(Enum(DescriptionType), nullable=False)
    category = Column(String(100), index=True)  # Product category this template applies to
    
    # Template content
    template_text = Column(Text, nullable=False)  # Template with placeholders
    required_fields = Column(JSON)  # List of required product fields
    optional_fields = Column(JSON)  # List of optional product fields
    
    # Template settings
    min_word_count = Column(Integer, default=10)
    max_word_count = Column(Integer, default=500)
    tone = Column(String(50), default="professional")  # professional, casual, luxury, technical
    target_audience = Column(String(100))  # Target customer segment
    
    # SEO settings
    keyword_density_target = Column(Float, default=0.02)  # Target keyword density
    include_specifications = Column(Boolean, default=True)
    include_benefits = Column(Boolean, default=True)
    include_features = Column(Boolean, default=True)
    
    # Performance metrics
    usage_count = Column(Integer, default=0)
    average_quality_score = Column(Float, default=0.0)
    average_conversion_rate = Column(Float, default=0.0)
    
    # Template versioning
    version = Column(Integer, default=1)
    is_active = Column(Boolean, default=True)
    
    # Metadata
    created_by = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class DescriptionAnalytics(Base):
    """
    Analytics and performance metrics for product descriptions
    """
    __tablename__ = "description_analytics"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    description_id = Column(UUID(as_uuid=True), ForeignKey("product_descriptions.id"), nullable=False, unique=True)
    
    # Engagement metrics
    total_views = Column(Integer, default=0)
    unique_views = Column(Integer, default=0)
    time_on_page = Column(Float, default=0.0)  # Average time spent reading
    scroll_depth = Column(Float, default=0.0)  # How far users scroll
    
    # Conversion metrics
    add_to_cart_rate = Column(Float, default=0.0)
    purchase_conversion_rate = Column(Float, default=0.0)
    revenue_per_view = Column(Float, default=0.0)
    
    # SEO metrics
    organic_traffic = Column(Integer, default=0)
    search_impressions = Column(Integer, default=0)
    search_clicks = Column(Integer, default=0)
    average_position = Column(Float, default=0.0)
    
    # Quality indicators
    user_feedback_score = Column(Float, default=0.0)  # User ratings/feedback
    return_visitor_rate = Column(Float, default=0.0)
    share_rate = Column(Float, default=0.0)
    
    # Comparative metrics
    category_performance_rank = Column(Integer, default=0)
    improvement_vs_previous = Column(Float, default=0.0)
    
    # Timestamps
    measurement_start_date = Column(DateTime, default=datetime.utcnow)
    last_updated = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    description = relationship("ProductDescription", back_populates="analytics")


class DescriptionABTest(Base):
    """
    A/B testing for product descriptions
    """
    __tablename__ = "description_ab_tests"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    
    # Test identification
    test_name = Column(String(200), nullable=False)
    product_id = Column(UUID(as_uuid=True), ForeignKey("products.id"), nullable=False, index=True)
    description_id = Column(UUID(as_uuid=True), ForeignKey("product_descriptions.id"), nullable=False)
    
    # Test configuration
    variant_name = Column(String(100), nullable=False)  # A, B, C, etc.
    traffic_allocation = Column(Float, default=0.5)  # Percentage of traffic for this variant
    
    # Test status
    status = Column(String(20), default="draft")  # draft, running, paused, completed
    start_date = Column(DateTime, nullable=True)
    end_date = Column(DateTime, nullable=True)
    planned_duration_days = Column(Integer, default=14)
    
    # Test results
    participant_count = Column(Integer, default=0)
    conversion_rate = Column(Float, default=0.0)
    confidence_level = Column(Float, default=0.0)  # Statistical confidence
    is_winner = Column(Boolean, default=False)
    
    # Test metrics
    views = Column(Integer, default=0)
    clicks = Column(Integer, default=0)
    conversions = Column(Integer, default=0)
    revenue = Column(Float, default=0.0)
    
    # Statistical analysis
    statistical_significance = Column(Float, default=0.0)
    effect_size = Column(Float, default=0.0)  # How much better/worse than control
    
    # Metadata
    created_by = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    description = relationship("ProductDescription", back_populates="a_b_tests")


class SEOKeyword(Base):
    """
    SEO keywords for product descriptions
    """
    __tablename__ = "seo_keywords"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    
    # Keyword data
    keyword = Column(String(200), nullable=False, index=True)
    category = Column(String(100), index=True)
    
    # Search metrics
    search_volume = Column(Integer, default=0)
    competition_level = Column(String(20), default="medium")  # low, medium, high
    cost_per_click = Column(Float, default=0.0)
    difficulty_score = Column(Float, default=0.0)  # SEO difficulty (0-100)
    
    # Performance tracking
    current_rank = Column(Integer, default=0)
    target_rank = Column(Integer, default=1)
    click_through_rate = Column(Float, default=0.0)
    
    # Keyword context
    intent = Column(String(50))  # informational, commercial, transactional
    seasonal_trend = Column(JSON)  # Monthly trend data
    related_keywords = Column(JSON)  # List of related keywords
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_rank_check = Column(DateTime, nullable=True)


class CompetitorAnalysis(Base):
    """
    Competitive analysis for product descriptions
    """
    __tablename__ = "competitor_analysis"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Competitor information
    competitor_name = Column(String(100), nullable=False)
    competitor_url = Column(String(500))
    product_category = Column(String(100), index=True)
    
    # Analysis data
    description_length = Column(Integer, default=0)
    keyword_usage = Column(JSON)  # Keywords found in their descriptions
    features_mentioned = Column(JSON)  # Features they emphasize
    tone_analysis = Column(JSON)  # Tone and style analysis
    
    # Performance indicators
    estimated_traffic = Column(Integer, default=0)
    social_mentions = Column(Integer, default=0)
    review_count = Column(Integer, default=0)
    average_rating = Column(Float, default=0.0)
    
    # Content analysis
    unique_selling_points = Column(JSON)  # Their unique positioning
    call_to_action_analysis = Column(JSON)  # CTA analysis
    media_usage = Column(JSON)  # Images, videos used
    
    # Tracking
    analysis_date = Column(DateTime, default=datetime.utcnow)
    confidence_score = Column(Float, default=0.0)  # Analysis confidence
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class ContentGeneration(Base):
    """
    Track content generation requests and results
    """
    __tablename__ = "content_generation"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    
    # Generation request
    product_id = Column(UUID(as_uuid=True), ForeignKey("products.id"), nullable=False, index=True)
    description_type = Column(Enum(DescriptionType), nullable=False)
    generation_method = Column(Enum(GenerationMethod), nullable=False)
    
    # Input parameters
    template_id = Column(UUID(as_uuid=True), ForeignKey("description_templates.id"), nullable=True)
    custom_prompt = Column(Text)
    target_keywords = Column(JSON)  # Keywords to include
    target_word_count = Column(Integer)
    tone_requirements = Column(JSON)  # Tone specifications
    
    # Generation results
    generated_content = Column(Text)
    alternative_versions = Column(JSON)  # Multiple generated versions
    quality_score = Column(Float, default=0.0)
    generation_time_ms = Column(Integer, default=0)
    
    # AI model information
    model_name = Column(String(100))
    model_version = Column(String(50))
    generation_parameters = Column(JSON)  # Model-specific parameters
    cost = Column(Float, default=0.0)  # Generation cost
    
    # Status and feedback
    status = Column(String(20), default="completed")  # completed, failed, processing
    error_message = Column(Text)
    user_rating = Column(Integer)  # 1-5 rating from user
    user_feedback = Column(Text)
    
    # Metadata
    requested_by = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)


class BrandGuideline(Base):
    """
    Brand guidelines for consistent content generation
    """
    __tablename__ = "brand_guidelines"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Brand information
    brand_name = Column(String(100), nullable=False, unique=True)
    brand_description = Column(Text)
    
    # Tone and voice
    tone_of_voice = Column(JSON)  # Detailed tone specifications
    writing_style = Column(JSON)  # Style guidelines
    prohibited_words = Column(JSON)  # Words to avoid
    preferred_words = Column(JSON)  # Preferred terminology
    
    # Content guidelines
    min_description_length = Column(Integer, default=50)
    max_description_length = Column(Integer, default=500)
    required_sections = Column(JSON)  # Required content sections
    optional_sections = Column(JSON)  # Optional content sections
    
    # SEO preferences
    keyword_integration_style = Column(String(50), default="natural")  # natural, aggressive, minimal
    meta_description_template = Column(String(200))
    title_format_template = Column(String(100))
    
    # Legal and compliance
    mandatory_disclaimers = Column(JSON)  # Legal disclaimers to include
    compliance_requirements = Column(JSON)  # Regulatory requirements
    
    # Templates and examples
    approved_templates = Column(JSON)  # List of approved template IDs
    example_descriptions = Column(JSON)  # Example good descriptions
    
    # Metadata
    is_active = Column(Boolean, default=True)
    created_by = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


# Update the Product model to include description relationship
from src.models.product import Product
Product.descriptions = relationship("ProductDescription", back_populates="product", cascade="all, delete-orphan")