from sqlalchemy import Column, Integer, String, Text, Float, DateTime, Boolean, ForeignKey, JSON, Enum
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID
from datetime import datetime
import uuid
import enum

from src.core.database import Base


class ConversationStatus(enum.Enum):
    ACTIVE = "active"
    WAITING = "waiting"
    RESOLVED = "resolved"
    ESCALATED = "escalated"
    CLOSED = "closed"


class MessageType(enum.Enum):
    USER = "user"
    BOT = "bot"
    SYSTEM = "system"
    HUMAN_AGENT = "human_agent"


class IntentType(enum.Enum):
    # Product-related intents
    PRODUCT_SEARCH = "product_search"
    PRODUCT_INFO = "product_info"
    PRODUCT_COMPARE = "product_compare"
    PRODUCT_RECOMMEND = "product_recommend"
    PRODUCT_AVAILABILITY = "product_availability"
    
    # Order-related intents
    ORDER_STATUS = "order_status"
    ORDER_TRACK = "order_track"
    ORDER_CANCEL = "order_cancel"
    ORDER_RETURN = "order_return"
    ORDER_EXCHANGE = "order_exchange"
    
    # Support intents
    TECHNICAL_SUPPORT = "technical_support"
    SHIPPING_INFO = "shipping_info"
    PAYMENT_HELP = "payment_help"
    ACCOUNT_HELP = "account_help"
    COMPLAINT = "complaint"
    
    # General intents
    GREETING = "greeting"
    FAREWELL = "farewell"
    SMALL_TALK = "small_talk"
    ESCALATE_HUMAN = "escalate_human"
    UNKNOWN = "unknown"


class Conversation(Base):
    """
    Stores conversation sessions between users and the chatbot
    """
    __tablename__ = "conversations"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    session_id = Column(String(200), nullable=False, unique=True, index=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True, index=True)
    
    # Conversation metadata
    status = Column(Enum(ConversationStatus), default=ConversationStatus.ACTIVE, index=True)
    channel = Column(String(50), default="web")  # web, mobile, api
    language = Column(String(10), default="en")
    
    # Context and state
    current_intent = Column(Enum(IntentType), nullable=True)
    context = Column(JSON)  # Stores conversation context and entities
    user_agent = Column(String(500))
    ip_address = Column(String(50))
    
    # Resolution metrics
    resolved_automatically = Column(Boolean, default=False)
    escalated_to_human = Column(Boolean, default=False)
    human_agent_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    satisfaction_rating = Column(Integer, nullable=True)  # 1-5 rating
    
    # Timestamps
    started_at = Column(DateTime, default=datetime.utcnow, index=True)
    last_activity_at = Column(DateTime, default=datetime.utcnow)
    resolved_at = Column(DateTime, nullable=True)
    
    # Relationships
    messages = relationship("ConversationMessage", back_populates="conversation", cascade="all, delete-orphan")
    user = relationship("User", foreign_keys=[user_id])
    human_agent = relationship("User", foreign_keys=[human_agent_id])
    analytics = relationship("ConversationAnalytics", back_populates="conversation", uselist=False)


class ConversationMessage(Base):
    """
    Individual messages within a conversation
    """
    __tablename__ = "conversation_messages"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    conversation_id = Column(UUID(as_uuid=True), ForeignKey("conversations.id"), nullable=False, index=True)
    
    # Message content
    message_type = Column(Enum(MessageType), nullable=False)
    content = Column(Text, nullable=False)
    formatted_content = Column(JSON)  # Rich formatted content (buttons, cards, etc.)
    
    # Intent and entity analysis
    detected_intent = Column(Enum(IntentType), nullable=True)
    intent_confidence = Column(Float, nullable=True)
    extracted_entities = Column(JSON)  # Named entities and values
    
    # Response metadata
    response_time_ms = Column(Integer, nullable=True)  # Time to generate response
    response_source = Column(String(50))  # ai_model, knowledge_base, human, etc.
    model_version = Column(String(50))
    
    # User feedback
    helpful = Column(Boolean, nullable=True)
    feedback_text = Column(Text, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    
    # Relationships
    conversation = relationship("Conversation", back_populates="messages")


class ConversationAnalytics(Base):
    """
    Analytics and metrics for each conversation
    """
    __tablename__ = "conversation_analytics"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    conversation_id = Column(UUID(as_uuid=True), ForeignKey("conversations.id"), nullable=False, unique=True)
    
    # Conversation metrics
    total_messages = Column(Integer, default=0)
    user_messages = Column(Integer, default=0)
    bot_messages = Column(Integer, default=0)
    
    # Timing metrics
    duration_seconds = Column(Integer, nullable=True)
    avg_response_time_ms = Column(Float, nullable=True)
    max_response_time_ms = Column(Integer, nullable=True)
    
    # Quality metrics
    resolution_score = Column(Float, nullable=True)  # 0-1 score
    intent_accuracy = Column(Float, nullable=True)  # % of correctly identified intents
    entity_extraction_accuracy = Column(Float, nullable=True)
    
    # Interaction patterns
    context_switches = Column(Integer, default=0)  # Number of topic changes
    clarification_requests = Column(Integer, default=0)
    fallback_responses = Column(Integer, default=0)
    
    # Outcome metrics
    goals_achieved = Column(JSON)  # List of user goals that were achieved
    products_mentioned = Column(JSON)  # Products discussed during conversation
    
    # Timestamps
    analyzed_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    conversation = relationship("Conversation", back_populates="analytics")


class IntentTrainingData(Base):
    """
    Training data for intent classification
    """
    __tablename__ = "intent_training_data"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Training sample
    text = Column(Text, nullable=False)
    intent = Column(Enum(IntentType), nullable=False)
    confidence = Column(Float, default=1.0)
    
    # Context and metadata
    entities = Column(JSON)  # Associated entities for this sample
    context_required = Column(JSON)  # Context needed for this intent
    response_template = Column(Text)  # Template response for this intent
    
    # Quality control
    verified = Column(Boolean, default=False)
    created_by = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    language = Column(String(10), default="en")
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class KnowledgeBase(Base):
    """
    Knowledge base entries for RAG-based responses
    """
    __tablename__ = "knowledge_base"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Content
    title = Column(String(500), nullable=False)
    content = Column(Text, nullable=False)
    summary = Column(Text)
    
    # Categorization
    category = Column(String(100), index=True)
    tags = Column(JSON)  # List of relevant tags
    applicable_intents = Column(JSON)  # Intents this entry can help with
    
    # Context and usage
    context_required = Column(JSON)  # Context needed to use this entry
    priority = Column(Integer, default=0)  # Higher priority entries are preferred
    
    # Metadata
    source_url = Column(String(1000))
    last_updated = Column(DateTime, default=datetime.utcnow)
    verified = Column(Boolean, default=False)
    usage_count = Column(Integer, default=0)
    effectiveness_score = Column(Float, default=0.5)  # How helpful this entry is
    
    # Relationships
    embedding_id = Column(String(200))  # Vector embedding for semantic search


class ChatbotConfiguration(Base):
    """
    Configuration settings for the chatbot
    """
    __tablename__ = "chatbot_configuration"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # General settings
    bot_name = Column(String(100), default="EcommerceBot")
    welcome_message = Column(Text, default="Hello! How can I help you today?")
    fallback_message = Column(Text, default="I'm not sure I understand. Could you please rephrase that?")
    escalation_message = Column(Text, default="Let me connect you with a human agent who can better assist you.")
    
    # Behavior settings
    confidence_threshold = Column(Float, default=0.7)  # Minimum confidence for intent
    max_context_length = Column(Integer, default=10)  # Messages to keep in context
    response_delay_ms = Column(Integer, default=1000)  # Artificial delay for natural feel
    
    # Escalation rules
    max_fallback_attempts = Column(Integer, default=3)
    auto_escalate_keywords = Column(JSON)  # Keywords that trigger escalation
    business_hours_start = Column(String(10), default="09:00")
    business_hours_end = Column(String(10), default="17:00")
    
    # Feature flags
    enable_small_talk = Column(Boolean, default=True)
    enable_product_recommendations = Column(Boolean, default=True)
    enable_order_tracking = Column(Boolean, default=True)
    enable_sentiment_analysis = Column(Boolean, default=True)
    
    # Model settings
    primary_model = Column(String(100), default="openai")  # openai, transformers, rule_based
    backup_model = Column(String(100), default="rule_based")
    max_response_length = Column(Integer, default=500)
    
    # Analytics
    collect_feedback = Column(Boolean, default=True)
    log_conversations = Column(Boolean, default=True)
    anonymize_data = Column(Boolean, default=False)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class ConversationFeedback(Base):
    """
    User feedback on chatbot conversations
    """
    __tablename__ = "conversation_feedback"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    conversation_id = Column(UUID(as_uuid=True), ForeignKey("conversations.id"), nullable=False)
    message_id = Column(UUID(as_uuid=True), ForeignKey("conversation_messages.id"), nullable=True)
    
    # Feedback content
    rating = Column(Integer, nullable=True)  # 1-5 rating
    feedback_text = Column(Text)
    feedback_type = Column(String(50))  # helpful, unhelpful, suggestion, complaint
    
    # Metadata
    submitted_by = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    submitted_at = Column(DateTime, default=datetime.utcnow)
    
    # Follow-up
    reviewed = Column(Boolean, default=False)
    action_taken = Column(Text)
    reviewed_by = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    reviewed_at = Column(DateTime, nullable=True)