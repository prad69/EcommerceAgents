import asyncio
from typing import Dict, List, Any, Optional, Tuple
from loguru import logger
import json
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum

from src.core.database import get_db, redis_client
from src.models.conversation import Conversation, ConversationMessage, IntentType, MessageType, ConversationStatus
from src.models.product import Product


class ConversationFlow(Enum):
    """Define conversation flow states"""
    GREETING = "greeting"
    INTENT_DETECTION = "intent_detection"
    ENTITY_COLLECTION = "entity_collection"
    INFORMATION_GATHERING = "information_gathering"
    PROCESSING = "processing"
    RESPONSE_GENERATION = "response_generation"
    CONFIRMATION = "confirmation"
    RESOLUTION = "resolution"
    ESCALATION = "escalation"
    FAREWELL = "farewell"


@dataclass
class ConversationContext:
    """Conversation context data structure"""
    session_id: str
    user_id: Optional[str]
    current_flow: ConversationFlow
    current_intent: Optional[IntentType]
    entities: Dict[str, Any]
    conversation_history: List[Dict[str, Any]]
    user_preferences: Dict[str, Any]
    pending_actions: List[str]
    confidence_scores: Dict[str, float]
    last_activity: datetime
    escalation_requested: bool
    resolution_attempts: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            **asdict(self),
            "last_activity": self.last_activity.isoformat(),
            "current_flow": self.current_flow.value if self.current_flow else None,
            "current_intent": self.current_intent.value if self.current_intent else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConversationContext':
        """Create from dictionary"""
        data = data.copy()
        data["last_activity"] = datetime.fromisoformat(data["last_activity"])
        data["current_flow"] = ConversationFlow(data["current_flow"]) if data.get("current_flow") else ConversationFlow.GREETING
        data["current_intent"] = IntentType(data["current_intent"]) if data.get("current_intent") else None
        return cls(**data)


class ConversationStateManager:
    """
    Manages conversation state, context, and flow
    """
    
    def __init__(self):
        self.context_cache = {}  # In-memory cache for active conversations
        self.flow_handlers = self._setup_flow_handlers()
        self.intent_requirements = self._setup_intent_requirements()
    
    def _setup_flow_handlers(self) -> Dict[ConversationFlow, callable]:
        """Setup handlers for each conversation flow state"""
        return {
            ConversationFlow.GREETING: self._handle_greeting,
            ConversationFlow.INTENT_DETECTION: self._handle_intent_detection,
            ConversationFlow.ENTITY_COLLECTION: self._handle_entity_collection,
            ConversationFlow.INFORMATION_GATHERING: self._handle_information_gathering,
            ConversationFlow.PROCESSING: self._handle_processing,
            ConversationFlow.RESPONSE_GENERATION: self._handle_response_generation,
            ConversationFlow.CONFIRMATION: self._handle_confirmation,
            ConversationFlow.RESOLUTION: self._handle_resolution,
            ConversationFlow.ESCALATION: self._handle_escalation,
            ConversationFlow.FAREWELL: self._handle_farewell
        }
    
    def _setup_intent_requirements(self) -> Dict[IntentType, Dict[str, Any]]:
        """Define required entities and context for each intent"""
        return {
            IntentType.PRODUCT_SEARCH: {
                "required_entities": ["product_name", "category"],
                "optional_entities": ["price_range", "brand", "color", "size"],
                "context_needed": ["search_preferences"],
                "max_attempts": 3
            },
            IntentType.PRODUCT_INFO: {
                "required_entities": ["product_id", "product_name"],
                "optional_entities": ["specific_question"],
                "context_needed": [],
                "max_attempts": 2
            },
            IntentType.ORDER_STATUS: {
                "required_entities": ["order_number"],
                "optional_entities": ["email", "phone"],
                "context_needed": ["user_verification"],
                "max_attempts": 3
            },
            IntentType.ORDER_TRACK: {
                "required_entities": ["tracking_number", "order_number"],
                "optional_entities": ["email"],
                "context_needed": [],
                "max_attempts": 2
            },
            IntentType.TECHNICAL_SUPPORT: {
                "required_entities": ["problem_description"],
                "optional_entities": ["product_id", "error_message"],
                "context_needed": ["problem_details"],
                "max_attempts": 4
            },
            IntentType.ORDER_RETURN: {
                "required_entities": ["order_number", "return_reason"],
                "optional_entities": ["product_id"],
                "context_needed": ["return_policy"],
                "max_attempts": 3
            }
        }
    
    async def get_conversation_context(self, session_id: str) -> ConversationContext:
        """
        Get or create conversation context for a session
        """
        # Try cache first
        if session_id in self.context_cache:
            context = self.context_cache[session_id]
            # Update last activity
            context.last_activity = datetime.utcnow()
            return context
        
        # Try Redis cache
        try:
            cached_data = redis_client.get(f"conversation_context:{session_id}")
            if cached_data:
                context_data = json.loads(cached_data)
                context = ConversationContext.from_dict(context_data)
                self.context_cache[session_id] = context
                context.last_activity = datetime.utcnow()
                return context
        except Exception as e:
            logger.warning(f"Failed to get cached context: {e}")
        
        # Get from database
        db = next(get_db())
        try:
            conversation = db.query(Conversation).filter(
                Conversation.session_id == session_id
            ).first()
            
            if conversation:
                # Load context from database
                context = ConversationContext(
                    session_id=session_id,
                    user_id=str(conversation.user_id) if conversation.user_id else None,
                    current_flow=ConversationFlow.INTENT_DETECTION,
                    current_intent=conversation.current_intent,
                    entities=conversation.context or {},
                    conversation_history=[],
                    user_preferences={},
                    pending_actions=[],
                    confidence_scores={},
                    last_activity=conversation.last_activity_at or datetime.utcnow(),
                    escalation_requested=conversation.escalated_to_human,
                    resolution_attempts=0
                )
                
                # Load recent conversation history
                recent_messages = db.query(ConversationMessage).filter(
                    ConversationMessage.conversation_id == conversation.id
                ).order_by(ConversationMessage.created_at.desc()).limit(20).all()
                
                context.conversation_history = [
                    {
                        "id": str(msg.id),
                        "content": msg.content,
                        "message_type": msg.message_type.value,
                        "intent": msg.detected_intent.value if msg.detected_intent else None,
                        "entities": msg.extracted_entities or {},
                        "created_at": msg.created_at.isoformat()
                    }
                    for msg in reversed(recent_messages)
                ]
                
            else:
                # Create new context
                context = ConversationContext(
                    session_id=session_id,
                    user_id=None,
                    current_flow=ConversationFlow.GREETING,
                    current_intent=None,
                    entities={},
                    conversation_history=[],
                    user_preferences={},
                    pending_actions=[],
                    confidence_scores={},
                    last_activity=datetime.utcnow(),
                    escalation_requested=False,
                    resolution_attempts=0
                )
            
            # Cache the context
            self.context_cache[session_id] = context
            await self._cache_context(context)
            
            return context
            
        finally:
            db.close()
    
    async def update_conversation_context(
        self,
        session_id: str,
        message_content: str,
        message_type: MessageType,
        intent: Optional[IntentType] = None,
        entities: Optional[Dict[str, Any]] = None,
        confidence_scores: Optional[Dict[str, float]] = None
    ) -> ConversationContext:
        """
        Update conversation context with new message and extracted data
        """
        context = await self.get_conversation_context(session_id)
        
        # Add message to history
        message_data = {
            "content": message_content,
            "message_type": message_type.value,
            "intent": intent.value if intent else None,
            "entities": entities or {},
            "confidence_scores": confidence_scores or {},
            "created_at": datetime.utcnow().isoformat()
        }
        
        context.conversation_history.append(message_data)
        
        # Keep only recent messages
        if len(context.conversation_history) > 50:
            context.conversation_history = context.conversation_history[-50:]
        
        # Update current intent and entities
        if intent:
            context.current_intent = intent
        
        if entities:
            # Merge entities, keeping most recent values
            for entity_type, values in entities.items():
                if isinstance(values, list) and values:
                    context.entities[entity_type] = values[-1]  # Keep most recent
                elif values:
                    context.entities[entity_type] = values
        
        if confidence_scores:
            context.confidence_scores.update(confidence_scores)
        
        context.last_activity = datetime.utcnow()
        
        # Determine next flow state
        await self._update_conversation_flow(context)
        
        # Cache updated context
        await self._cache_context(context)
        
        return context
    
    async def _update_conversation_flow(self, context: ConversationContext):
        """
        Determine and update the next conversation flow state
        """
        current_flow = context.current_flow
        current_intent = context.current_intent
        
        # Greeting flow
        if current_flow == ConversationFlow.GREETING:
            if current_intent and current_intent != IntentType.GREETING:
                context.current_flow = ConversationFlow.INTENT_DETECTION
            return
        
        # Intent detection flow
        if current_flow == ConversationFlow.INTENT_DETECTION:
            if current_intent and current_intent != IntentType.UNKNOWN:
                # Check if we need to collect entities
                if await self._needs_entity_collection(context):
                    context.current_flow = ConversationFlow.ENTITY_COLLECTION
                else:
                    context.current_flow = ConversationFlow.PROCESSING
            return
        
        # Entity collection flow
        if current_flow == ConversationFlow.ENTITY_COLLECTION:
            if await self._has_required_entities(context):
                context.current_flow = ConversationFlow.PROCESSING
            elif context.resolution_attempts >= 3:
                context.current_flow = ConversationFlow.ESCALATION
            return
        
        # Processing flow
        if current_flow == ConversationFlow.PROCESSING:
            context.current_flow = ConversationFlow.RESPONSE_GENERATION
            return
        
        # Response generation flow
        if current_flow == ConversationFlow.RESPONSE_GENERATION:
            if await self._needs_confirmation(context):
                context.current_flow = ConversationFlow.CONFIRMATION
            else:
                context.current_flow = ConversationFlow.RESOLUTION
            return
        
        # Confirmation flow
        if current_flow == ConversationFlow.CONFIRMATION:
            last_message = context.conversation_history[-1] if context.conversation_history else {}
            if self._is_confirmation(last_message.get("content", "")):
                context.current_flow = ConversationFlow.RESOLUTION
            else:
                context.current_flow = ConversationFlow.ENTITY_COLLECTION
            return
        
        # Check for escalation request
        if current_intent == IntentType.ESCALATE_HUMAN or context.escalation_requested:
            context.current_flow = ConversationFlow.ESCALATION
        
        # Check for farewell
        if current_intent == IntentType.FAREWELL:
            context.current_flow = ConversationFlow.FAREWELL
    
    async def _needs_entity_collection(self, context: ConversationContext) -> bool:
        """
        Check if we need to collect more entities for the current intent
        """
        if not context.current_intent:
            return False
        
        requirements = self.intent_requirements.get(context.current_intent, {})
        required_entities = requirements.get("required_entities", [])
        
        # Check if we have all required entities
        for entity_type in required_entities:
            if entity_type not in context.entities or not context.entities[entity_type]:
                return True
        
        return False
    
    async def _has_required_entities(self, context: ConversationContext) -> bool:
        """
        Check if we have all required entities for the current intent
        """
        return not await self._needs_entity_collection(context)
    
    async def _needs_confirmation(self, context: ConversationContext) -> bool:
        """
        Check if the current intent/action needs user confirmation
        """
        confirmation_intents = [
            IntentType.ORDER_CANCEL,
            IntentType.ORDER_RETURN,
            IntentType.ESCALATE_HUMAN
        ]
        
        return context.current_intent in confirmation_intents
    
    def _is_confirmation(self, message: str) -> bool:
        """
        Check if message is a confirmation (yes/no)
        """
        message_lower = message.lower().strip()
        
        positive = ["yes", "yeah", "yep", "sure", "ok", "okay", "correct", "right", "confirm"]
        negative = ["no", "nope", "cancel", "wrong", "incorrect", "stop"]
        
        return any(word in message_lower for word in positive + negative)
    
    async def get_missing_entities(self, context: ConversationContext) -> List[str]:
        """
        Get list of missing required entities for current intent
        """
        if not context.current_intent:
            return []
        
        requirements = self.intent_requirements.get(context.current_intent, {})
        required_entities = requirements.get("required_entities", [])
        
        missing = []
        for entity_type in required_entities:
            if entity_type not in context.entities or not context.entities[entity_type]:
                missing.append(entity_type)
        
        return missing
    
    async def execute_flow_handler(
        self, 
        context: ConversationContext
    ) -> Dict[str, Any]:
        """
        Execute the appropriate flow handler for current state
        """
        handler = self.flow_handlers.get(context.current_flow)
        if handler:
            return await handler(context)
        else:
            logger.warning(f"No handler found for flow: {context.current_flow}")
            return {"action": "fallback", "message": "I need to think about that."}
    
    # Flow handlers
    async def _handle_greeting(self, context: ConversationContext) -> Dict[str, Any]:
        """Handle greeting flow"""
        return {
            "action": "greeting",
            "message": "Hello! How can I help you today?",
            "next_flow": ConversationFlow.INTENT_DETECTION
        }
    
    async def _handle_intent_detection(self, context: ConversationContext) -> Dict[str, Any]:
        """Handle intent detection flow"""
        if context.current_intent == IntentType.UNKNOWN:
            return {
                "action": "clarification",
                "message": "I'm not sure I understand. Could you please tell me what you'd like help with?",
                "suggestions": [
                    "Find a product",
                    "Check order status",
                    "Get technical support",
                    "Speak to a human agent"
                ]
            }
        
        return {
            "action": "intent_detected",
            "intent": context.current_intent.value,
            "message": f"I understand you want help with {context.current_intent.value.replace('_', ' ')}."
        }
    
    async def _handle_entity_collection(self, context: ConversationContext) -> Dict[str, Any]:
        """Handle entity collection flow"""
        missing_entities = await self.get_missing_entities(context)
        
        if not missing_entities:
            return {"action": "entities_complete"}
        
        # Generate appropriate question for missing entity
        entity_questions = {
            "order_number": "What's your order number?",
            "product_name": "What product are you looking for?",
            "product_id": "Do you have the product ID or SKU?",
            "email": "What's your email address?",
            "phone": "What's your phone number?",
            "problem_description": "Can you describe the problem you're experiencing?",
            "return_reason": "What's the reason for the return?",
            "tracking_number": "What's your tracking number?",
            "price_range": "What's your budget or price range?",
            "category": "What type of product are you looking for?"
        }
        
        first_missing = missing_entities[0]
        question = entity_questions.get(first_missing, f"Could you provide {first_missing.replace('_', ' ')}?")
        
        return {
            "action": "collect_entity",
            "entity_type": first_missing,
            "message": question,
            "missing_entities": missing_entities
        }
    
    async def _handle_information_gathering(self, context: ConversationContext) -> Dict[str, Any]:
        """Handle information gathering flow"""
        # This would involve searching databases, calling APIs, etc.
        return {
            "action": "gather_information",
            "message": "Let me look that up for you..."
        }
    
    async def _handle_processing(self, context: ConversationContext) -> Dict[str, Any]:
        """Handle processing flow"""
        return {
            "action": "processing",
            "message": "I'm processing your request...",
            "intent": context.current_intent.value if context.current_intent else None
        }
    
    async def _handle_response_generation(self, context: ConversationContext) -> Dict[str, Any]:
        """Handle response generation flow"""
        return {
            "action": "generate_response",
            "context": context.to_dict()
        }
    
    async def _handle_confirmation(self, context: ConversationContext) -> Dict[str, Any]:
        """Handle confirmation flow"""
        return {
            "action": "request_confirmation",
            "message": "Is this correct? Please confirm with yes or no.",
            "context": context.entities
        }
    
    async def _handle_resolution(self, context: ConversationContext) -> Dict[str, Any]:
        """Handle resolution flow"""
        return {
            "action": "resolve",
            "message": "Great! Is there anything else I can help you with?",
            "resolved": True
        }
    
    async def _handle_escalation(self, context: ConversationContext) -> Dict[str, Any]:
        """Handle escalation flow"""
        context.escalation_requested = True
        return {
            "action": "escalate",
            "message": "I'm connecting you with a human agent who can better assist you.",
            "escalation_reason": "user_request"
        }
    
    async def _handle_farewell(self, context: ConversationContext) -> Dict[str, Any]:
        """Handle farewell flow"""
        return {
            "action": "farewell",
            "message": "Thank you for contacting us! Have a great day!",
            "conversation_ended": True
        }
    
    async def _cache_context(self, context: ConversationContext):
        """
        Cache conversation context in Redis
        """
        try:
            cache_key = f"conversation_context:{context.session_id}"
            cache_data = json.dumps(context.to_dict(), default=str)
            redis_client.setex(cache_key, 3600, cache_data)  # 1 hour TTL
        except Exception as e:
            logger.warning(f"Failed to cache context: {e}")
    
    async def cleanup_expired_contexts(self, max_age_hours: int = 24):
        """
        Clean up expired conversation contexts
        """
        cutoff_time = datetime.utcnow() - timedelta(hours=max_age_hours)
        
        # Clean up in-memory cache
        expired_sessions = []
        for session_id, context in self.context_cache.items():
            if context.last_activity < cutoff_time:
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            del self.context_cache[session_id]
        
        # Clean up Redis cache
        try:
            pattern = "conversation_context:*"
            keys = redis_client.keys(pattern)
            
            for key in keys:
                try:
                    cached_data = redis_client.get(key)
                    if cached_data:
                        context_data = json.loads(cached_data)
                        last_activity = datetime.fromisoformat(context_data["last_activity"])
                        if last_activity < cutoff_time:
                            redis_client.delete(key)
                except Exception as e:
                    logger.warning(f"Failed to process cached context {key}: {e}")
                    
        except Exception as e:
            logger.error(f"Failed to cleanup expired contexts: {e}")
    
    async def get_conversation_summary(self, session_id: str) -> Dict[str, Any]:
        """
        Get a summary of the conversation state
        """
        context = await self.get_conversation_context(session_id)
        
        return {
            "session_id": session_id,
            "current_flow": context.current_flow.value,
            "current_intent": context.current_intent.value if context.current_intent else None,
            "entities_collected": list(context.entities.keys()),
            "message_count": len(context.conversation_history),
            "last_activity": context.last_activity.isoformat(),
            "escalation_requested": context.escalation_requested,
            "resolution_attempts": context.resolution_attempts
        }