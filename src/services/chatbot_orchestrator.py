import asyncio
from typing import Dict, List, Any, Optional, Tuple
from loguru import logger
import json
from datetime import datetime
import uuid
from dataclasses import dataclass

from src.core.database import get_db
from src.models.conversation import (
    Conversation, ConversationMessage, IntentType, 
    MessageType, ConversationStatus, ConversationFlow
)
from src.services.intent_classification import IntentClassificationService
from src.services.entity_extraction import EntityExtractionService
from src.services.conversation_state import ConversationStateManager, ConversationContext
from src.services.response_generation import ResponseGenerationService, ResponseContext
from src.services.knowledge_base import KnowledgeBaseService


@dataclass
class ChatbotResponse:
    """Structured chatbot response"""
    message: str
    actions: List[Dict[str, Any]]
    session_id: str
    message_id: str
    intent: Optional[IntentType] = None
    entities: Dict[str, Any] = None
    confidence: float = 0.0
    response_time_ms: int = 0
    suggestions: List[str] = None
    conversation_flow: Optional[ConversationFlow] = None
    metadata: Dict[str, Any] = None


class ChatbotOrchestrator:
    """
    Main orchestrator for chatbot conversations
    Coordinates all chatbot services for end-to-end conversation handling
    """
    
    def __init__(self):
        self.intent_service = IntentClassificationService()
        self.entity_service = EntityExtractionService()
        self.state_manager = ConversationStateManager()
        self.response_service = ResponseGenerationService()
        self.knowledge_service = KnowledgeBaseService()
        
        # Performance tracking
        self.conversation_metrics = {}
        
        logger.info("Chatbot orchestrator initialized")
    
    async def process_user_message(
        self,
        session_id: str,
        user_message: str,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ChatbotResponse:
        """
        Process user message through complete chatbot pipeline
        """
        start_time = datetime.utcnow()
        message_id = f"msg_{uuid.uuid4().hex[:12]}"
        
        try:
            logger.info(f"Processing message for session {session_id}: {user_message[:100]}...")
            
            # 1. Get or create conversation context
            context = await self.state_manager.get_conversation_context(session_id)
            
            # 2. Intent classification
            intent_result = await self.intent_service.classify_intent(
                text=user_message,
                context={
                    "previous_intent": context.current_intent,
                    "entities": context.entities,
                    "conversation_history": context.conversation_history[-5:]
                }
            )
            
            detected_intent = intent_result["intent"]
            intent_confidence = intent_result["confidence"]
            
            logger.debug(f"Intent: {detected_intent.value}, Confidence: {intent_confidence:.3f}")
            
            # 3. Entity extraction
            entity_result = await self.entity_service.extract_entities(
                text=user_message,
                context={
                    "current_intent": detected_intent,
                    "conversation_history": context.conversation_history
                }
            )
            
            extracted_entities = entity_result["entities"]
            entity_confidence = entity_result["confidence"]
            
            logger.debug(f"Entities: {list(extracted_entities.keys())}")
            
            # 4. Update conversation state
            updated_context = await self.state_manager.update_conversation_context(
                session_id=session_id,
                message_content=user_message,
                message_type=MessageType.USER,
                intent=detected_intent,
                entities=extracted_entities,
                confidence_scores={
                    "intent": intent_confidence,
                    "entity": entity_confidence
                }
            )
            
            # 5. Create response context
            response_context = ResponseContext(
                intent=detected_intent,
                entities=updated_context.entities,
                conversation_flow=updated_context.current_flow,
                user_id=user_id,
                conversation_history=updated_context.conversation_history,
                confidence_scores=updated_context.confidence_scores,
                missing_entities=await self.state_manager.get_missing_entities(updated_context),
                products_mentioned=self._extract_product_mentions(updated_context),
                user_preferences=updated_context.user_preferences
            )
            
            # 6. Execute flow handler to get initial response strategy
            flow_result = await self.state_manager.execute_flow_handler(updated_context)
            
            # 7. Generate response
            response_result = await self.response_service.generate_response(
                context=updated_context,
                user_message=user_message,
                response_context=response_context
            )
            
            # 8. Enhance response with knowledge base if needed
            if self._needs_knowledge_enhancement(detected_intent, response_result):
                response_result = await self._enhance_with_knowledge(
                    response_result,
                    user_message,
                    detected_intent,
                    updated_context
                )
            
            # 9. Store conversation in database
            await self._store_conversation_data(
                session_id=session_id,
                user_id=user_id,
                user_message=user_message,
                bot_response=response_result["message"],
                intent=detected_intent,
                entities=extracted_entities,
                confidence_scores={
                    "intent": intent_confidence,
                    "entity": entity_confidence,
                    "response": response_result.get("confidence", 0.8)
                },
                metadata=metadata or {}
            )
            
            # 10. Update conversation state with bot response
            await self.state_manager.update_conversation_context(
                session_id=session_id,
                message_content=response_result["message"],
                message_type=MessageType.BOT,
                confidence_scores={"response": response_result.get("confidence", 0.8)}
            )
            
            # 11. Calculate response time
            response_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            # 12. Get response suggestions
            suggestions = await self._get_response_suggestions(updated_context, detected_intent)
            
            # 13. Create structured response
            chatbot_response = ChatbotResponse(
                message=response_result["message"],
                actions=response_result.get("actions", []),
                session_id=session_id,
                message_id=message_id,
                intent=detected_intent,
                entities=extracted_entities,
                confidence=min(intent_confidence, entity_confidence),
                response_time_ms=int(response_time),
                suggestions=suggestions,
                conversation_flow=updated_context.current_flow,
                metadata={
                    "flow_action": flow_result.get("action"),
                    "response_method": response_result.get("generation_method"),
                    "knowledge_used": response_result.get("knowledge_used", False),
                    **response_result.get("metadata", {})
                }
            )
            
            logger.info(f"Response generated in {response_time:.0f}ms for session {session_id}")
            return chatbot_response
            
        except Exception as e:
            logger.error(f"Error processing message for session {session_id}: {e}")
            return await self._create_error_response(session_id, message_id, str(e))
    
    def _extract_product_mentions(self, context: ConversationContext) -> List[str]:
        """
        Extract product mentions from conversation context
        """
        products = []
        
        # From current entities
        if context.entities.get("product_name"):
            products.extend(context.entities["product_name"])
        
        if context.entities.get("product_id"):
            products.extend(context.entities["product_id"])
        
        # From conversation history
        for message in context.conversation_history:
            entities = message.get("entities", {})
            if entities.get("product_name"):
                products.extend(entities["product_name"])
        
        return list(set(products))  # Remove duplicates
    
    def _needs_knowledge_enhancement(
        self,
        intent: IntentType,
        response_result: Dict[str, Any]
    ) -> bool:
        """
        Determine if response needs knowledge base enhancement
        """
        # Enhance for information-seeking intents
        knowledge_intents = [
            IntentType.PRODUCT_INFO,
            IntentType.TECHNICAL_SUPPORT,
            IntentType.SHIPPING_INFO,
            IntentType.PAYMENT_HELP,
            IntentType.ACCOUNT_HELP,
            IntentType.ORDER_STATUS
        ]
        
        if intent in knowledge_intents:
            return True
        
        # Enhance if response confidence is low
        if response_result.get("confidence", 1.0) < 0.7:
            return True
        
        # Enhance if response is fallback
        if response_result.get("method") == "fallback":
            return True
        
        return False
    
    async def _enhance_with_knowledge(
        self,
        response_result: Dict[str, Any],
        user_message: str,
        intent: IntentType,
        context: ConversationContext
    ) -> Dict[str, Any]:
        """
        Enhance response with knowledge base information
        """
        try:
            # Search knowledge base
            knowledge_entries = await self.knowledge_service.search_knowledge(
                query=user_message,
                intent=intent,
                context=context.entities,
                limit=3
            )
            
            if not knowledge_entries:
                return response_result
            
            # Use the most relevant knowledge entry
            best_entry = knowledge_entries[0]
            
            # Enhance the response message
            enhanced_message = response_result["message"]
            
            # Add knowledge content if it's helpful
            if best_entry.effectiveness_score > 0.6:
                if len(enhanced_message) < 200:  # Only add if response is short
                    enhanced_message += f"\n\n{best_entry.summary}"
                
                # Add knowledge as action if content is substantial
                if len(best_entry.content) > 100:
                    knowledge_action = {
                        "type": "knowledge",
                        "title": best_entry.title,
                        "content": best_entry.content[:500],  # Truncate long content
                        "category": best_entry.category
                    }
                    
                    if "actions" not in response_result:
                        response_result["actions"] = []
                    response_result["actions"].append(knowledge_action)
            
            # Update response
            response_result["message"] = enhanced_message
            response_result["knowledge_used"] = True
            response_result["knowledge_entries"] = [
                {
                    "title": entry.title,
                    "score": entry.effectiveness_score,
                    "category": entry.category
                }
                for entry in knowledge_entries[:2]
            ]
            
            logger.debug(f"Enhanced response with knowledge: {best_entry.title}")
            
            return response_result
            
        except Exception as e:
            logger.warning(f"Knowledge enhancement failed: {e}")
            return response_result
    
    async def _store_conversation_data(
        self,
        session_id: str,
        user_id: Optional[str],
        user_message: str,
        bot_response: str,
        intent: IntentType,
        entities: Dict[str, Any],
        confidence_scores: Dict[str, float],
        metadata: Dict[str, Any]
    ):
        """
        Store conversation data in database
        """
        try:
            db = next(get_db())
            
            # Get or create conversation
            conversation = db.query(Conversation).filter(
                Conversation.session_id == session_id
            ).first()
            
            if not conversation:
                conversation = Conversation(
                    session_id=session_id,
                    user_id=user_id,
                    status=ConversationStatus.ACTIVE,
                    current_intent=intent,
                    context={},
                    user_agent=metadata.get("user_agent"),
                    ip_address=metadata.get("ip_address")
                )
                db.add(conversation)
                db.commit()
                db.refresh(conversation)
            else:
                # Update existing conversation
                conversation.current_intent = intent
                conversation.last_activity_at = datetime.utcnow()
                db.commit()
            
            # Store user message
            user_msg = ConversationMessage(
                conversation_id=conversation.id,
                message_type=MessageType.USER,
                content=user_message,
                detected_intent=intent,
                intent_confidence=confidence_scores.get("intent"),
                extracted_entities=entities
            )
            db.add(user_msg)
            
            # Store bot response
            bot_msg = ConversationMessage(
                conversation_id=conversation.id,
                message_type=MessageType.BOT,
                content=bot_response,
                detected_intent=intent,
                intent_confidence=confidence_scores.get("response"),
                response_source="ai_chatbot",
                model_version="v1.0"
            )
            db.add(bot_msg)
            
            db.commit()
            db.close()
            
        except Exception as e:
            logger.error(f"Failed to store conversation data: {e}")
    
    async def _get_response_suggestions(
        self,
        context: ConversationContext,
        intent: IntentType
    ) -> List[str]:
        """
        Get response suggestions for user
        """
        try:
            suggestions = []
            
            # Intent-specific suggestions
            if intent == IntentType.PRODUCT_SEARCH:
                suggestions.extend([
                    "Show me popular products",
                    "Browse by category",
                    "What's on sale?"
                ])
            
            elif intent == IntentType.PRODUCT_INFO:
                suggestions.extend([
                    "Show specifications",
                    "Customer reviews",
                    "Similar products"
                ])
            
            elif intent == IntentType.ORDER_STATUS:
                suggestions.extend([
                    "Track my order",
                    "Order details",
                    "Change delivery address"
                ])
            
            elif intent == IntentType.TECHNICAL_SUPPORT:
                suggestions.extend([
                    "Common solutions",
                    "Contact support",
                    "Troubleshooting guide"
                ])
            
            # Flow-specific suggestions
            if context.current_flow == ConversationFlow.ENTITY_COLLECTION:
                missing_entities = await self.state_manager.get_missing_entities(context)
                if "order_number" in missing_entities:
                    suggestions.append("I don't have my order number")
                if "product_name" in missing_entities:
                    suggestions.append("I'm not sure of the exact name")
            
            # General suggestions
            suggestions.extend([
                "Speak with human agent",
                "Start over"
            ])
            
            return suggestions[:4]  # Return top 4 suggestions
            
        except Exception as e:
            logger.warning(f"Failed to get response suggestions: {e}")
            return []
    
    async def _create_error_response(
        self,
        session_id: str,
        message_id: str,
        error_message: str
    ) -> ChatbotResponse:
        """
        Create error response when processing fails
        """
        return ChatbotResponse(
            message="I apologize, but I'm having trouble right now. Please try again in a moment or contact our support team.",
            actions=[
                {
                    "type": "quick_reply",
                    "text": "Try again",
                    "payload": "retry"
                },
                {
                    "type": "quick_reply",
                    "text": "Contact support",
                    "payload": "contact_support"
                }
            ],
            session_id=session_id,
            message_id=message_id,
            intent=IntentType.UNKNOWN,
            entities={},
            confidence=0.0,
            response_time_ms=0,
            suggestions=["Try again", "Contact support"],
            conversation_flow=ConversationFlow.GREETING,
            metadata={"error": error_message, "error_type": "processing_failure"}
        )
    
    async def get_conversation_history(
        self,
        session_id: str,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Get conversation history for a session
        """
        try:
            db = next(get_db())
            
            conversation = db.query(Conversation).filter(
                Conversation.session_id == session_id
            ).first()
            
            if not conversation:
                return []
            
            messages = db.query(ConversationMessage).filter(
                ConversationMessage.conversation_id == conversation.id
            ).order_by(ConversationMessage.created_at.desc()).limit(limit).all()
            
            history = []
            for msg in reversed(messages):  # Reverse to get chronological order
                history.append({
                    "id": str(msg.id),
                    "message_type": msg.message_type.value,
                    "content": msg.content,
                    "intent": msg.detected_intent.value if msg.detected_intent else None,
                    "entities": msg.extracted_entities or {},
                    "confidence": msg.intent_confidence,
                    "created_at": msg.created_at.isoformat(),
                    "helpful": msg.helpful
                })
            
            db.close()
            return history
            
        except Exception as e:
            logger.error(f"Failed to get conversation history: {e}")
            return []
    
    async def end_conversation(
        self,
        session_id: str,
        reason: str = "user_ended"
    ) -> bool:
        """
        End conversation and mark as resolved
        """
        try:
            db = next(get_db())
            
            conversation = db.query(Conversation).filter(
                Conversation.session_id == session_id
            ).first()
            
            if conversation:
                conversation.status = ConversationStatus.RESOLVED
                conversation.resolved_at = datetime.utcnow()
                conversation.resolved_automatically = reason != "escalated"
                db.commit()
                
                # Clean up context cache
                if session_id in self.state_manager.context_cache:
                    del self.state_manager.context_cache[session_id]
                
                logger.info(f"Conversation ended: {session_id}, Reason: {reason}")
                
            db.close()
            return True
            
        except Exception as e:
            logger.error(f"Failed to end conversation: {e}")
            return False
    
    async def escalate_to_human(
        self,
        session_id: str,
        escalation_reason: str = "user_request"
    ) -> Dict[str, Any]:
        """
        Escalate conversation to human agent
        """
        try:
            db = next(get_db())
            
            conversation = db.query(Conversation).filter(
                Conversation.session_id == session_id
            ).first()
            
            if conversation:
                conversation.status = ConversationStatus.ESCALATED
                conversation.escalated_to_human = True
                db.commit()
                
                logger.info(f"Conversation escalated: {session_id}, Reason: {escalation_reason}")
                
                # Return escalation information
                return {
                    "escalated": True,
                    "ticket_id": f"TICKET_{session_id[-8:].upper()}",
                    "estimated_wait_time": "5-10 minutes",
                    "escalation_reason": escalation_reason,
                    "message": "I'm connecting you with a human agent who can better assist you."
                }
            
            db.close()
            
        except Exception as e:
            logger.error(f"Failed to escalate conversation: {e}")
            
        return {
            "escalated": False,
            "error": "Failed to escalate to human agent"
        }
    
    async def process_feedback(
        self,
        session_id: str,
        message_id: str,
        is_helpful: bool,
        feedback_text: Optional[str] = None
    ) -> bool:
        """
        Process user feedback on chatbot responses
        """
        try:
            db = next(get_db())
            
            # Find the message
            message = db.query(ConversationMessage).filter(
                ConversationMessage.id == message_id
            ).first()
            
            if message:
                message.helpful = is_helpful
                message.feedback_text = feedback_text
                db.commit()
                
                # Update knowledge base effectiveness if knowledge was used
                # This would be enhanced with more sophisticated feedback processing
                
                logger.info(f"Feedback processed: {message_id}, Helpful: {is_helpful}")
                
            db.close()
            return True
            
        except Exception as e:
            logger.error(f"Failed to process feedback: {e}")
            return False
    
    async def get_conversation_analytics(
        self,
        session_id: Optional[str] = None,
        date_range: Optional[Tuple[datetime, datetime]] = None
    ) -> Dict[str, Any]:
        """
        Get conversation analytics
        """
        try:
            db = next(get_db())
            
            # Base query
            query = db.query(Conversation)
            
            # Apply filters
            if session_id:
                query = query.filter(Conversation.session_id == session_id)
            
            if date_range:
                start_date, end_date = date_range
                query = query.filter(
                    Conversation.started_at >= start_date,
                    Conversation.started_at <= end_date
                )
            
            conversations = query.all()
            
            if not conversations:
                return {"total_conversations": 0}
            
            # Calculate metrics
            total_conversations = len(conversations)
            resolved_automatically = sum(1 for c in conversations if c.resolved_automatically)
            escalated = sum(1 for c in conversations if c.escalated_to_human)
            
            # Average satisfaction (if available)
            ratings = [c.satisfaction_rating for c in conversations if c.satisfaction_rating]
            avg_satisfaction = sum(ratings) / len(ratings) if ratings else None
            
            # Intent distribution
            intent_counts = {}
            for conv in conversations:
                if conv.current_intent:
                    intent = conv.current_intent.value
                    intent_counts[intent] = intent_counts.get(intent, 0) + 1
            
            db.close()
            
            return {
                "total_conversations": total_conversations,
                "resolved_automatically": resolved_automatically,
                "escalated_to_human": escalated,
                "automation_rate": resolved_automatically / total_conversations if total_conversations > 0 else 0,
                "escalation_rate": escalated / total_conversations if total_conversations > 0 else 0,
                "average_satisfaction": avg_satisfaction,
                "intent_distribution": intent_counts,
                "analysis_date": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get conversation analytics: {e}")
            return {"error": str(e)}