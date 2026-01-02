from fastapi import APIRouter, HTTPException, Depends, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
from datetime import datetime
import json
import uuid
from loguru import logger

from src.services.chatbot_orchestrator import ChatbotOrchestrator, ChatbotResponse
from src.core.auth import get_current_user_optional
from src.models.user import User


# Pydantic models for API
class ChatMessageRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=1000)
    session_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

class ChatMessageResponse(BaseModel):
    message: str
    actions: List[Dict[str, Any]]
    session_id: str
    message_id: str
    intent: Optional[str] = None
    entities: Dict[str, Any] = Field(default_factory=dict)
    confidence: float
    response_time_ms: int
    suggestions: List[str] = Field(default_factory=list)
    conversation_flow: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class FeedbackRequest(BaseModel):
    message_id: str
    is_helpful: bool
    feedback_text: Optional[str] = None

class ConversationHistoryResponse(BaseModel):
    messages: List[Dict[str, Any]]
    total_messages: int
    session_id: str

class AnalyticsResponse(BaseModel):
    total_conversations: int
    automation_rate: float
    escalation_rate: float
    average_satisfaction: Optional[float]
    intent_distribution: Dict[str, int]


# Router setup
router = APIRouter(prefix="/chatbot", tags=["chatbot"])

# Initialize chatbot orchestrator
chatbot = ChatbotOrchestrator()

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
    
    async def connect(self, websocket: WebSocket, session_id: str):
        await websocket.accept()
        self.active_connections[session_id] = websocket
        logger.info(f"WebSocket connected for session: {session_id}")
    
    def disconnect(self, session_id: str):
        if session_id in self.active_connections:
            del self.active_connections[session_id]
            logger.info(f"WebSocket disconnected for session: {session_id}")
    
    async def send_message(self, message: dict, session_id: str):
        if session_id in self.active_connections:
            await self.active_connections[session_id].send_text(json.dumps(message))

manager = ConnectionManager()


@router.post("/chat", response_model=ChatMessageResponse)
async def send_message(
    request: ChatMessageRequest,
    current_user: Optional[User] = Depends(get_current_user_optional)
):
    """
    Send message to chatbot and get response
    """
    try:
        # Generate session ID if not provided
        if not request.session_id:
            request.session_id = f"session_{uuid.uuid4().hex[:12]}"
        
        # Get user ID if authenticated
        user_id = str(current_user.id) if current_user else None
        
        # Add user info to metadata
        metadata = request.metadata.copy()
        if current_user:
            metadata.update({
                "user_id": user_id,
                "user_email": current_user.email
            })
        
        # Process message through chatbot
        response = await chatbot.process_user_message(
            session_id=request.session_id,
            user_message=request.message,
            user_id=user_id,
            metadata=metadata
        )
        
        # Convert to API response format
        api_response = ChatMessageResponse(
            message=response.message,
            actions=response.actions,
            session_id=response.session_id,
            message_id=response.message_id,
            intent=response.intent.value if response.intent else None,
            entities=response.entities or {},
            confidence=response.confidence,
            response_time_ms=response.response_time_ms,
            suggestions=response.suggestions or [],
            conversation_flow=response.conversation_flow.value if response.conversation_flow else None,
            metadata=response.metadata or {}
        )
        
        # Send to WebSocket if connected
        await manager.send_message({
            "type": "chat_response",
            "data": api_response.dict()
        }, request.session_id)
        
        return api_response
        
    except Exception as e:
        logger.error(f"Chat API error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for real-time chat
    """
    await manager.connect(websocket, session_id)
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            if message_data.get("type") == "chat_message":
                # Process chat message
                user_message = message_data.get("message", "")
                user_id = message_data.get("user_id")
                metadata = message_data.get("metadata", {})
                
                if user_message:
                    # Process through chatbot
                    response = await chatbot.process_user_message(
                        session_id=session_id,
                        user_message=user_message,
                        user_id=user_id,
                        metadata=metadata
                    )
                    
                    # Send response back
                    await websocket.send_text(json.dumps({
                        "type": "chat_response",
                        "data": {
                            "message": response.message,
                            "actions": response.actions,
                            "session_id": response.session_id,
                            "message_id": response.message_id,
                            "intent": response.intent.value if response.intent else None,
                            "entities": response.entities or {},
                            "confidence": response.confidence,
                            "response_time_ms": response.response_time_ms,
                            "suggestions": response.suggestions or [],
                            "conversation_flow": response.conversation_flow.value if response.conversation_flow else None,
                            "metadata": response.metadata or {}
                        }
                    }))
            
            elif message_data.get("type") == "typing":
                # Handle typing indicators
                await websocket.send_text(json.dumps({
                    "type": "typing_received",
                    "session_id": session_id
                }))
            
            elif message_data.get("type") == "feedback":
                # Handle feedback
                message_id = message_data.get("message_id")
                is_helpful = message_data.get("is_helpful", True)
                feedback_text = message_data.get("feedback_text")
                
                if message_id:
                    await chatbot.process_feedback(
                        session_id=session_id,
                        message_id=message_id,
                        is_helpful=is_helpful,
                        feedback_text=feedback_text
                    )
                    
                    await websocket.send_text(json.dumps({
                        "type": "feedback_received",
                        "message_id": message_id
                    }))
    
    except WebSocketDisconnect:
        manager.disconnect(session_id)
    except Exception as e:
        logger.error(f"WebSocket error for session {session_id}: {e}")
        manager.disconnect(session_id)


@router.post("/feedback")
async def submit_feedback(
    request: FeedbackRequest,
    current_user: Optional[User] = Depends(get_current_user_optional)
):
    """
    Submit feedback on chatbot response
    """
    try:
        # Extract session ID from message ID (simplified approach)
        session_id = "unknown"  # Would need to be tracked properly
        
        success = await chatbot.process_feedback(
            session_id=session_id,
            message_id=request.message_id,
            is_helpful=request.is_helpful,
            feedback_text=request.feedback_text
        )
        
        if success:
            return {"message": "Feedback submitted successfully"}
        else:
            raise HTTPException(status_code=404, detail="Message not found")
            
    except Exception as e:
        logger.error(f"Feedback API error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history/{session_id}", response_model=ConversationHistoryResponse)
async def get_conversation_history(
    session_id: str,
    limit: int = 20,
    current_user: Optional[User] = Depends(get_current_user_optional)
):
    """
    Get conversation history for a session
    """
    try:
        history = await chatbot.get_conversation_history(
            session_id=session_id,
            limit=limit
        )
        
        return ConversationHistoryResponse(
            messages=history,
            total_messages=len(history),
            session_id=session_id
        )
        
    except Exception as e:
        logger.error(f"History API error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/end/{session_id}")
async def end_conversation(
    session_id: str,
    current_user: Optional[User] = Depends(get_current_user_optional)
):
    """
    End a conversation session
    """
    try:
        success = await chatbot.end_conversation(
            session_id=session_id,
            reason="user_ended"
        )
        
        # Disconnect WebSocket if connected
        manager.disconnect(session_id)
        
        if success:
            return {"message": "Conversation ended successfully"}
        else:
            raise HTTPException(status_code=404, detail="Conversation not found")
            
    except Exception as e:
        logger.error(f"End conversation API error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/escalate/{session_id}")
async def escalate_conversation(
    session_id: str,
    escalation_reason: str = "user_request",
    current_user: Optional[User] = Depends(get_current_user_optional)
):
    """
    Escalate conversation to human agent
    """
    try:
        result = await chatbot.escalate_to_human(
            session_id=session_id,
            escalation_reason=escalation_reason
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Escalation API error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analytics", response_model=AnalyticsResponse)
async def get_chatbot_analytics(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    current_user: User = Depends(get_current_user_optional)  # Might require admin role
):
    """
    Get chatbot analytics and metrics
    """
    try:
        date_range = None
        if start_date and end_date:
            date_range = (start_date, end_date)
        
        analytics = await chatbot.get_conversation_analytics(
            date_range=date_range
        )
        
        return AnalyticsResponse(
            total_conversations=analytics.get("total_conversations", 0),
            automation_rate=analytics.get("automation_rate", 0.0),
            escalation_rate=analytics.get("escalation_rate", 0.0),
            average_satisfaction=analytics.get("average_satisfaction"),
            intent_distribution=analytics.get("intent_distribution", {})
        )
        
    except Exception as e:
        logger.error(f"Analytics API error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def chatbot_health():
    """
    Health check endpoint for chatbot service
    """
    try:
        # Basic health checks
        health_status = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "services": {
                "intent_classification": True,
                "entity_extraction": True,
                "response_generation": True,
                "knowledge_base": True
            },
            "active_websockets": len(manager.active_connections)
        }
        
        return health_status
        
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }


# Additional utility endpoints

@router.get("/intents")
async def get_available_intents():
    """
    Get list of available intents
    """
    from src.models.conversation import IntentType
    
    intents = [
        {
            "value": intent.value,
            "name": intent.value.replace("_", " ").title(),
            "category": intent.value.split("_")[0] if "_" in intent.value else "general"
        }
        for intent in IntentType
    ]
    
    return {"intents": intents}


@router.get("/suggestions/{intent}")
async def get_intent_suggestions(intent: str):
    """
    Get example phrases for a specific intent
    """
    try:
        from src.models.conversation import IntentType
        
        # Map intent string to enum
        intent_enum = None
        for i in IntentType:
            if i.value == intent:
                intent_enum = i
                break
        
        if not intent_enum:
            raise HTTPException(status_code=404, detail="Intent not found")
        
        # Get suggestions from response service
        suggestions = await chatbot.response_service.get_response_suggestions(
            ResponseContext(
                intent=intent_enum,
                entities={},
                conversation_flow=ConversationFlow.INTENT_DETECTION,
                user_id=None,
                conversation_history=[],
                confidence_scores={},
                missing_entities=[],
                products_mentioned=[],
                user_preferences={}
            )
        )
        
        return {"suggestions": [s["text"] for s in suggestions]}
        
    except Exception as e:
        logger.error(f"Suggestions API error: {e}")
        raise HTTPException(status_code=500, detail=str(e))