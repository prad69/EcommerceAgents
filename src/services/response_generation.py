import asyncio
from typing import Dict, List, Any, Optional, Tuple
from loguru import logger
import json
from datetime import datetime
import random
from dataclasses import dataclass

from src.core.config import settings
from src.core.database import get_db
from src.models.conversation import IntentType, ConversationFlow
from src.models.product import Product
from src.services.conversation_state import ConversationContext
from src.services.recommendation_engine import RecommendationService
from src.services.review_analysis import ReviewAnalysisService


@dataclass
class ResponseContext:
    """Context for response generation"""
    intent: IntentType
    entities: Dict[str, Any]
    conversation_flow: ConversationFlow
    user_id: Optional[str]
    conversation_history: List[Dict[str, Any]]
    confidence_scores: Dict[str, float]
    missing_entities: List[str]
    products_mentioned: List[str]
    user_preferences: Dict[str, Any]


class ResponseGenerationService:
    """
    Intelligent response generation for chatbot conversations
    """
    
    def __init__(self):
        self.openai_client = None
        self.recommendation_service = RecommendationService()
        self.review_service = ReviewAnalysisService()
        self.response_templates = {}
        self._setup_models()
        self._setup_response_templates()
    
    def _setup_models(self):
        """
        Initialize AI models and services
        """
        # Setup OpenAI
        if settings.openai_api_key:
            try:
                import openai
                self.openai_client = openai.AsyncOpenAI(
                    api_key=settings.openai_api_key
                )
                logger.info("OpenAI response generation client initialized")
            except ImportError:
                logger.warning("OpenAI library not available")
    
    def _setup_response_templates(self):
        """
        Setup response templates for different intents and scenarios
        """
        self.response_templates = {
            # Greeting responses
            IntentType.GREETING: {
                "default": [
                    "Hello! Welcome to our store. How can I help you today?",
                    "Hi there! I'm here to help you find what you're looking for.",
                    "Good day! What can I assist you with?",
                    "Hello! I'm your shopping assistant. How may I help you?"
                ],
                "returning_user": [
                    "Welcome back! How can I help you today?",
                    "Great to see you again! What are you looking for?",
                    "Hello again! Ready to find something great?"
                ]
            },
            
            # Product search responses
            IntentType.PRODUCT_SEARCH: {
                "initial": [
                    "I'd be happy to help you find products! What are you looking for?",
                    "Let me help you search for products. What do you have in mind?",
                    "I can help you find exactly what you need. Tell me what you're looking for."
                ],
                "found_results": "I found {count} products that match your search for '{query}'. Here are the top results:",
                "no_results": "I couldn't find any products matching '{query}'. Would you like to try a different search term or browse our categories?",
                "refined_search": "Based on your preferences, here are some refined results:"
            },
            
            # Product information responses
            IntentType.PRODUCT_INFO: {
                "product_found": "Here's detailed information about {product_name}:",
                "product_not_found": "I couldn't find information about that specific product. Could you provide more details like a product ID or name?",
                "features": "Let me tell you about the key features of {product_name}:",
                "specifications": "Here are the specifications for {product_name}:"
            },
            
            # Order status responses
            IntentType.ORDER_STATUS: {
                "order_found": "I found your order #{order_number}. Here's the current status:",
                "order_not_found": "I couldn't find an order with that number. Please double-check the order number or contact support.",
                "need_verification": "For security, I need to verify your identity. Can you provide the email address associated with this order?"
            },
            
            # Technical support responses
            IntentType.TECHNICAL_SUPPORT: {
                "acknowledge": "I understand you're experiencing a technical issue. Let me help you resolve this.",
                "gathering_info": "To better assist you, could you tell me more about the problem?",
                "troubleshooting": "Let's try these troubleshooting steps:",
                "escalation": "This seems like it might need specialized help. Let me connect you with our technical support team."
            },
            
            # Farewell responses
            IntentType.FAREWELL: {
                "default": [
                    "Thank you for visiting! Have a wonderful day!",
                    "It was great helping you today. Come back anytime!",
                    "Thank you for shopping with us. Take care!",
                    "Have a great day! Feel free to return if you need anything else."
                ],
                "purchase_made": [
                    "Thank you for your purchase! Your order will be processed shortly.",
                    "Thanks for shopping with us! You'll receive order confirmation soon.",
                    "Great choice! Your order is being prepared for shipment."
                ]
            },
            
            # Error and fallback responses
            "fallback": [
                "I'm not sure I understand. Could you please rephrase that?",
                "I need a bit more information to help you properly. Could you be more specific?",
                "I'm having trouble understanding your request. Would you like to speak with a human agent?",
                "Let me try to help you in a different way. What specifically are you looking for?"
            ],
            
            "clarification": [
                "Could you clarify what you mean by '{unclear_part}'?",
                "I want to make sure I understand correctly. Are you asking about '{interpretation}'?",
                "To better assist you, could you provide more details about '{topic}'?"
            ]
        }
    
    async def generate_response(
        self, 
        context: ConversationContext,
        user_message: str,
        response_context: ResponseContext
    ) -> Dict[str, Any]:
        """
        Generate intelligent response based on conversation context
        """
        try:
            # Determine response type based on intent and flow
            response_type = self._determine_response_type(response_context)
            
            # Generate response using multiple strategies
            response_candidates = []
            
            # 1. Template-based response (fast and reliable)
            template_response = await self._generate_template_response(response_context, response_type)
            if template_response:
                response_candidates.append(template_response)
            
            # 2. AI-generated response (higher quality, personalized)
            if self.openai_client and len(response_candidates) == 0:
                try:
                    ai_response = await self._generate_ai_response(context, user_message, response_context)
                    if ai_response:
                        response_candidates.append(ai_response)
                except Exception as e:
                    logger.warning(f"AI response generation failed: {e}")
            
            # 3. Rule-based fallback
            if not response_candidates:
                fallback_response = await self._generate_fallback_response(response_context)
                response_candidates.append(fallback_response)
            
            # Select best response
            best_response = self._select_best_response(response_candidates, response_context)
            
            # Enhance response with additional data
            enhanced_response = await self._enhance_response(best_response, response_context)
            
            # Add metadata
            enhanced_response.update({
                "response_id": f"resp_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{random.randint(1000, 9999)}",
                "generation_method": best_response.get("method", "template"),
                "confidence": best_response.get("confidence", 0.8),
                "response_time": datetime.utcnow().isoformat(),
                "intent": response_context.intent.value,
                "flow": response_context.conversation_flow.value
            })
            
            return enhanced_response
            
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            return self._create_error_response(str(e))
    
    def _determine_response_type(self, context: ResponseContext) -> str:
        """
        Determine what type of response to generate
        """
        intent = context.intent
        flow = context.conversation_flow
        missing_entities = context.missing_entities
        
        # Entity collection flow
        if flow == ConversationFlow.ENTITY_COLLECTION and missing_entities:
            return "collect_entity"
        
        # Confirmation flow
        if flow == ConversationFlow.CONFIRMATION:
            return "confirmation"
        
        # Intent-specific responses
        if intent == IntentType.GREETING:
            return "returning_user" if context.user_id else "default"
        
        elif intent == IntentType.PRODUCT_SEARCH:
            if context.entities.get("product_name") or context.entities.get("category"):
                return "search_results"
            else:
                return "initial"
        
        elif intent == IntentType.PRODUCT_INFO:
            if context.entities.get("product_id") or context.entities.get("product_name"):
                return "product_details"
            else:
                return "need_product_info"
        
        elif intent == IntentType.ORDER_STATUS:
            if context.entities.get("order_number"):
                return "check_order"
            else:
                return "need_order_number"
        
        elif intent == IntentType.TECHNICAL_SUPPORT:
            if context.entities.get("problem_description"):
                return "troubleshoot"
            else:
                return "gather_problem_info"
        
        elif intent == IntentType.ESCALATE_HUMAN:
            return "escalate"
        
        elif intent == IntentType.FAREWELL:
            return "default"
        
        else:
            return "default"
    
    async def _generate_template_response(
        self, 
        context: ResponseContext, 
        response_type: str
    ) -> Optional[Dict[str, Any]]:
        """
        Generate response using predefined templates
        """
        try:
            intent = context.intent
            templates = self.response_templates.get(intent, {})
            
            if response_type in templates:
                template = templates[response_type]
            else:
                template = templates.get("default", self.response_templates["fallback"])
            
            # Handle different template types
            if isinstance(template, list):
                message = random.choice(template)
            elif isinstance(template, str):
                message = template
            else:
                message = str(template)
            
            # Format template with context entities
            try:
                message = message.format(**context.entities)
            except KeyError:
                # If formatting fails, use message as-is
                pass
            
            # Add action items based on intent and context
            actions = await self._generate_template_actions(context, response_type)
            
            return {
                "message": message,
                "actions": actions,
                "method": "template",
                "confidence": 0.8,
                "response_type": response_type
            }
            
        except Exception as e:
            logger.warning(f"Template response generation failed: {e}")
            return None
    
    async def _generate_template_actions(
        self, 
        context: ResponseContext, 
        response_type: str
    ) -> List[Dict[str, Any]]:
        """
        Generate action items for template responses
        """
        actions = []
        intent = context.intent
        
        # Product search actions
        if intent == IntentType.PRODUCT_SEARCH:
            if response_type == "search_results":
                # Get product recommendations
                try:
                    query = context.entities.get("product_name", "") or context.entities.get("category", "")
                    if query:
                        # This would normally call the recommendation service
                        actions.append({
                            "type": "product_list",
                            "title": "Search Results",
                            "products": []  # Would be populated by recommendation service
                        })
                except Exception as e:
                    logger.warning(f"Failed to get product recommendations: {e}")
            
            actions.extend([
                {
                    "type": "quick_reply",
                    "text": "Show categories",
                    "payload": "browse_categories"
                },
                {
                    "type": "quick_reply",
                    "text": "Popular products",
                    "payload": "show_popular"
                }
            ])
        
        # Product info actions
        elif intent == IntentType.PRODUCT_INFO:
            actions.extend([
                {
                    "type": "quick_reply",
                    "text": "Show specifications",
                    "payload": "show_specs"
                },
                {
                    "type": "quick_reply", 
                    "text": "Customer reviews",
                    "payload": "show_reviews"
                },
                {
                    "type": "quick_reply",
                    "text": "Similar products",
                    "payload": "show_similar"
                }
            ])
        
        # Order status actions
        elif intent == IntentType.ORDER_STATUS:
            actions.extend([
                {
                    "type": "quick_reply",
                    "text": "Track shipment",
                    "payload": "track_order"
                },
                {
                    "type": "quick_reply",
                    "text": "Order details",
                    "payload": "order_details"
                }
            ])
        
        # Technical support actions
        elif intent == IntentType.TECHNICAL_SUPPORT:
            actions.extend([
                {
                    "type": "quick_reply",
                    "text": "Common solutions",
                    "payload": "common_solutions"
                },
                {
                    "type": "quick_reply",
                    "text": "Contact support",
                    "payload": "contact_support"
                }
            ])
        
        # General help actions
        if not actions:
            actions.extend([
                {
                    "type": "quick_reply",
                    "text": "Browse products",
                    "payload": "browse_products"
                },
                {
                    "type": "quick_reply",
                    "text": "Help",
                    "payload": "show_help"
                }
            ])
        
        return actions
    
    async def _generate_ai_response(
        self, 
        context: ConversationContext,
        user_message: str,
        response_context: ResponseContext
    ) -> Optional[Dict[str, Any]]:
        """
        Generate response using OpenAI GPT
        """
        try:
            # Build conversation history for context
            conversation_history = self._build_conversation_history(context.conversation_history)
            
            # Create system prompt
            system_prompt = self._create_system_prompt(response_context)
            
            # Create user prompt with context
            user_prompt = self._create_user_prompt(user_message, response_context)
            
            # Generate response
            response = await self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    *conversation_history,
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=300,
                temperature=0.7
            )
            
            ai_message = response.choices[0].message.content.strip()
            
            # Parse structured response if JSON format
            try:
                if ai_message.startswith('{') and ai_message.endswith('}'):
                    parsed_response = json.loads(ai_message)
                    return {
                        "message": parsed_response.get("message", ai_message),
                        "actions": parsed_response.get("actions", []),
                        "method": "ai_generated",
                        "confidence": 0.9,
                        "metadata": parsed_response.get("metadata", {})
                    }
            except json.JSONDecodeError:
                pass
            
            # Return simple text response
            return {
                "message": ai_message,
                "actions": [],
                "method": "ai_generated", 
                "confidence": 0.9
            }
            
        except Exception as e:
            logger.error(f"AI response generation error: {e}")
            return None
    
    def _create_system_prompt(self, context: ResponseContext) -> str:
        """
        Create system prompt for AI response generation
        """
        return f"""
        You are a helpful e-commerce customer service assistant. Your role is to:
        
        1. Help customers find products, get information, and resolve issues
        2. Be friendly, professional, and concise
        3. Always try to be helpful and guide users toward solutions
        4. When you don't have specific information, offer alternative help
        
        Current conversation context:
        - Intent: {context.intent.value}
        - Flow: {context.conversation_flow.value}
        - Available entities: {list(context.entities.keys())}
        - Missing entities: {context.missing_entities}
        
        Response guidelines:
        - Keep responses under 150 words
        - Be conversational and helpful
        - Include specific product information when available
        - Offer clear next steps or actions
        - If you need more information, ask specific questions
        """
    
    def _create_user_prompt(self, user_message: str, context: ResponseContext) -> str:
        """
        Create user prompt with context for AI
        """
        context_info = []
        
        if context.entities:
            context_info.append(f"Extracted entities: {json.dumps(context.entities, indent=2)}")
        
        if context.products_mentioned:
            context_info.append(f"Products mentioned: {', '.join(context.products_mentioned)}")
        
        if context.missing_entities:
            context_info.append(f"Still need: {', '.join(context.missing_entities)}")
        
        context_str = "\n".join(context_info) if context_info else "No additional context"
        
        return f"""
        User message: "{user_message}"
        
        Context:
        {context_str}
        
        Generate an appropriate response that addresses the user's needs.
        """
    
    def _build_conversation_history(self, history: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """
        Build conversation history for AI context
        """
        messages = []
        
        # Only include recent messages to stay within token limits
        recent_history = history[-6:] if len(history) > 6 else history
        
        for msg in recent_history:
            role = "user" if msg.get("message_type") == "user" else "assistant"
            content = msg.get("content", "")
            
            if content and len(content.strip()) > 0:
                messages.append({
                    "role": role,
                    "content": content
                })
        
        return messages
    
    async def _generate_fallback_response(
        self, 
        context: ResponseContext
    ) -> Dict[str, Any]:
        """
        Generate fallback response when other methods fail
        """
        # Choose appropriate fallback based on context
        if context.missing_entities:
            entity_type = context.missing_entities[0]
            message = f"I need some information to help you better. Could you provide {entity_type.replace('_', ' ')}?"
        elif context.intent == IntentType.UNKNOWN:
            message = random.choice(self.response_templates["fallback"])
        else:
            message = "I'm here to help! Could you tell me more about what you're looking for?"
        
        return {
            "message": message,
            "actions": [
                {
                    "type": "quick_reply",
                    "text": "Browse products",
                    "payload": "browse_products"
                },
                {
                    "type": "quick_reply",
                    "text": "Speak to human",
                    "payload": "escalate_human"
                }
            ],
            "method": "fallback",
            "confidence": 0.6
        }
    
    def _select_best_response(
        self, 
        candidates: List[Dict[str, Any]], 
        context: ResponseContext
    ) -> Dict[str, Any]:
        """
        Select the best response from candidates
        """
        if not candidates:
            return self._create_error_response("No response candidates available")
        
        # Sort by confidence score
        candidates.sort(key=lambda x: x.get("confidence", 0.0), reverse=True)
        
        # Return highest confidence response
        return candidates[0]
    
    async def _enhance_response(
        self, 
        response: Dict[str, Any], 
        context: ResponseContext
    ) -> Dict[str, Any]:
        """
        Enhance response with additional product data, recommendations, etc.
        """
        try:
            enhanced = response.copy()
            
            # Add product information for product-related intents
            if context.intent in [IntentType.PRODUCT_SEARCH, IntentType.PRODUCT_INFO]:
                enhanced = await self._add_product_information(enhanced, context)
            
            # Add order information for order-related intents
            elif context.intent in [IntentType.ORDER_STATUS, IntentType.ORDER_TRACK]:
                enhanced = await self._add_order_information(enhanced, context)
            
            # Add recommendations when appropriate
            if context.intent in [IntentType.PRODUCT_SEARCH, IntentType.PRODUCT_INFO]:
                enhanced = await self._add_recommendations(enhanced, context)
            
            return enhanced
            
        except Exception as e:
            logger.warning(f"Response enhancement failed: {e}")
            return response
    
    async def _add_product_information(
        self, 
        response: Dict[str, Any], 
        context: ResponseContext
    ) -> Dict[str, Any]:
        """
        Add product information to response
        """
        try:
            # Get product data based on entities
            product_info = []
            
            if context.entities.get("product_id"):
                # Fetch specific product by ID
                db = next(get_db())
                try:
                    product = db.query(Product).filter(
                        Product.id == context.entities["product_id"]
                    ).first()
                    
                    if product:
                        product_info.append({
                            "id": str(product.id),
                            "name": product.name,
                            "price": float(product.price),
                            "category": product.category,
                            "description": product.description,
                            "rating": float(product.rating) if product.rating else None,
                            "image_url": product.image_url
                        })
                finally:
                    db.close()
            
            if product_info:
                response["product_data"] = product_info
                response["has_products"] = True
            
            return response
            
        except Exception as e:
            logger.warning(f"Failed to add product information: {e}")
            return response
    
    async def _add_order_information(
        self, 
        response: Dict[str, Any], 
        context: ResponseContext
    ) -> Dict[str, Any]:
        """
        Add order information to response
        """
        try:
            # This would normally fetch order data from order service
            order_number = context.entities.get("order_number")
            
            if order_number:
                # Placeholder order data - would be fetched from order service
                order_info = {
                    "order_number": order_number,
                    "status": "In Transit",
                    "estimated_delivery": "2024-01-15",
                    "tracking_number": f"TRK{order_number[-6:]}"
                }
                
                response["order_data"] = order_info
                response["has_order"] = True
            
            return response
            
        except Exception as e:
            logger.warning(f"Failed to add order information: {e}")
            return response
    
    async def _add_recommendations(
        self, 
        response: Dict[str, Any], 
        context: ResponseContext
    ) -> Dict[str, Any]:
        """
        Add product recommendations to response
        """
        try:
            # Get recommendations based on context
            query = context.entities.get("product_name") or context.entities.get("category")
            
            if query and context.user_id:
                # This would normally call recommendation service
                # recommendations = await self.recommendation_service.get_recommendations(
                #     user_id=context.user_id,
                #     query=query,
                #     limit=3
                # )
                
                # Placeholder recommendations
                recommendations = []
                
                if recommendations:
                    response["recommendations"] = recommendations
                    response["has_recommendations"] = True
            
            return response
            
        except Exception as e:
            logger.warning(f"Failed to add recommendations: {e}")
            return response
    
    def _create_error_response(self, error_message: str) -> Dict[str, Any]:
        """
        Create error response
        """
        return {
            "message": "I apologize, but I'm having trouble processing your request right now. Please try again or contact support.",
            "actions": [
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
            "method": "error",
            "confidence": 0.1,
            "error": error_message,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def personalize_response(
        self, 
        response: Dict[str, Any], 
        user_preferences: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Personalize response based on user preferences
        """
        try:
            personalized = response.copy()
            
            # Adjust tone based on user preferences
            if user_preferences.get("communication_style") == "formal":
                # Make response more formal
                message = personalized["message"]
                personalized["message"] = message.replace("Hi!", "Hello").replace("Hey", "Hello")
            
            # Filter recommendations based on preferences
            if "recommendations" in personalized:
                filtered_recs = []
                for rec in personalized["recommendations"]:
                    # Filter based on price range, categories, etc.
                    if self._matches_user_preferences(rec, user_preferences):
                        filtered_recs.append(rec)
                
                personalized["recommendations"] = filtered_recs
            
            return personalized
            
        except Exception as e:
            logger.warning(f"Response personalization failed: {e}")
            return response
    
    def _matches_user_preferences(
        self, 
        product: Dict[str, Any], 
        preferences: Dict[str, Any]
    ) -> bool:
        """
        Check if product matches user preferences
        """
        # Price range filter
        price_range = preferences.get("price_range", {})
        if price_range:
            min_price = price_range.get("min", 0)
            max_price = price_range.get("max", float('inf'))
            product_price = product.get("price", 0)
            
            if not (min_price <= product_price <= max_price):
                return False
        
        # Category filter
        preferred_categories = preferences.get("categories", [])
        if preferred_categories:
            product_category = product.get("category", "")
            if product_category not in preferred_categories:
                return False
        
        return True
    
    async def get_response_suggestions(
        self, 
        context: ResponseContext
    ) -> List[Dict[str, Any]]:
        """
        Get suggested responses for given context
        """
        suggestions = []
        
        intent = context.intent
        
        if intent == IntentType.PRODUCT_SEARCH:
            suggestions.extend([
                {"text": "Show me popular products", "type": "product_search", "confidence": 0.9},
                {"text": "Browse by category", "type": "browse", "confidence": 0.8},
                {"text": "What's on sale?", "type": "deals", "confidence": 0.7}
            ])
        
        elif intent == IntentType.PRODUCT_INFO:
            suggestions.extend([
                {"text": "Show product specifications", "type": "specs", "confidence": 0.9},
                {"text": "Customer reviews", "type": "reviews", "confidence": 0.8},
                {"text": "Similar products", "type": "similar", "confidence": 0.7}
            ])
        
        elif intent == IntentType.ORDER_STATUS:
            suggestions.extend([
                {"text": "Track my order", "type": "tracking", "confidence": 0.9},
                {"text": "Order details", "type": "details", "confidence": 0.8},
                {"text": "Change delivery address", "type": "modify", "confidence": 0.6}
            ])
        
        # Add common suggestions
        suggestions.extend([
            {"text": "Speak with a human agent", "type": "escalate", "confidence": 0.5},
            {"text": "Start over", "type": "restart", "confidence": 0.4}
        ])
        
        return suggestions[:5]  # Return top 5 suggestions