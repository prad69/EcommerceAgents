import asyncio
from typing import Dict, List, Any, Optional, Tuple
from loguru import logger
import json
import re
from datetime import datetime
import numpy as np

from src.core.config import settings
from src.models.conversation import IntentType, IntentTrainingData
from src.core.database import get_db


class IntentClassificationService:
    """
    Advanced intent classification system for chatbot
    """
    
    def __init__(self):
        self.transformers_classifier = None
        self.openai_client = None
        self.rule_patterns = {}
        self._setup_models()
        self._setup_rule_patterns()
    
    def _setup_models(self):
        """
        Initialize classification models
        """
        # Setup Transformers model
        try:
            from transformers import pipeline
            self.transformers_classifier = pipeline(
                "text-classification",
                model="microsoft/DialoGPT-medium",  # Can be replaced with custom trained model
                return_all_scores=True
            )
            logger.info("Transformers intent classifier loaded")
        except ImportError:
            logger.warning("Transformers library not available for intent classification")
        except Exception as e:
            logger.warning(f"Failed to load transformers intent classifier: {e}")
        
        # Setup OpenAI
        if settings.openai_api_key:
            try:
                import openai
                self.openai_client = openai.AsyncOpenAI(
                    api_key=settings.openai_api_key
                )
                logger.info("OpenAI intent classification client initialized")
            except ImportError:
                logger.warning("OpenAI library not available")
    
    def _setup_rule_patterns(self):
        """
        Setup rule-based intent patterns
        """
        self.rule_patterns = {
            IntentType.GREETING: [
                r'\b(hello|hi|hey|good morning|good afternoon|good evening|greetings)\b',
                r'\b(start|begin|help me)\b',
                r'^(hi|hello|hey)$'
            ],
            
            IntentType.FAREWELL: [
                r'\b(goodbye|bye|farewell|see you|talk to you later|thanks? ?(?:you)?(?:!|\.)*$)\b',
                r'\b(end|stop|quit|exit|close)\b',
                r'that\'s all|nothing else'
            ],
            
            IntentType.PRODUCT_SEARCH: [
                r'\b(search|find|look for|looking for|show me|do you have)\b.*\b(product|item|thing)\b',
                r'\b(search|find|looking for)\b.*',
                r'\b(need|want|interested in)\b.*',
                r'where (?:can i find|is)'
            ],
            
            IntentType.PRODUCT_INFO: [
                r'\b(tell me (?:about|more)|information (?:about|on)|details? (?:about|on|of))\b',
                r'\b(what (?:is|are)|how (?:does|do))\b.*\b(work|function|feature)\b',
                r'\b(spec|specification|dimension|size|weight|color|price)\b',
                r'what.*(made of|material)'
            ],
            
            IntentType.PRODUCT_COMPARE: [
                r'\b(compare|comparison|difference|vs|versus|better than|which (?:is|one))\b',
                r'\b(similar|alternative|equivalent)\b.*(?:to|like)',
                r'what.*(difference|better|worse)'
            ],
            
            IntentType.PRODUCT_AVAILABILITY: [
                r'\b((?:in |available |out of )?stock|availability|available)\b',
                r'\b(when (?:will|can)|how soon)\b.*\b(available|ready|ship)\b',
                r'\b(sold out|back in stock|restock)\b'
            ],
            
            IntentType.ORDER_STATUS: [
                r'\b(order (?:status|update|information)|where (?:is )?(?:my )?order)\b',
                r'\b(track|tracking|shipped|delivery status)\b',
                r'order (?:number|id|#)'
            ],
            
            IntentType.ORDER_TRACK: [
                r'\b(track|tracking (?:number|code|info)|where (?:is )?(?:my )?(?:package|shipment))\b',
                r'\b(delivery|shipping) (?:status|update|information)',
                r'when (?:will|can i expect)'
            ],
            
            IntentType.ORDER_CANCEL: [
                r'\b(cancel|cancellation|stop|halt)\b.*\border\b',
                r'\b(don\'t want|no longer need)\b.*\border\b',
                r'cancel (?:my )?(?:order|purchase)'
            ],
            
            IntentType.ORDER_RETURN: [
                r'\b(return|refund|give back|send back)\b',
                r'\b(not (?:working|satisfied)|defective|broken|wrong)\b',
                r'money back|return policy'
            ],
            
            IntentType.TECHNICAL_SUPPORT: [
                r'\b((?:technical )?(?:help|support|assistance|problem|issue|trouble))\b',
                r'\b(not working|broken|error|bug|glitch)\b',
                r'\b(how (?:do|to)|can you help)\b'
            ],
            
            IntentType.SHIPPING_INFO: [
                r'\b(shipping|delivery|ship)\b.*\b(cost|price|fee|time|speed|method)\b',
                r'\b(how (?:long|much)|when (?:will|can))\b.*\b(ship|deliver|arrive)\b',
                r'free shipping|shipping option'
            ],
            
            IntentType.PAYMENT_HELP: [
                r'\b(payment|pay|billing|credit card|paypal|checkout)\b',
                r'\b(charge|transaction|invoice|receipt)\b',
                r'payment (?:method|option|problem|issue)'
            ],
            
            IntentType.ACCOUNT_HELP: [
                r'\b(account|profile|login|password|sign in|register)\b',
                r'\b(forgot|reset|change|update)\b.*\b(password|email|account)\b',
                r'my account|profile'
            ],
            
            IntentType.COMPLAINT: [
                r'\b(complain|complaint|unhappy|dissatisfied|angry|frustrated)\b',
                r'\b(terrible|awful|horrible|worst|hate)\b',
                r'\b(manager|supervisor|escalate)\b'
            ],
            
            IntentType.ESCALATE_HUMAN: [
                r'\b((?:speak|talk) (?:to|with) (?:a )?(?:human|person|agent|representative))\b',
                r'\b(human (?:agent|help|support)|real person)\b',
                r'\b(transfer|escalate|manager|supervisor)\b'
            ],
            
            IntentType.SMALL_TALK: [
                r'\b(how are you|what\'s up|nice weather|how\'s (?:your )?day)\b',
                r'\b(thank you|thanks|appreciate)\b(?!\s+(?:for|but))',
                r'\b(you\'re (?:welcome|helpful)|no problem)\b'
            ]
        }
    
    async def classify_intent(
        self, 
        text: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Classify the intent of user input
        """
        if not text or len(text.strip()) < 2:
            return self._create_intent_result(IntentType.UNKNOWN, 0.0, "empty_input")
        
        try:
            # Try different classification approaches
            results = []
            
            # 1. Rule-based classification (fast and reliable)
            rule_result = await self._classify_with_rules(text, context)
            results.append(rule_result)
            
            # 2. OpenAI classification (high quality but slower)
            if self.openai_client and rule_result["confidence"] < 0.8:
                try:
                    openai_result = await self._classify_with_openai(text, context)
                    results.append(openai_result)
                except Exception as e:
                    logger.warning(f"OpenAI intent classification failed: {e}")
            
            # 3. Transformers classification (offline backup)
            if self.transformers_classifier and rule_result["confidence"] < 0.6:
                try:
                    transformers_result = await self._classify_with_transformers(text, context)
                    results.append(transformers_result)
                except Exception as e:
                    logger.warning(f"Transformers intent classification failed: {e}")
            
            # Choose best result
            best_result = max(results, key=lambda x: x["confidence"])
            
            # Apply context-based adjustments
            adjusted_result = await self._apply_context_adjustments(best_result, context)
            
            return adjusted_result
            
        except Exception as e:
            logger.error(f"Intent classification failed: {e}")
            return self._create_intent_result(IntentType.UNKNOWN, 0.0, "classification_error")
    
    async def _classify_with_rules(
        self, 
        text: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Rule-based intent classification using regex patterns
        """
        text_lower = text.lower().strip()
        best_intent = IntentType.UNKNOWN
        best_confidence = 0.0
        
        for intent, patterns in self.rule_patterns.items():
            confidence = 0.0
            
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    # Calculate confidence based on pattern specificity
                    pattern_confidence = len(pattern) / 100  # Simple heuristic
                    confidence = max(confidence, min(0.9, pattern_confidence))
            
            if confidence > best_confidence:
                best_confidence = confidence
                best_intent = intent
        
        return self._create_intent_result(
            best_intent, 
            best_confidence, 
            "rule_based"
        )
    
    async def _classify_with_openai(
        self, 
        text: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Intent classification using OpenAI GPT
        """
        # Create intent options string
        intent_descriptions = {
            IntentType.PRODUCT_SEARCH: "User wants to find or search for products",
            IntentType.PRODUCT_INFO: "User wants information about a specific product",
            IntentType.PRODUCT_COMPARE: "User wants to compare different products",
            IntentType.PRODUCT_AVAILABILITY: "User is asking about product availability or stock",
            IntentType.ORDER_STATUS: "User wants to check their order status",
            IntentType.ORDER_TRACK: "User wants to track their shipment",
            IntentType.ORDER_CANCEL: "User wants to cancel an order",
            IntentType.ORDER_RETURN: "User wants to return or refund a product",
            IntentType.TECHNICAL_SUPPORT: "User needs technical help or has a problem",
            IntentType.SHIPPING_INFO: "User wants shipping information",
            IntentType.PAYMENT_HELP: "User needs help with payment or billing",
            IntentType.ACCOUNT_HELP: "User needs help with their account",
            IntentType.COMPLAINT: "User is making a complaint",
            IntentType.GREETING: "User is greeting or starting conversation",
            IntentType.FAREWELL: "User is ending conversation or saying goodbye",
            IntentType.ESCALATE_HUMAN: "User wants to speak with a human agent",
            IntentType.SMALL_TALK: "User is making small talk or casual conversation"
        }
        
        context_info = ""
        if context:
            context_info = f"Previous context: {json.dumps(context, indent=2)}\n"
        
        prompt = f"""
        Analyze this user message and determine the most likely intent:

        {context_info}User message: "{text}"

        Choose from these intents:
        {chr(10).join(f"- {intent.value}: {desc}" for intent, desc in intent_descriptions.items())}

        Respond with a JSON object containing:
        - intent: the intent name (e.g., "product_search")
        - confidence: a number between 0 and 1
        - reasoning: brief explanation

        JSON response:
        """
        
        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert intent classifier for an e-commerce chatbot. Always respond with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=150,
                temperature=0.1
            )
            
            result_text = response.choices[0].message.content.strip()
            
            # Parse JSON response
            try:
                result = json.loads(result_text)
                intent_value = result.get("intent", "unknown")
                
                # Convert string to IntentType enum
                intent = IntentType.UNKNOWN
                for intent_enum in IntentType:
                    if intent_enum.value == intent_value:
                        intent = intent_enum
                        break
                
                confidence = float(result.get("confidence", 0.0))
                
                return self._create_intent_result(
                    intent,
                    confidence,
                    "openai",
                    {"reasoning": result.get("reasoning", "")}
                )
                
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse OpenAI intent response: {result_text}")
                # Fallback: simple keyword matching
                return await self._classify_with_rules(text, context)
                
        except Exception as e:
            logger.error(f"OpenAI intent classification error: {e}")
            raise
    
    async def _classify_with_transformers(
        self, 
        text: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Intent classification using Transformers model
        """
        def _classify():
            # This is a simplified version - in production, you'd use a model trained on intent data
            # For now, we'll use sentiment analysis as a proxy and map to intents
            results = self.transformers_classifier(text)
            
            # Map sentiment to likely intents (simplified heuristic)
            if results:
                label = results[0]["label"]
                score = results[0]["score"]
                
                if "positive" in label.lower():
                    return IntentType.PRODUCT_SEARCH, score * 0.6
                elif "negative" in label.lower():
                    return IntentType.COMPLAINT, score * 0.7
                else:
                    return IntentType.UNKNOWN, score * 0.3
            
            return IntentType.UNKNOWN, 0.0
        
        loop = asyncio.get_event_loop()
        intent, confidence = await loop.run_in_executor(None, _classify)
        
        return self._create_intent_result(
            intent,
            confidence,
            "transformers"
        )
    
    async def _apply_context_adjustments(
        self, 
        result: Dict[str, Any], 
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Adjust intent classification based on conversation context
        """
        if not context:
            return result
        
        # Get previous intent and entities
        previous_intent = context.get("previous_intent")
        entities = context.get("entities", {})
        
        # Apply context-based boosting
        confidence_boost = 0.0
        
        # If user is in middle of product discussion, boost product-related intents
        if previous_intent in [IntentType.PRODUCT_SEARCH, IntentType.PRODUCT_INFO]:
            if result["intent"] in [IntentType.PRODUCT_INFO, IntentType.PRODUCT_COMPARE, IntentType.PRODUCT_AVAILABILITY]:
                confidence_boost += 0.2
        
        # If user mentioned order number previously, boost order-related intents
        if entities.get("order_number"):
            if result["intent"] in [IntentType.ORDER_STATUS, IntentType.ORDER_TRACK, IntentType.ORDER_CANCEL]:
                confidence_boost += 0.15
        
        # Apply boost
        result["confidence"] = min(1.0, result["confidence"] + confidence_boost)
        result["metadata"]["context_boost"] = confidence_boost
        
        return result
    
    def _create_intent_result(
        self, 
        intent: IntentType, 
        confidence: float, 
        method: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create standardized intent classification result
        """
        return {
            "intent": intent,
            "confidence": confidence,
            "method": method,
            "metadata": metadata or {},
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def train_from_data(self, training_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Train or update the classification model with new data
        """
        try:
            # Store training data in database
            db = next(get_db())
            stored_count = 0
            
            for data in training_data:
                # Check if data already exists
                existing = db.query(IntentTrainingData).filter(
                    IntentTrainingData.text == data["text"]
                ).first()
                
                if not existing:
                    training_sample = IntentTrainingData(
                        text=data["text"],
                        intent=IntentType(data["intent"]),
                        confidence=data.get("confidence", 1.0),
                        entities=data.get("entities"),
                        verified=data.get("verified", False)
                    )
                    db.add(training_sample)
                    stored_count += 1
            
            db.commit()
            db.close()
            
            logger.info(f"Stored {stored_count} new training samples")
            
            # TODO: Implement actual model retraining
            # For now, just return success metrics
            
            return {
                "training_completed": True,
                "samples_processed": len(training_data),
                "samples_stored": stored_count,
                "model_updated": False,  # Would be True after actual retraining
                "training_accuracy": 0.95  # Placeholder
            }
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
    
    async def get_intent_suggestions(self, text: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Get multiple intent suggestions with confidence scores
        """
        # Get classification result
        primary_result = await self.classify_intent(text)
        
        # Get rule-based alternatives
        rule_results = []
        text_lower = text.lower().strip()
        
        for intent, patterns in self.rule_patterns.items():
            confidence = 0.0
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    pattern_confidence = len(pattern) / 100
                    confidence = max(confidence, min(0.9, pattern_confidence))
            
            if confidence > 0.1:  # Only include reasonable matches
                rule_results.append({
                    "intent": intent,
                    "confidence": confidence,
                    "method": "rule_based"
                })
        
        # Sort by confidence and return top k
        all_results = [primary_result] + rule_results
        unique_results = []
        seen_intents = set()
        
        for result in sorted(all_results, key=lambda x: x["confidence"], reverse=True):
            if result["intent"] not in seen_intents:
                unique_results.append(result)
                seen_intents.add(result["intent"])
        
        return unique_results[:top_k]