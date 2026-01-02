import asyncio
from typing import Dict, List, Any, Optional, Tuple
from loguru import logger
import re
import json
from datetime import datetime
import spacy
from dateutil import parser as date_parser

from src.core.config import settings


class EntityExtractionService:
    """
    Named Entity Recognition and extraction service for chatbot
    """
    
    def __init__(self):
        self.nlp = None
        self.openai_client = None
        self.entity_patterns = {}
        self._setup_models()
        self._setup_patterns()
    
    def _setup_models(self):
        """
        Initialize NLP models
        """
        # Setup spaCy model
        try:
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("spaCy model loaded for entity extraction")
        except IOError:
            logger.warning("spaCy model not found. Please install with: python -m spacy download en_core_web_sm")
        except Exception as e:
            logger.warning(f"Failed to load spaCy model: {e}")
        
        # Setup OpenAI
        if settings.openai_api_key:
            try:
                import openai
                self.openai_client = openai.AsyncOpenAI(
                    api_key=settings.openai_api_key
                )
                logger.info("OpenAI entity extraction client initialized")
            except ImportError:
                logger.warning("OpenAI library not available")
    
    def _setup_patterns(self):
        """
        Setup regex patterns for specific entity types
        """
        self.entity_patterns = {
            # E-commerce specific entities
            "order_number": [
                r'(?:order\s*(?:number|#|id)\s*:?\s*)?([A-Z]{2,}\d{6,}|\d{8,})',
                r'(?:order|purchase)\s*(?:#|id)?\s*([A-Z0-9]{6,})',
                r'#([A-Z0-9]{6,})'
            ],
            
            "product_id": [
                r'(?:product\s*(?:id|code|number)\s*:?\s*)?([A-Z]{2,}\d{4,})',
                r'(?:sku|item)\s*(?:#|id)?\s*([A-Z0-9]{4,})',
                r'model\s*(?:number|#)?\s*([A-Z0-9]{3,})'
            ],
            
            "tracking_number": [
                r'(?:tracking\s*(?:number|#|id|code)\s*:?\s*)?([A-Z0-9]{10,})',
                r'(?:ups|fedex|usps|dhl)\s*(?:#|id)?\s*([A-Z0-9]{8,})',
                r'\b(1Z[A-Z0-9]{16})\b'  # UPS tracking pattern
            ],
            
            "price": [
                r'\$(\d+(?:\.\d{2})?)',
                r'(\d+(?:\.\d{2})?)\s*(?:dollars?|usd|$)',
                r'price.*?(\d+(?:\.\d{2})?)',
                r'costs?\s*(\d+(?:\.\d{2})?)'
            ],
            
            "quantity": [
                r'(\d+)\s*(?:pieces?|items?|units?)',
                r'(?:quantity|qty|amount)\s*:?\s*(\d+)',
                r'(\d+)\s*of\s+(?:these?|them|this)'
            ],
            
            "size": [
                r'\b(XS|S|M|L|XL|XXL|XXXL)\b',
                r'size\s*:?\s*(XS|S|M|L|XL|XXL|XXXL|\d+)',
                r'(\d+(?:\.\d+)?)\s*(?:inches?|in|cm|mm|ft|feet)',
                r'(\d+)\s*x\s*(\d+)(?:\s*x\s*(\d+))?'  # Dimensions
            ],
            
            "color": [
                r'\b(red|blue|green|yellow|orange|purple|pink|black|white|gray|grey|brown|navy|maroon|teal|cyan|magenta|silver|gold|beige|tan|khaki)\b',
                r'color\s*:?\s*(\w+)',
                r'in\s+(\w+)\s+color'
            ],
            
            "email": [
                r'\b([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})\b'
            ],
            
            "phone": [
                r'(?:\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})',
                r'(?:\+?1[-.\s]?)?([0-9]{3})[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})',
                r'phone.*?(\+?1?[-.\s]?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4})'
            ],
            
            "zip_code": [
                r'\b(\d{5}(?:-\d{4})?)\b',
                r'zip\s*(?:code)?\s*:?\s*(\d{5}(?:-\d{4})?)'
            ],
            
            "date": [
                r'\b(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\b',
                r'\b(\d{4}[-/]\d{1,2}[-/]\d{1,2})\b',
                r'\b(today|tomorrow|yesterday)\b',
                r'\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b',
                r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}(?:st|nd|rd|th)?\b'
            ],
            
            "time": [
                r'\b(\d{1,2}:\d{2}(?::\d{2})?\s*(?:am|pm|AM|PM)?)\b',
                r'\b(\d{1,2}\s*(?:am|pm|AM|PM))\b'
            ]
        }
    
    async def extract_entities(
        self, 
        text: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Extract entities from text using multiple approaches
        """
        if not text or len(text.strip()) < 2:
            return {"entities": {}, "confidence": 0.0}
        
        try:
            # Combine results from different extraction methods
            entities = {}
            
            # 1. Pattern-based extraction (fast and accurate for known patterns)
            pattern_entities = await self._extract_with_patterns(text)
            entities.update(pattern_entities)
            
            # 2. spaCy NER (good for general entities)
            if self.nlp:
                try:
                    spacy_entities = await self._extract_with_spacy(text)
                    entities.update(spacy_entities)
                except Exception as e:
                    logger.warning(f"spaCy entity extraction failed: {e}")
            
            # 3. OpenAI extraction for complex cases
            if self.openai_client and len(entities) < 3:
                try:
                    openai_entities = await self._extract_with_openai(text, context)
                    # Merge OpenAI results carefully to avoid overwriting good pattern matches
                    for key, value in openai_entities.items():
                        if key not in entities or not entities[key]:
                            entities[key] = value
                except Exception as e:
                    logger.warning(f"OpenAI entity extraction failed: {e}")
            
            # Post-process and validate entities
            validated_entities = await self._validate_entities(entities, text)
            
            # Calculate overall confidence
            confidence = self._calculate_extraction_confidence(validated_entities, text)
            
            return {
                "entities": validated_entities,
                "confidence": confidence,
                "extraction_methods": ["patterns", "spacy", "openai"] if self.openai_client else ["patterns", "spacy"],
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            return {"entities": {}, "confidence": 0.0, "error": str(e)}
    
    async def _extract_with_patterns(self, text: str) -> Dict[str, List[str]]:
        """
        Extract entities using regex patterns
        """
        entities = {}
        
        for entity_type, patterns in self.entity_patterns.items():
            matches = []
            
            for pattern in patterns:
                found = re.finditer(pattern, text, re.IGNORECASE)
                for match in found:
                    if match.groups():
                        # If pattern has groups, use the groups
                        for group in match.groups():
                            if group and group.strip():
                                matches.append(group.strip())
                    else:
                        # If no groups, use the full match
                        matches.append(match.group().strip())
            
            if matches:
                # Remove duplicates while preserving order
                entities[entity_type] = list(dict.fromkeys(matches))
        
        return entities
    
    async def _extract_with_spacy(self, text: str) -> Dict[str, List[str]]:
        """
        Extract entities using spaCy NER
        """
        def _extract():
            doc = self.nlp(text)
            entities = {}
            
            for ent in doc.ents:
                entity_type = self._map_spacy_label(ent.label_)
                if entity_type:
                    if entity_type not in entities:
                        entities[entity_type] = []
                    entities[entity_type].append(ent.text.strip())
            
            return entities
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _extract)
    
    def _map_spacy_label(self, spacy_label: str) -> Optional[str]:
        """
        Map spaCy entity labels to our domain-specific labels
        """
        mapping = {
            "PERSON": "person_name",
            "ORG": "organization",
            "GPE": "location",
            "LOC": "location",
            "MONEY": "price",
            "DATE": "date",
            "TIME": "time",
            "CARDINAL": "quantity",
            "ORDINAL": "quantity",
            "PRODUCT": "product_name"
        }
        return mapping.get(spacy_label)
    
    async def _extract_with_openai(
        self, 
        text: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, List[str]]:
        """
        Extract entities using OpenAI GPT
        """
        context_info = ""
        if context:
            context_info = f"Context: {json.dumps(context, indent=2)}\n"
        
        prompt = f"""
        Extract relevant entities from this e-commerce customer message:

        {context_info}Message: "{text}"

        Find these types of entities if present:
        - product_name: Names of products mentioned
        - order_number: Order IDs or numbers
        - product_id: Product IDs, SKUs, or model numbers
        - tracking_number: Shipping tracking numbers
        - price: Monetary amounts
        - quantity: Amounts or quantities
        - size: Product sizes
        - color: Colors mentioned
        - date: Dates or time references
        - location: Places, addresses, or shipping locations
        - person_name: Names of people
        - email: Email addresses
        - phone: Phone numbers

        Respond with a JSON object where keys are entity types and values are arrays of found entities.
        Only include entities that are clearly present. If no entities found, return empty object.

        JSON response:
        """
        
        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert entity extractor for e-commerce conversations. Always respond with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
                temperature=0.1
            )
            
            result_text = response.choices[0].message.content.strip()
            
            # Parse JSON response
            try:
                entities = json.loads(result_text)
                # Ensure all values are lists
                for key, value in entities.items():
                    if not isinstance(value, list):
                        entities[key] = [str(value)] if value else []
                return entities
                
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse OpenAI entity response: {result_text}")
                return {}
                
        except Exception as e:
            logger.error(f"OpenAI entity extraction error: {e}")
            raise
    
    async def _validate_entities(
        self, 
        entities: Dict[str, List[str]], 
        original_text: str
    ) -> Dict[str, List[str]]:
        """
        Validate and clean extracted entities
        """
        validated = {}
        
        for entity_type, values in entities.items():
            validated_values = []
            
            for value in values:
                if not value or len(value.strip()) < 1:
                    continue
                
                cleaned_value = value.strip()
                
                # Apply type-specific validation
                if entity_type == "email":
                    if "@" in cleaned_value and "." in cleaned_value:
                        validated_values.append(cleaned_value)
                
                elif entity_type == "price":
                    # Ensure price is a valid number
                    try:
                        price_num = float(re.sub(r'[^\d.]', '', cleaned_value))
                        if 0 < price_num < 1000000:  # Reasonable price range
                            validated_values.append(f"${price_num:.2f}")
                    except ValueError:
                        pass
                
                elif entity_type == "phone":
                    # Clean and format phone number
                    digits = re.sub(r'\D', '', cleaned_value)
                    if len(digits) == 10:
                        formatted = f"({digits[:3]}) {digits[3:6]}-{digits[6:]}"
                        validated_values.append(formatted)
                    elif len(digits) == 11 and digits[0] == '1':
                        formatted = f"({digits[1:4]}) {digits[4:7]}-{digits[7:]}"
                        validated_values.append(formatted)
                
                elif entity_type == "date":
                    # Validate and normalize dates
                    try:
                        parsed_date = date_parser.parse(cleaned_value, fuzzy=True)
                        validated_values.append(parsed_date.strftime("%Y-%m-%d"))
                    except (ValueError, TypeError):
                        # Keep original if parsing fails but looks like a date
                        if re.search(r'\d', cleaned_value):
                            validated_values.append(cleaned_value)
                
                elif entity_type == "order_number":
                    # Order numbers should be alphanumeric and reasonable length
                    if re.match(r'^[A-Z0-9]{6,20}$', cleaned_value.upper()):
                        validated_values.append(cleaned_value.upper())
                
                elif entity_type == "quantity":
                    # Ensure quantity is a valid number
                    try:
                        qty = int(re.sub(r'\D', '', cleaned_value))
                        if 0 < qty < 10000:  # Reasonable quantity range
                            validated_values.append(str(qty))
                    except ValueError:
                        pass
                
                else:
                    # For other types, just clean and validate length
                    if 1 <= len(cleaned_value) <= 200:
                        validated_values.append(cleaned_value)
            
            if validated_values:
                # Remove duplicates while preserving order
                validated[entity_type] = list(dict.fromkeys(validated_values))
        
        return validated
    
    def _calculate_extraction_confidence(
        self, 
        entities: Dict[str, List[str]], 
        text: str
    ) -> float:
        """
        Calculate confidence score for entity extraction
        """
        if not entities:
            return 0.0
        
        # Base confidence from number of entities found
        base_confidence = min(0.8, len(entities) * 0.2)
        
        # Boost for high-value entities
        high_value_entities = ["order_number", "product_id", "tracking_number", "email"]
        for entity_type in high_value_entities:
            if entity_type in entities and entities[entity_type]:
                base_confidence += 0.1
        
        # Reduce if text is very short
        if len(text.split()) < 5:
            base_confidence *= 0.8
        
        return min(1.0, base_confidence)
    
    async def get_entity_suggestions(
        self, 
        text: str, 
        entity_type: str
    ) -> List[str]:
        """
        Get suggestions for a specific entity type
        """
        if entity_type not in self.entity_patterns:
            return []
        
        suggestions = []
        patterns = self.entity_patterns[entity_type]
        
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                if match.groups():
                    for group in match.groups():
                        if group and group.strip():
                            suggestions.append(group.strip())
                else:
                    suggestions.append(match.group().strip())
        
        return list(dict.fromkeys(suggestions))  # Remove duplicates
    
    async def extract_context_entities(
        self, 
        conversation_history: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Extract entities from conversation history to build context
        """
        all_entities = {}
        
        for message in conversation_history:
            if message.get("message_type") == "user":
                result = await self.extract_entities(message.get("content", ""))
                
                # Merge entities
                for entity_type, values in result.get("entities", {}).items():
                    if entity_type not in all_entities:
                        all_entities[entity_type] = []
                    
                    # Add new values that aren't already present
                    for value in values:
                        if value not in all_entities[entity_type]:
                            all_entities[entity_type].append(value)
        
        return {
            "entities": all_entities,
            "entity_count": sum(len(values) for values in all_entities.values()),
            "most_recent_entities": self._get_most_recent_entities(conversation_history)
        }
    
    def _get_most_recent_entities(
        self, 
        conversation_history: List[Dict[str, Any]]
    ) -> Dict[str, str]:
        """
        Get the most recently mentioned entity of each type
        """
        recent_entities = {}
        
        # Iterate in reverse to get most recent first
        for message in reversed(conversation_history):
            if message.get("message_type") == "user":
                entities = message.get("extracted_entities", {})
                for entity_type, values in entities.items():
                    if entity_type not in recent_entities and values:
                        recent_entities[entity_type] = values[0]
        
        return recent_entities