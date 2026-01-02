import asyncio
from typing import Dict, List, Any, Optional, Tuple
from loguru import logger
import json
import re
from datetime import datetime
import numpy as np
from dataclasses import dataclass, field

from src.core.config import settings
from src.core.database import get_db, redis_client
from src.models.conversation import KnowledgeBase, IntentType
from src.models.product import Product


@dataclass
class KnowledgeEntry:
    """Structured knowledge base entry"""
    id: str
    title: str
    content: str
    summary: str
    category: str
    tags: List[str] = field(default_factory=list)
    applicable_intents: List[IntentType] = field(default_factory=list)
    priority: int = 0
    context_required: Dict[str, Any] = field(default_factory=dict)
    effectiveness_score: float = 0.5
    embedding_vector: Optional[List[float]] = None


class KnowledgeBaseService:
    """
    RAG (Retrieval-Augmented Generation) knowledge base service
    """
    
    def __init__(self):
        self.openai_client = None
        self.knowledge_cache = {}
        self.embedding_cache = {}
        self.faq_data = {}
        self._setup_models()
        self._load_default_knowledge()
    
    def _setup_models(self):
        """
        Initialize embedding models and services
        """
        # Setup OpenAI for embeddings
        if settings.openai_api_key:
            try:
                import openai
                self.openai_client = openai.AsyncOpenAI(
                    api_key=settings.openai_api_key
                )
                logger.info("OpenAI embeddings client initialized")
            except ImportError:
                logger.warning("OpenAI library not available")
    
    def _load_default_knowledge(self):
        """
        Load default knowledge base entries
        """
        self.faq_data = {
            # Product-related FAQs
            "product_search_tips": {
                "title": "How to Search for Products",
                "content": """
                To find products quickly:
                1. Use specific product names or model numbers
                2. Browse by category (Electronics, Clothing, Home & Garden, etc.)
                3. Use filters for price, brand, rating, and features
                4. Check our "Popular Products" and "Best Sellers" sections
                5. Use search suggestions that appear as you type
                
                Pro tips:
                - Be specific with your search terms
                - Try different keywords if you don't find what you want
                - Use our advanced search for detailed filtering
                """,
                "category": "product_help",
                "intents": [IntentType.PRODUCT_SEARCH],
                "tags": ["search", "products", "tips", "help"]
            },
            
            "product_information": {
                "title": "Understanding Product Information",
                "content": """
                Each product page includes:
                - Detailed description and specifications
                - High-quality images and videos
                - Customer reviews and ratings
                - Price and availability information
                - Shipping options and delivery times
                - Related and recommended products
                
                Look for these key details:
                - Product dimensions and weight
                - Material composition
                - Warranty information
                - Compatible accessories
                - Care instructions
                """,
                "category": "product_help",
                "intents": [IntentType.PRODUCT_INFO],
                "tags": ["product", "information", "details", "specifications"]
            },
            
            # Order-related FAQs
            "order_tracking": {
                "title": "How to Track Your Order",
                "content": """
                To track your order:
                1. Use your order confirmation number
                2. Check your email for tracking updates
                3. Visit the "My Orders" section in your account
                4. Use our order tracking tool with your order number
                
                Order statuses:
                - Confirmed: Order received and being processed
                - Preparing: Items being picked and packed
                - Shipped: Package in transit
                - Out for Delivery: Package with delivery carrier
                - Delivered: Package delivered to specified address
                
                Delivery typically takes 2-5 business days depending on your location.
                """,
                "category": "order_help",
                "intents": [IntentType.ORDER_TRACK, IntentType.ORDER_STATUS],
                "tags": ["order", "tracking", "shipping", "delivery"]
            },
            
            "order_changes": {
                "title": "Modifying or Canceling Orders",
                "content": """
                Order modifications:
                - Orders can be modified within 1 hour of placement
                - Changes include quantity, shipping address, payment method
                - Contact customer service for urgent modifications
                
                Cancellation policy:
                - Cancel within 2 hours for full refund
                - After shipping: return process applies
                - Digital products: 48-hour cancellation window
                
                To cancel or modify:
                1. Go to "My Orders" in your account
                2. Find the order and click "Modify" or "Cancel"
                3. Follow the prompts to make changes
                4. Confirmation email will be sent
                """,
                "category": "order_help",
                "intents": [IntentType.ORDER_CANCEL],
                "tags": ["order", "cancel", "modify", "change", "refund"]
            },
            
            # Return and refund FAQs
            "return_policy": {
                "title": "Return and Refund Policy",
                "content": """
                Return window: 30 days from delivery date
                
                Returnable items:
                - New, unused items in original packaging
                - Items with tags and labels attached
                - Electronics in original condition with accessories
                
                Non-returnable items:
                - Personal care products
                - Customized or personalized items
                - Digital downloads
                - Perishable goods
                
                Return process:
                1. Log into your account and go to "Returns"
                2. Select the order and items to return
                3. Choose return reason and preferred refund method
                4. Print return label and package items
                5. Drop off at designated location or schedule pickup
                
                Refunds processed within 5-7 business days after we receive the return.
                """,
                "category": "returns",
                "intents": [IntentType.ORDER_RETURN],
                "tags": ["return", "refund", "exchange", "policy"]
            },
            
            # Technical support FAQs
            "account_help": {
                "title": "Account and Login Help",
                "content": """
                Creating an account:
                - Faster checkout process
                - Order history and tracking
                - Personalized recommendations
                - Exclusive member offers
                
                Login issues:
                - Check email and password spelling
                - Use "Forgot Password" to reset
                - Clear browser cache and cookies
                - Try incognito/private browsing mode
                
                Account management:
                - Update profile information in "Account Settings"
                - Manage shipping addresses and payment methods
                - Set communication preferences
                - View order history and download receipts
                
                Password requirements:
                - At least 8 characters
                - Include uppercase and lowercase letters
                - Include at least one number
                - Include at least one special character
                """,
                "category": "account",
                "intents": [IntentType.ACCOUNT_HELP],
                "tags": ["account", "login", "password", "profile"]
            },
            
            "payment_help": {
                "title": "Payment Options and Issues",
                "content": """
                Accepted payment methods:
                - Credit cards (Visa, MasterCard, American Express)
                - Debit cards
                - PayPal
                - Apple Pay and Google Pay
                - Gift cards and store credit
                
                Payment security:
                - SSL encryption for all transactions
                - PCI DSS compliant payment processing
                - No card information stored on our servers
                - Fraud protection and monitoring
                
                Common payment issues:
                - Declined card: Contact your bank
                - Expired card: Update payment method
                - Billing address mismatch: Verify address
                - Insufficient funds: Use different payment method
                
                Payment problems? Contact customer service for immediate assistance.
                """,
                "category": "payment",
                "intents": [IntentType.PAYMENT_HELP],
                "tags": ["payment", "billing", "credit card", "paypal"]
            },
            
            # Shipping information
            "shipping_info": {
                "title": "Shipping Options and Information",
                "content": """
                Shipping options:
                - Standard (5-7 business days): FREE on orders over $35
                - Express (2-3 business days): $7.99
                - Next Day (1 business day): $15.99
                - Same Day (selected cities): $9.99
                
                Shipping locations:
                - Domestic shipping to all 50 states
                - International shipping to 25+ countries
                - APO/FPO addresses supported
                - P.O. Box delivery available
                
                Processing time:
                - Most orders ship within 1-2 business days
                - Custom orders: 3-5 business days
                - Bulk orders: 5-7 business days
                
                Shipping notifications:
                - Order confirmation email
                - Shipping confirmation with tracking
                - Delivery confirmation
                - SMS alerts available
                """,
                "category": "shipping",
                "intents": [IntentType.SHIPPING_INFO],
                "tags": ["shipping", "delivery", "freight", "express"]
            },
            
            # General store information
            "store_hours": {
                "title": "Store Hours and Contact Information",
                "content": """
                Online store: Available 24/7
                
                Customer service hours:
                - Monday-Friday: 8 AM - 8 PM EST
                - Saturday-Sunday: 9 AM - 6 PM EST
                - Holidays: 10 AM - 4 PM EST
                
                Contact methods:
                - Live chat (fastest response)
                - Email: support@store.com
                - Phone: 1-800-STORE-01
                - Social media: @StoreSupport
                
                Response times:
                - Live chat: Immediate during business hours
                - Email: Within 24 hours
                - Phone: Average wait time 2-3 minutes
                
                For urgent issues outside business hours, use live chat or email.
                """,
                "category": "contact",
                "intents": [IntentType.ESCALATE_HUMAN],
                "tags": ["hours", "contact", "support", "phone", "email"]
            }
        }
    
    async def search_knowledge(
        self,
        query: str,
        intent: Optional[IntentType] = None,
        context: Optional[Dict[str, Any]] = None,
        limit: int = 5
    ) -> List[KnowledgeEntry]:
        """
        Search knowledge base for relevant entries
        """
        try:
            # Get all relevant entries
            relevant_entries = []
            
            # 1. Search database entries
            db_entries = await self._search_database_knowledge(query, intent, limit)
            relevant_entries.extend(db_entries)
            
            # 2. Search FAQ data
            faq_entries = await self._search_faq_knowledge(query, intent)
            relevant_entries.extend(faq_entries)
            
            # 3. Search product information if product-related
            if intent in [IntentType.PRODUCT_SEARCH, IntentType.PRODUCT_INFO]:
                product_entries = await self._search_product_knowledge(query, context)
                relevant_entries.extend(product_entries)
            
            # Score and rank entries
            scored_entries = await self._score_knowledge_entries(relevant_entries, query, intent, context)
            
            # Return top entries
            return scored_entries[:limit]
            
        except Exception as e:
            logger.error(f"Knowledge search failed: {e}")
            return []
    
    async def _search_database_knowledge(
        self,
        query: str,
        intent: Optional[IntentType],
        limit: int
    ) -> List[KnowledgeEntry]:
        """
        Search knowledge base entries in database
        """
        try:
            db = next(get_db())
            
            # Build query filters
            query_filters = []
            
            # Text search in title and content
            if query:
                query_filters.append(
                    KnowledgeBase.title.ilike(f"%{query}%") |
                    KnowledgeBase.content.ilike(f"%{query}%") |
                    KnowledgeBase.summary.ilike(f"%{query}%")
                )
            
            # Intent filter
            if intent:
                query_filters.append(
                    KnowledgeBase.applicable_intents.contains([intent.value])
                )
            
            # Execute query
            kb_query = db.query(KnowledgeBase)
            if query_filters:
                kb_query = kb_query.filter(*query_filters)
            
            db_results = kb_query.order_by(
                KnowledgeBase.priority.desc(),
                KnowledgeBase.effectiveness_score.desc()
            ).limit(limit).all()
            
            # Convert to KnowledgeEntry objects
            entries = []
            for kb_entry in db_results:
                entry = KnowledgeEntry(
                    id=str(kb_entry.id),
                    title=kb_entry.title,
                    content=kb_entry.content,
                    summary=kb_entry.summary or "",
                    category=kb_entry.category or "general",
                    tags=kb_entry.tags or [],
                    applicable_intents=[IntentType(intent) for intent in (kb_entry.applicable_intents or [])],
                    priority=kb_entry.priority,
                    context_required=kb_entry.context_required or {},
                    effectiveness_score=kb_entry.effectiveness_score
                )
                entries.append(entry)
            
            db.close()
            return entries
            
        except Exception as e:
            logger.warning(f"Database knowledge search failed: {e}")
            return []
    
    async def _search_faq_knowledge(
        self,
        query: str,
        intent: Optional[IntentType]
    ) -> List[KnowledgeEntry]:
        """
        Search FAQ knowledge entries
        """
        relevant_entries = []
        
        for key, faq in self.faq_data.items():
            # Check intent match
            if intent and intent in faq.get("intents", []):
                score = 0.9
            else:
                score = 0.0
            
            # Check text match
            query_lower = query.lower()
            title_lower = faq["title"].lower()
            content_lower = faq["content"].lower()
            tags_text = " ".join(faq.get("tags", [])).lower()
            
            if query_lower in title_lower:
                score += 0.8
            elif query_lower in content_lower:
                score += 0.6
            elif any(tag in query_lower for tag in faq.get("tags", [])):
                score += 0.4
            elif any(word in title_lower or word in tags_text for word in query_lower.split()):
                score += 0.3
            
            if score > 0.3:  # Minimum relevance threshold
                entry = KnowledgeEntry(
                    id=key,
                    title=faq["title"],
                    content=faq["content"],
                    summary=faq["title"],
                    category=faq.get("category", "faq"),
                    tags=faq.get("tags", []),
                    applicable_intents=faq.get("intents", []),
                    priority=1,
                    effectiveness_score=score
                )
                relevant_entries.append(entry)
        
        return relevant_entries
    
    async def _search_product_knowledge(
        self,
        query: str,
        context: Optional[Dict[str, Any]]
    ) -> List[KnowledgeEntry]:
        """
        Search product information as knowledge
        """
        try:
            db = next(get_db())
            
            # Search products related to query
            products = db.query(Product).filter(
                Product.name.ilike(f"%{query}%") |
                Product.description.ilike(f"%{query}%") |
                Product.category.ilike(f"%{query}%")
            ).limit(3).all()
            
            entries = []
            for product in products:
                # Create knowledge entry from product
                content = f"""
                Product: {product.name}
                Category: {product.category}
                Price: ${product.price}
                Rating: {product.rating}/5 ({product.review_count} reviews)
                
                Description:
                {product.description}
                
                Features:
                {product.features}
                
                Availability: {"In Stock" if product.stock_quantity > 0 else "Out of Stock"}
                """
                
                entry = KnowledgeEntry(
                    id=f"product_{product.id}",
                    title=f"Product Information: {product.name}",
                    content=content.strip(),
                    summary=f"{product.name} - ${product.price}",
                    category="product_info",
                    tags=[product.category, "product", "information"],
                    applicable_intents=[IntentType.PRODUCT_INFO, IntentType.PRODUCT_SEARCH],
                    priority=2,
                    effectiveness_score=0.8
                )
                entries.append(entry)
            
            db.close()
            return entries
            
        except Exception as e:
            logger.warning(f"Product knowledge search failed: {e}")
            return []
    
    async def _score_knowledge_entries(
        self,
        entries: List[KnowledgeEntry],
        query: str,
        intent: Optional[IntentType],
        context: Optional[Dict[str, Any]]
    ) -> List[KnowledgeEntry]:
        """
        Score and rank knowledge entries by relevance
        """
        try:
            if not entries:
                return []
            
            # Calculate relevance scores
            for entry in entries:
                score = entry.effectiveness_score
                
                # Intent matching bonus
                if intent and intent in entry.applicable_intents:
                    score += 0.3
                
                # Query text matching
                query_lower = query.lower()
                title_match = self._text_similarity(query_lower, entry.title.lower())
                content_match = self._text_similarity(query_lower, entry.content.lower())
                
                score += (title_match * 0.4) + (content_match * 0.2)
                
                # Context matching
                if context and entry.context_required:
                    context_match = self._context_similarity(context, entry.context_required)
                    score += context_match * 0.2
                
                # Priority bonus
                score += entry.priority * 0.1
                
                # Update effectiveness score
                entry.effectiveness_score = min(1.0, score)
            
            # Sort by score
            entries.sort(key=lambda x: x.effectiveness_score, reverse=True)
            
            return entries
            
        except Exception as e:
            logger.warning(f"Knowledge scoring failed: {e}")
            return entries
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate simple text similarity score
        """
        if not text1 or not text2:
            return 0.0
        
        # Simple word overlap similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _context_similarity(self, context1: Dict[str, Any], context2: Dict[str, Any]) -> float:
        """
        Calculate context similarity score
        """
        if not context1 or not context2:
            return 0.0
        
        # Simple key overlap
        keys1 = set(context1.keys())
        keys2 = set(context2.keys())
        
        intersection = keys1.intersection(keys2)
        union = keys1.union(keys2)
        
        return len(intersection) / len(union) if union else 0.0
    
    async def get_embedding(self, text: str) -> Optional[List[float]]:
        """
        Get text embedding using OpenAI
        """
        try:
            if not self.openai_client:
                return None
            
            # Check cache first
            cache_key = f"embedding:{hash(text)}"
            cached = self.embedding_cache.get(cache_key)
            if cached:
                return cached
            
            # Generate embedding
            response = await self.openai_client.embeddings.create(
                model="text-embedding-ada-002",
                input=text
            )
            
            embedding = response.data[0].embedding
            
            # Cache embedding
            self.embedding_cache[cache_key] = embedding
            
            return embedding
            
        except Exception as e:
            logger.warning(f"Embedding generation failed: {e}")
            return None
    
    async def semantic_search(
        self,
        query: str,
        intent: Optional[IntentType] = None,
        limit: int = 5
    ) -> List[KnowledgeEntry]:
        """
        Perform semantic search using embeddings
        """
        try:
            if not self.openai_client:
                # Fall back to text search
                return await self.search_knowledge(query, intent, limit=limit)
            
            # Get query embedding
            query_embedding = await self.get_embedding(query)
            if not query_embedding:
                return await self.search_knowledge(query, intent, limit=limit)
            
            # Get all knowledge entries
            all_entries = await self._get_all_knowledge_entries()
            
            # Calculate similarity scores
            scored_entries = []
            for entry in all_entries:
                if not entry.embedding_vector:
                    # Generate embedding for entry
                    entry.embedding_vector = await self.get_embedding(
                        f"{entry.title} {entry.content}"
                    )
                
                if entry.embedding_vector:
                    similarity = self._cosine_similarity(query_embedding, entry.embedding_vector)
                    entry.effectiveness_score = similarity
                    scored_entries.append(entry)
            
            # Filter by intent if provided
            if intent:
                scored_entries = [
                    entry for entry in scored_entries
                    if intent in entry.applicable_intents or not entry.applicable_intents
                ]
            
            # Sort by similarity
            scored_entries.sort(key=lambda x: x.effectiveness_score, reverse=True)
            
            return scored_entries[:limit]
            
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return await self.search_knowledge(query, intent, limit=limit)
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Calculate cosine similarity between two vectors
        """
        try:
            # Convert to numpy arrays
            a = np.array(vec1)
            b = np.array(vec2)
            
            # Calculate cosine similarity
            dot_product = np.dot(a, b)
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            
            if norm_a == 0 or norm_b == 0:
                return 0.0
            
            return dot_product / (norm_a * norm_b)
            
        except Exception as e:
            logger.warning(f"Cosine similarity calculation failed: {e}")
            return 0.0
    
    async def _get_all_knowledge_entries(self) -> List[KnowledgeEntry]:
        """
        Get all available knowledge entries
        """
        all_entries = []
        
        # Get FAQ entries
        for key, faq in self.faq_data.items():
            entry = KnowledgeEntry(
                id=key,
                title=faq["title"],
                content=faq["content"],
                summary=faq["title"],
                category=faq.get("category", "faq"),
                tags=faq.get("tags", []),
                applicable_intents=faq.get("intents", []),
                priority=1,
                effectiveness_score=0.5
            )
            all_entries.append(entry)
        
        # Get database entries
        try:
            db = next(get_db())
            db_entries = db.query(KnowledgeBase).all()
            
            for kb_entry in db_entries:
                entry = KnowledgeEntry(
                    id=str(kb_entry.id),
                    title=kb_entry.title,
                    content=kb_entry.content,
                    summary=kb_entry.summary or "",
                    category=kb_entry.category or "general",
                    tags=kb_entry.tags or [],
                    applicable_intents=[IntentType(intent) for intent in (kb_entry.applicable_intents or [])],
                    priority=kb_entry.priority,
                    context_required=kb_entry.context_required or {},
                    effectiveness_score=kb_entry.effectiveness_score
                )
                all_entries.append(entry)
            
            db.close()
            
        except Exception as e:
            logger.warning(f"Failed to get database knowledge entries: {e}")
        
        return all_entries
    
    async def add_knowledge_entry(
        self,
        title: str,
        content: str,
        category: str,
        tags: List[str] = None,
        applicable_intents: List[IntentType] = None,
        priority: int = 0
    ) -> str:
        """
        Add new knowledge base entry
        """
        try:
            db = next(get_db())
            
            # Create knowledge base entry
            kb_entry = KnowledgeBase(
                title=title,
                content=content,
                summary=title,
                category=category,
                tags=tags or [],
                applicable_intents=[intent.value for intent in (applicable_intents or [])],
                priority=priority,
                effectiveness_score=0.5,
                verified=False
            )
            
            db.add(kb_entry)
            db.commit()
            db.refresh(kb_entry)
            
            entry_id = str(kb_entry.id)
            logger.info(f"Added knowledge entry: {entry_id}")
            
            # Generate and store embedding
            embedding = await self.get_embedding(f"{title} {content}")
            if embedding:
                kb_entry.embedding_id = f"emb_{entry_id}"
                db.commit()
            
            db.close()
            return entry_id
            
        except Exception as e:
            logger.error(f"Failed to add knowledge entry: {e}")
            raise
    
    async def update_effectiveness_score(
        self,
        entry_id: str,
        feedback_score: float,
        was_helpful: bool
    ) -> bool:
        """
        Update knowledge entry effectiveness based on feedback
        """
        try:
            db = next(get_db())
            
            kb_entry = db.query(KnowledgeBase).filter(
                KnowledgeBase.id == entry_id
            ).first()
            
            if kb_entry:
                # Update effectiveness score using exponential moving average
                current_score = kb_entry.effectiveness_score
                alpha = 0.1  # Learning rate
                
                if was_helpful:
                    new_score = current_score + alpha * (feedback_score - current_score)
                else:
                    new_score = current_score - alpha * 0.2  # Penalty for unhelpful
                
                kb_entry.effectiveness_score = max(0.0, min(1.0, new_score))
                kb_entry.usage_count = (kb_entry.usage_count or 0) + 1
                
                db.commit()
                db.close()
                
                logger.info(f"Updated effectiveness score for {entry_id}: {new_score:.3f}")
                return True
            
            db.close()
            return False
            
        except Exception as e:
            logger.error(f"Failed to update effectiveness score: {e}")
            return False
    
    async def get_knowledge_stats(self) -> Dict[str, Any]:
        """
        Get knowledge base statistics
        """
        try:
            db = next(get_db())
            
            total_entries = db.query(KnowledgeBase).count()
            avg_effectiveness = db.query(KnowledgeBase.effectiveness_score).scalar() or 0.0
            
            category_stats = {}
            categories = db.query(KnowledgeBase.category).distinct().all()
            for (category,) in categories:
                if category:
                    count = db.query(KnowledgeBase).filter(
                        KnowledgeBase.category == category
                    ).count()
                    category_stats[category] = count
            
            db.close()
            
            return {
                "total_entries": total_entries + len(self.faq_data),
                "database_entries": total_entries,
                "faq_entries": len(self.faq_data),
                "avg_effectiveness": avg_effectiveness,
                "categories": category_stats,
                "last_updated": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get knowledge stats: {e}")
            return {"error": str(e)}