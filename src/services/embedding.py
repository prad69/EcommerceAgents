import asyncio
from typing import List, Optional, Dict, Any, Tuple
import numpy as np
from loguru import logger
from datetime import datetime, timedelta
import json

from src.core.config import settings
from src.core.database import redis_client


class EmbeddingService:
    """
    Enhanced embedding service for product recommendations with RAG capabilities
    """
    
    def __init__(self):
        self.openai_client = None
        self.sentence_transformer = None
        self.setup_providers()
    
    def setup_providers(self):
        """
        Initialize embedding providers
        """
        # Setup OpenAI
        if settings.openai_api_key:
            try:
                import openai
                self.openai_client = openai.AsyncOpenAI(
                    api_key=settings.openai_api_key
                )
                logger.info("OpenAI embedding client initialized")
            except ImportError:
                logger.warning("OpenAI library not available")
        
        # Setup Sentence Transformers as fallback
        try:
            from sentence_transformers import SentenceTransformer
            self.sentence_transformer = SentenceTransformer(
                'all-MiniLM-L6-v2'  # Lightweight model
            )
            logger.info("Sentence Transformers model loaded")
        except ImportError:
            logger.warning("Sentence Transformers library not available")
    
    async def generate_embedding(
        self,
        text: str,
        model: str = "text-embedding-3-large"
    ) -> List[float]:
        """
        Generate embedding for given text
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        # Try OpenAI first
        if self.openai_client:
            try:
                return await self._generate_openai_embedding(text, model)
            except Exception as e:
                logger.warning(f"OpenAI embedding failed: {e}")
        
        # Fallback to Sentence Transformers
        if self.sentence_transformer:
            try:
                return await self._generate_sentence_transformer_embedding(text)
            except Exception as e:
                logger.error(f"Sentence Transformers embedding failed: {e}")
        
        raise Exception("No embedding provider available")
    
    async def _generate_openai_embedding(
        self,
        text: str,
        model: str = "text-embedding-3-large"
    ) -> List[float]:
        """
        Generate embedding using OpenAI API
        """
        response = await self.openai_client.embeddings.create(
            model=model,
            input=text.strip()[:8000],  # Limit input length
            encoding_format="float"
        )
        
        return response.data[0].embedding
    
    async def _generate_sentence_transformer_embedding(self, text: str) -> List[float]:
        """
        Generate embedding using Sentence Transformers (local)
        """
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(
            None,
            lambda: self.sentence_transformer.encode(text.strip()[:1000])
        )
        
        return embedding.tolist()
    
    async def generate_product_embedding(
        self,
        title: str,
        description: str = "",
        features: List[str] = None,
        category: str = "",
        brand: str = "",
        weight_config: Dict[str, float] = None
    ) -> List[float]:
        """
        Generate optimized embedding for product recommendation
        """
        # Default weights for different product attributes
        weights = weight_config or {
            "title": 0.4,
            "description": 0.3,
            "features": 0.2,
            "category": 0.06,
            "brand": 0.04
        }
        
        # Build weighted text representation
        text_parts = []
        
        if title:
            # Repeat title based on weight to give it more importance
            title_repeat = int(weights["title"] * 10)
            text_parts.extend([title] * max(1, title_repeat))
        
        if description:
            desc_repeat = int(weights["description"] * 10)
            text_parts.extend([description] * max(1, desc_repeat))
        
        if features:
            feature_text = ", ".join(features)
            feature_repeat = int(weights["features"] * 10)
            text_parts.extend([feature_text] * max(1, feature_repeat))
        
        if category:
            cat_repeat = int(weights["category"] * 10)
            text_parts.extend([category] * max(1, cat_repeat))
        
        if brand:
            brand_repeat = int(weights["brand"] * 10)
            text_parts.extend([brand] * max(1, brand_repeat))
        
        # Combine all text
        combined_text = " ".join(text_parts)
        
        return await self.generate_embedding(combined_text)
    
    async def generate_user_preference_embedding(
        self,
        user_interactions: List[Dict[str, Any]],
        time_decay: bool = True
    ) -> List[float]:
        """
        Generate user preference embedding based on interaction history
        """
        if not user_interactions:
            # Return zero vector for new users
            return [0.0] * 1536  # OpenAI embedding dimension
        
        # Weight interactions by type and recency
        interaction_weights = {
            "view": 0.1,
            "click": 0.3,
            "cart": 0.6,
            "purchase": 1.0,
            "like": 0.4,
            "share": 0.3
        }
        
        weighted_embeddings = []
        total_weight = 0
        
        now = datetime.utcnow()
        
        for interaction in user_interactions:
            # Get base weight for interaction type
            base_weight = interaction_weights.get(interaction.get("type", "view"), 0.1)
            
            # Apply time decay if enabled
            weight = base_weight
            if time_decay and "timestamp" in interaction:
                interaction_time = datetime.fromisoformat(interaction["timestamp"])
                days_ago = (now - interaction_time).days
                # Exponential decay: recent interactions matter more
                decay_factor = np.exp(-days_ago / 30.0)  # 30-day half-life
                weight *= decay_factor
            
            # Get product embedding (from cache or generate)
            product_embedding = await self._get_cached_product_embedding(
                interaction["product_id"]
            )
            
            if product_embedding:
                weighted_embeddings.append(np.array(product_embedding) * weight)
                total_weight += weight
        
        if not weighted_embeddings or total_weight == 0:
            return [0.0] * 1536
        
        # Calculate weighted average
        user_embedding = np.sum(weighted_embeddings, axis=0) / total_weight
        
        # Normalize the embedding
        norm = np.linalg.norm(user_embedding)
        if norm > 0:
            user_embedding = user_embedding / norm
        
        return user_embedding.tolist()
    
    async def generate_batch_embeddings(
        self,
        texts: List[str],
        batch_size: int = 10
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in batches
        """
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = await asyncio.gather(*[
                self.generate_embedding(text) for text in batch
            ])
            embeddings.extend(batch_embeddings)
        
        return embeddings
    
    def calculate_similarity(
        self,
        embedding1: List[float],
        embedding2: List[float]
    ) -> float:
        """
        Calculate cosine similarity between two embeddings
        """
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        # Cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)
        
        if norm_product == 0:
            return 0.0
        
        return float(dot_product / norm_product)
    
    async def find_similar_embeddings(
        self,
        query_embedding: List[float],
        candidate_embeddings: List[Dict[str, Any]],
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Find most similar embeddings from a list of candidates
        """
        similarities = []
        
        for candidate in candidate_embeddings:
            similarity = self.calculate_similarity(
                query_embedding,
                candidate["embedding"]
            )
            similarities.append({
                **candidate,
                "similarity": similarity
            })
        
        # Sort by similarity and return top K
        similarities.sort(key=lambda x: x["similarity"], reverse=True)
        return similarities[:top_k]
    
    async def find_diverse_recommendations(
        self,
        query_embedding: List[float],
        candidate_embeddings: List[Dict[str, Any]],
        top_k: int = 10,
        diversity_factor: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        Find diverse recommendations using MMR (Maximal Marginal Relevance)
        """
        if not candidate_embeddings:
            return []
        
        query_vec = np.array(query_embedding)
        candidate_vecs = np.array([item["embedding"] for item in candidate_embeddings])
        
        # Compute similarity to query
        query_similarities = np.dot(candidate_vecs, query_vec) / (
            np.linalg.norm(candidate_vecs, axis=1) * np.linalg.norm(query_vec)
        )
        
        selected_indices = []
        remaining_indices = list(range(len(candidate_embeddings)))
        
        # Select first item (highest similarity to query)
        first_idx = np.argmax(query_similarities)
        selected_indices.append(first_idx)
        remaining_indices.remove(first_idx)
        
        # Select remaining items using MMR
        for _ in range(min(top_k - 1, len(remaining_indices))):
            if not remaining_indices:
                break
            
            mmr_scores = []
            
            for idx in remaining_indices:
                # Relevance to query
                relevance = query_similarities[idx]
                
                # Maximum similarity to already selected items
                if selected_indices:
                    selected_vecs = candidate_vecs[selected_indices]
                    candidate_vec = candidate_vecs[idx]
                    
                    similarities_to_selected = np.dot(selected_vecs, candidate_vec) / (
                        np.linalg.norm(selected_vecs, axis=1) * np.linalg.norm(candidate_vec)
                    )
                    max_similarity = np.max(similarities_to_selected)
                else:
                    max_similarity = 0
                
                # MMR score: balance relevance and diversity
                mmr_score = (1 - diversity_factor) * relevance - diversity_factor * max_similarity
                mmr_scores.append(mmr_score)
            
            # Select item with highest MMR score
            best_remaining_idx = np.argmax(mmr_scores)
            best_idx = remaining_indices[best_remaining_idx]
            
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)
        
        # Return selected items with similarity scores
        results = []
        for idx in selected_indices:
            item = candidate_embeddings[idx].copy()
            item["similarity"] = float(query_similarities[idx])
            results.append(item)
        
        return results
    
    async def compute_similarity_matrix(
        self,
        embeddings_1: List[List[float]],
        embeddings_2: List[List[float]] = None
    ) -> np.ndarray:
        """
        Compute similarity matrix between two sets of embeddings
        """
        embeddings_1 = np.array(embeddings_1)
        
        if embeddings_2 is None:
            embeddings_2 = embeddings_1
        else:
            embeddings_2 = np.array(embeddings_2)
        
        # Compute cosine similarity matrix
        # Normalize embeddings
        norm_1 = np.linalg.norm(embeddings_1, axis=1, keepdims=True)
        norm_2 = np.linalg.norm(embeddings_2, axis=1, keepdims=True)
        
        normalized_1 = embeddings_1 / np.maximum(norm_1, 1e-8)
        normalized_2 = embeddings_2 / np.maximum(norm_2, 1e-8)
        
        # Compute similarity matrix
        similarity_matrix = np.dot(normalized_1, normalized_2.T)
        
        return similarity_matrix
    
    async def _get_cached_product_embedding(
        self,
        product_id: str
    ) -> Optional[List[float]]:
        """
        Get product embedding from cache or database
        """
        # Try Redis cache first
        cache_key = f"product_embedding:{product_id}"
        try:
            cached = redis_client.get(cache_key)
            if cached:
                return json.loads(cached)
        except Exception as e:
            logger.warning(f"Failed to get cached embedding: {e}")
        
        # TODO: Fallback to database lookup
        # For now, return None - this would be implemented with actual product lookup
        return None
    
    async def cache_product_embedding(
        self,
        product_id: str,
        embedding: List[float],
        ttl: int = 3600 * 24  # 24 hours
    ):
        """
        Cache product embedding in Redis
        """
        try:
            cache_key = f"product_embedding:{product_id}"
            redis_client.setex(
                cache_key,
                ttl,
                json.dumps(embedding)
            )
        except Exception as e:
            logger.warning(f"Failed to cache embedding: {e}")
    
    async def update_embeddings(
        self,
        product_id: str,
        new_text: str
    ) -> List[float]:
        """
        Update embeddings for a product with new text content
        """
        try:
            # Generate new embedding
            new_embedding = await self.generate_embedding(new_text)
            
            # Cache the new embedding
            await self.cache_product_embedding(product_id, new_embedding)
            
            # TODO: Update in vector database
            # This would typically involve updating the vector store
            
            logger.info(f"Updated embeddings for product {product_id}")
            return new_embedding
            
        except Exception as e:
            logger.error(f"Failed to update embeddings for product {product_id}: {e}")
            raise