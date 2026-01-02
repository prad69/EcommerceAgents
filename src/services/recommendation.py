from typing import List, Dict, Any, Optional, Tuple
import asyncio
import numpy as np
from datetime import datetime, timedelta
from loguru import logger
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc, func
import json

from src.core.database import get_db, vector_db, redis_client
from src.models.product import Product, Category, Brand
from src.models.analytics import UserInteraction, SearchQuery, RecommendationEvent
from src.services.embedding import EmbeddingService


class RecommendationService:
    """
    RAG-based recommendation service using vector embeddings and content filtering
    """
    
    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.vector_db = vector_db
    
    async def get_personalized_recommendations(
        self,
        user_id: str,
        context: str = "general",
        limit: int = 10,
        filters: Dict[str, Any] = None,
        algorithm: str = "hybrid"
    ) -> Dict[str, Any]:
        """
        Get personalized product recommendations for a user
        """
        start_time = datetime.utcnow()
        
        try:
            # Get user preference embedding
            user_embedding = await self._get_user_preference_embedding(user_id)
            
            # Get candidate products based on filters
            candidates = await self._get_candidate_products(filters, limit * 3)  # Get more candidates for filtering
            
            # Generate recommendations based on algorithm
            if algorithm == "content_based":
                recommendations = await self._content_based_recommendations(
                    user_embedding, candidates, limit
                )
            elif algorithm == "collaborative":
                recommendations = await self._collaborative_recommendations(
                    user_id, candidates, limit
                )
            else:  # hybrid
                recommendations = await self._hybrid_recommendations(
                    user_id, user_embedding, candidates, limit
                )
            
            # Add diversity and re-rank
            diverse_recommendations = await self._add_diversity(
                recommendations, diversity_factor=0.2
            )
            
            # Log recommendation event for analytics
            await self._log_recommendation_event(
                user_id, context, algorithm, diverse_recommendations[:limit]
            )
            
            response_time = (datetime.utcnow() - start_time).total_seconds()
            
            return {
                "user_id": user_id,
                "context": context,
                "algorithm": algorithm,
                "recommendations": diverse_recommendations[:limit],
                "total_candidates": len(candidates),
                "response_time": response_time,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get recommendations for user {user_id}: {e}")
            raise
    
    async def get_similar_products(
        self,
        product_id: str,
        limit: int = 10,
        similarity_threshold: float = 0.5,
        include_same_category: bool = True
    ) -> Dict[str, Any]:
        """
        Find products similar to a given product using vector similarity
        """
        start_time = datetime.utcnow()
        
        try:
            db = next(get_db())
            
            # Get the source product
            source_product = db.query(Product).filter(Product.id == product_id).first()
            if not source_product:
                raise ValueError(f"Product {product_id} not found")
            
            # Generate embedding for source product
            source_embedding = await self.embedding_service.generate_product_embedding(
                title=source_product.title,
                description=source_product.description or "",
                features=source_product.features or [],
                category=source_product.category.name if source_product.category else "",
                brand=source_product.brand.name if source_product.brand else ""
            )
            
            # Build filters for candidate search
            filters = {}
            if include_same_category and source_product.category_id:
                filters["category"] = source_product.category.name
            
            # Search similar products in vector database
            similar_products = await self.vector_db.search_similar_products(
                query_embedding=source_embedding,
                limit=limit + 1,  # +1 to account for the source product itself
                filters=filters
            )
            
            # Filter out the source product and apply similarity threshold
            filtered_results = []
            for product in similar_products:
                if (product["product_id"] != product_id and 
                    product["similarity_score"] >= similarity_threshold):
                    filtered_results.append(product)
            
            # Enrich with product details
            enriched_results = await self._enrich_with_product_details(
                filtered_results[:limit], db
            )
            
            response_time = (datetime.utcnow() - start_time).total_seconds()
            
            return {
                "source_product_id": product_id,
                "similar_products": enriched_results,
                "similarity_threshold": similarity_threshold,
                "algorithm": "vector_similarity",
                "response_time": response_time,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to find similar products for {product_id}: {e}")
            raise
        finally:
            db.close()
    
    async def semantic_search(
        self,
        query: str,
        user_id: Optional[str] = None,
        filters: Dict[str, Any] = None,
        limit: int = 20,
        boost_user_preferences: bool = True
    ) -> Dict[str, Any]:
        """
        Perform semantic search using natural language query
        """
        start_time = datetime.utcnow()
        
        try:
            # Generate embedding for search query
            query_embedding = await self.embedding_service.generate_embedding(query)
            
            # Get user preferences if available
            user_embedding = None
            if user_id and boost_user_preferences:
                user_embedding = await self._get_user_preference_embedding(user_id)
            
            # Combine query and user embeddings if both available
            search_embedding = query_embedding
            if user_embedding and any(x != 0 for x in user_embedding):
                # Weighted combination: 70% query, 30% user preferences
                query_weight = 0.7
                user_weight = 0.3
                search_embedding = [
                    query_weight * q + user_weight * u 
                    for q, u in zip(query_embedding, user_embedding)
                ]
            
            # Search in vector database
            search_results = await self.vector_db.search_similar_products(
                query_embedding=search_embedding,
                limit=limit,
                filters=filters
            )
            
            # Enrich with product details
            db = next(get_db())
            try:
                enriched_results = await self._enrich_with_product_details(
                    search_results, db
                )
            finally:
                db.close()
            
            # Log search query for analytics
            await self._log_search_query(query, user_id, len(enriched_results))
            
            response_time = (datetime.utcnow() - start_time).total_seconds()
            
            return {
                "query": query,
                "user_id": user_id,
                "results": enriched_results,
                "total_results": len(enriched_results),
                "algorithm": "semantic_search",
                "user_boost_applied": boost_user_preferences and user_embedding is not None,
                "response_time": response_time,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to perform semantic search for query '{query}': {e}")
            raise
    
    async def get_trending_recommendations(
        self,
        time_window: str = "24h",
        category: Optional[str] = None,
        limit: int = 20,
        boost_factor: float = 1.2
    ) -> Dict[str, Any]:
        """
        Get trending products based on recent user interactions
        """
        try:
            db = next(get_db())
            
            # Calculate time window
            if time_window == "1h":
                since = datetime.utcnow() - timedelta(hours=1)
            elif time_window == "24h":
                since = datetime.utcnow() - timedelta(days=1)
            elif time_window == "7d":
                since = datetime.utcnow() - timedelta(days=7)
            else:
                since = datetime.utcnow() - timedelta(days=1)  # Default to 24h
            
            # Build query for trending products based on interactions
            interaction_query = db.query(
                UserInteraction.product_id,
                func.count(UserInteraction.id).label("interaction_count"),
                func.sum(
                    func.case([
                        (UserInteraction.interaction_type == "purchase", 10),
                        (UserInteraction.interaction_type == "cart", 5),
                        (UserInteraction.interaction_type == "click", 2),
                        (UserInteraction.interaction_type == "view", 1)
                    ], else_=1)
                ).label("weighted_score")
            ).filter(
                UserInteraction.created_at >= since
            ).group_by(UserInteraction.product_id)
            
            # Add category filter if specified
            if category:
                interaction_query = interaction_query.join(Product).join(Category).filter(
                    Category.name.ilike(f"%{category}%")
                )
            
            # Order by weighted score and limit
            trending_data = interaction_query.order_by(
                desc("weighted_score")
            ).limit(limit).all()
            
            # Enrich with product details
            product_details = []
            for trend in trending_data:
                product = db.query(Product).filter(Product.id == trend.product_id).first()
                if product:
                    trending_score = float(trend.weighted_score) * boost_factor
                    product_details.append({
                        "product_id": str(product.id),
                        "title": product.title,
                        "price": product.price,
                        "category": product.category.name if product.category else None,
                        "brand": product.brand.name if product.brand else None,
                        "interaction_count": trend.interaction_count,
                        "trending_score": trending_score,
                        "primary_image": product.primary_image,
                        "average_rating": product.average_rating,
                        "review_count": product.review_count
                    })
            
            return {
                "time_window": time_window,
                "category": category,
                "trending_products": product_details,
                "algorithm": "interaction_based_trending",
                "boost_factor": boost_factor,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get trending recommendations: {e}")
            raise
        finally:
            db.close()
    
    async def _get_user_preference_embedding(self, user_id: str) -> List[float]:
        """
        Get or compute user preference embedding
        """
        # Try cache first
        cache_key = f"user_preferences:{user_id}"
        try:
            cached = redis_client.get(cache_key)
            if cached:
                return json.loads(cached)
        except Exception as e:
            logger.warning(f"Failed to get cached user preferences: {e}")
        
        # Compute from user interactions
        db = next(get_db())
        try:
            # Get recent user interactions (last 90 days)
            since = datetime.utcnow() - timedelta(days=90)
            interactions = db.query(UserInteraction).filter(
                and_(
                    UserInteraction.user_id == user_id,
                    UserInteraction.created_at >= since
                )
            ).order_by(desc(UserInteraction.created_at)).limit(100).all()
            
            # Convert to format expected by embedding service
            interaction_data = []
            for interaction in interactions:
                interaction_data.append({
                    "product_id": str(interaction.product_id),
                    "type": interaction.interaction_type,
                    "timestamp": interaction.created_at.isoformat()
                })
            
            # Generate user preference embedding
            user_embedding = await self.embedding_service.generate_user_preference_embedding(
                interaction_data, time_decay=True
            )
            
            # Cache the result
            try:
                redis_client.setex(
                    cache_key,
                    3600 * 4,  # 4 hours TTL
                    json.dumps(user_embedding)
                )
            except Exception as e:
                logger.warning(f"Failed to cache user preferences: {e}")
            
            return user_embedding
            
        finally:
            db.close()
    
    async def _get_candidate_products(
        self,
        filters: Dict[str, Any] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get candidate products for recommendation
        """
        db = next(get_db())
        try:
            query = db.query(Product).filter(
                and_(
                    Product.is_active == True,
                    Product.is_in_stock == True
                )
            )
            
            # Apply filters
            if filters:
                if "category" in filters:
                    query = query.join(Category).filter(
                        Category.name.ilike(f"%{filters['category']}%")
                    )
                if "brand" in filters:
                    query = query.join(Brand).filter(
                        Brand.name.ilike(f"%{filters['brand']}%")
                    )
                if "min_price" in filters:
                    query = query.filter(Product.price >= filters["min_price"])
                if "max_price" in filters:
                    query = query.filter(Product.price <= filters["max_price"])
                if "min_rating" in filters:
                    query = query.filter(Product.average_rating >= filters["min_rating"])
            
            products = query.order_by(desc(Product.average_rating)).limit(limit).all()
            
            # Convert to dictionary format
            candidates = []
            for product in products:
                candidates.append({
                    "product_id": str(product.id),
                    "title": product.title,
                    "description": product.description or "",
                    "price": product.price,
                    "category": product.category.name if product.category else "",
                    "brand": product.brand.name if product.brand else "",
                    "features": product.features or [],
                    "average_rating": product.average_rating,
                    "review_count": product.review_count,
                    "primary_image": product.primary_image
                })
            
            return candidates
            
        finally:
            db.close()
    
    async def _content_based_recommendations(
        self,
        user_embedding: List[float],
        candidates: List[Dict[str, Any]],
        limit: int
    ) -> List[Dict[str, Any]]:
        """
        Generate content-based recommendations using user preferences
        """
        if not any(x != 0 for x in user_embedding):  # Zero vector
            # For new users, return popular products
            return sorted(
                candidates,
                key=lambda x: (x["average_rating"] * x["review_count"]),
                reverse=True
            )[:limit]
        
        # Generate embeddings for all candidates
        candidate_embeddings = []
        for candidate in candidates:
            embedding = await self.embedding_service.generate_product_embedding(
                title=candidate["title"],
                description=candidate["description"],
                features=candidate["features"],
                category=candidate["category"],
                brand=candidate["brand"]
            )
            
            candidate_embeddings.append({
                **candidate,
                "embedding": embedding
            })
        
        # Find similar products using user preferences
        recommendations = await self.embedding_service.find_similar_embeddings(
            query_embedding=user_embedding,
            candidate_embeddings=candidate_embeddings,
            top_k=limit
        )
        
        return recommendations
    
    async def _collaborative_recommendations(
        self,
        user_id: str,
        candidates: List[Dict[str, Any]],
        limit: int
    ) -> List[Dict[str, Any]]:
        """
        Generate collaborative filtering recommendations
        """
        # For Phase 2, we'll use a simplified approach based on similar users' preferences
        # In a full implementation, this would use matrix factorization or deep learning
        
        db = next(get_db())
        try:
            # Find users with similar interaction patterns
            user_interactions = db.query(UserInteraction.product_id).filter(
                UserInteraction.user_id == user_id
            ).distinct().subquery()
            
            # Find other users who interacted with similar products
            similar_users = db.query(
                UserInteraction.user_id,
                func.count(UserInteraction.product_id).label("common_products")
            ).filter(
                UserInteraction.product_id.in_(user_interactions),
                UserInteraction.user_id != user_id
            ).group_by(UserInteraction.user_id).order_by(
                desc("common_products")
            ).limit(50).all()
            
            # Get products liked by similar users but not by current user
            similar_user_ids = [str(u.user_id) for u in similar_users]
            
            recommended_products = db.query(
                UserInteraction.product_id,
                func.count(UserInteraction.id).label("recommendation_score")
            ).filter(
                and_(
                    UserInteraction.user_id.in_(similar_user_ids),
                    UserInteraction.interaction_type.in_(["purchase", "cart", "like"]),
                    ~UserInteraction.product_id.in_(user_interactions)
                )
            ).group_by(UserInteraction.product_id).order_by(
                desc("recommendation_score")
            ).limit(limit).all()
            
            # Filter and format results
            result_ids = [str(p.product_id) for p in recommended_products]
            filtered_candidates = [
                c for c in candidates if c["product_id"] in result_ids
            ]
            
            # Sort by collaborative score
            score_map = {str(p.product_id): p.recommendation_score for p in recommended_products}
            filtered_candidates.sort(
                key=lambda x: score_map.get(x["product_id"], 0),
                reverse=True
            )
            
            return filtered_candidates[:limit]
            
        finally:
            db.close()
    
    async def _hybrid_recommendations(
        self,
        user_id: str,
        user_embedding: List[float],
        candidates: List[Dict[str, Any]],
        limit: int
    ) -> List[Dict[str, Any]]:
        """
        Generate hybrid recommendations combining content-based and collaborative filtering
        """
        # Get content-based recommendations
        content_recs = await self._content_based_recommendations(
            user_embedding, candidates, limit
        )
        
        # Get collaborative recommendations
        collab_recs = await self._collaborative_recommendations(
            user_id, candidates, limit
        )
        
        # Combine with weights: 60% content-based, 40% collaborative
        content_weight = 0.6
        collab_weight = 0.4
        
        # Create scoring map
        scores = {}
        
        # Add content-based scores
        for i, rec in enumerate(content_recs):
            product_id = rec["product_id"]
            content_score = (limit - i) / limit  # Normalize by position
            scores[product_id] = {
                "content_score": content_score,
                "collab_score": 0,
                "product": rec
            }
        
        # Add collaborative scores
        for i, rec in enumerate(collab_recs):
            product_id = rec["product_id"]
            collab_score = (limit - i) / limit
            
            if product_id in scores:
                scores[product_id]["collab_score"] = collab_score
            else:
                scores[product_id] = {
                    "content_score": 0,
                    "collab_score": collab_score,
                    "product": rec
                }
        
        # Calculate hybrid scores and sort
        hybrid_results = []
        for product_id, data in scores.items():
            hybrid_score = (
                content_weight * data["content_score"] + 
                collab_weight * data["collab_score"]
            )
            
            product = data["product"]
            product["hybrid_score"] = hybrid_score
            hybrid_results.append(product)
        
        # Sort by hybrid score
        hybrid_results.sort(key=lambda x: x.get("hybrid_score", 0), reverse=True)
        
        return hybrid_results[:limit]
    
    async def _add_diversity(
        self,
        recommendations: List[Dict[str, Any]],
        diversity_factor: float = 0.2
    ) -> List[Dict[str, Any]]:
        """
        Add diversity to recommendations using MMR
        """
        if not recommendations:
            return recommendations
        
        # Generate embeddings for diversity calculation
        candidate_embeddings = []
        for rec in recommendations:
            embedding = await self.embedding_service.generate_product_embedding(
                title=rec["title"],
                description=rec.get("description", ""),
                features=rec.get("features", []),
                category=rec.get("category", ""),
                brand=rec.get("brand", "")
            )
            candidate_embeddings.append({
                **rec,
                "embedding": embedding
            })
        
        # Use the first recommendation as query for diversity
        if candidate_embeddings:
            query_embedding = candidate_embeddings[0]["embedding"]
            diverse_results = await self.embedding_service.find_diverse_recommendations(
                query_embedding=query_embedding,
                candidate_embeddings=candidate_embeddings,
                top_k=len(recommendations),
                diversity_factor=diversity_factor
            )
            return diverse_results
        
        return recommendations
    
    async def _enrich_with_product_details(
        self,
        search_results: List[Dict[str, Any]],
        db: Session
    ) -> List[Dict[str, Any]]:
        """
        Enrich search results with full product details
        """
        enriched = []
        
        for result in search_results:
            product_id = result["product_id"]
            product = db.query(Product).filter(Product.id == product_id).first()
            
            if product:
                enriched.append({
                    "product_id": str(product.id),
                    "title": product.title,
                    "description": product.description,
                    "short_description": product.short_description,
                    "price": product.price,
                    "original_price": product.original_price,
                    "currency": product.currency,
                    "sku": product.sku,
                    "category": product.category.name if product.category else None,
                    "brand": product.brand.name if product.brand else None,
                    "features": product.features,
                    "tags": product.tags,
                    "primary_image": product.primary_image,
                    "images": product.images,
                    "average_rating": product.average_rating,
                    "review_count": product.review_count,
                    "is_featured": product.is_featured,
                    "similarity_score": result.get("similarity_score"),
                    "metadata": result.get("metadata", {})
                })
        
        return enriched
    
    async def _log_recommendation_event(
        self,
        user_id: str,
        context: str,
        algorithm: str,
        recommendations: List[Dict[str, Any]]
    ):
        """
        Log recommendation event for analytics
        """
        try:
            db = next(get_db())
            
            recommended_product_ids = [r["product_id"] for r in recommendations]
            
            event = RecommendationEvent(
                user_id=user_id,
                session_id=f"session_{user_id}_{int(datetime.utcnow().timestamp())}",
                recommendation_type=algorithm,
                context=context,
                recommended_products=recommended_product_ids,
                algorithm_version="phase2_v1"
            )
            
            db.add(event)
            db.commit()
            
        except Exception as e:
            logger.warning(f"Failed to log recommendation event: {e}")
        finally:
            db.close()
    
    async def _log_search_query(
        self,
        query: str,
        user_id: Optional[str],
        results_count: int
    ):
        """
        Log search query for analytics
        """
        try:
            db = next(get_db())
            
            search_event = SearchQuery(
                user_id=user_id,
                session_id=f"session_{user_id or 'anonymous'}_{int(datetime.utcnow().timestamp())}",
                query_text=query,
                results_count=results_count
            )
            
            db.add(search_event)
            db.commit()
            
        except Exception as e:
            logger.warning(f"Failed to log search query: {e}")
        finally:
            db.close()