from typing import List, Dict, Any, Optional
from celery import Task
from loguru import logger
import asyncio

from src.core.celery import celery_app
from src.core.database import SessionLocal
from src.services.embedding import EmbeddingService
from src.services.recommendation import RecommendationService


class CallbackTask(Task):
    """Base task class with database session handling"""
    
    def on_success(self, retval, task_id, args, kwargs):
        logger.info(f"Task {task_id} completed successfully")
    
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        logger.error(f"Task {task_id} failed: {exc}")


@celery_app.task(bind=True, base=CallbackTask)
def generate_product_recommendations(
    self,
    user_id: str,
    context: str = "general",
    limit: int = 10,
    filters: Optional[Dict[str, Any]] = None,
    algorithm: str = "hybrid"
) -> Dict[str, Any]:
    """
    Generate personalized product recommendations for a user using RAG
    """
    try:
        logger.info(f"Generating {algorithm} recommendations for user {user_id}")
        
        # Use the new recommendation service
        recommendation_service = RecommendationService()
        
        # Run async function in event loop
        async def get_recommendations():
            return await recommendation_service.get_personalized_recommendations(
                user_id=user_id,
                context=context,
                limit=limit,
                filters=filters or {},
                algorithm=algorithm
            )
        
        # Execute async function
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(get_recommendations())
            return result
        finally:
            loop.close()
        
    except Exception as e:
        logger.error(f"Error generating recommendations for user {user_id}: {e}")
        raise


@celery_app.task(bind=True, base=CallbackTask)
def find_similar_products(
    self,
    product_id: str,
    limit: int = 10,
    similarity_threshold: float = 0.7,
    include_same_category: bool = True
) -> Dict[str, Any]:
    """
    Find products similar to a given product using vector similarity
    """
    try:
        logger.info(f"Finding similar products for {product_id} using vector similarity")
        
        # Use the new recommendation service
        recommendation_service = RecommendationService()
        
        # Run async function in event loop
        async def get_similar():
            return await recommendation_service.get_similar_products(
                product_id=product_id,
                limit=limit,
                similarity_threshold=similarity_threshold,
                include_same_category=include_same_category
            )
        
        # Execute async function
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(get_similar())
            return result
        finally:
            loop.close()
        
    except Exception as e:
        logger.error(f"Error finding similar products for {product_id}: {e}")
        raise


@celery_app.task(bind=True, base=CallbackTask)
def semantic_product_search(
    self,
    query: str,
    user_id: Optional[str] = None,
    filters: Optional[Dict[str, Any]] = None,
    limit: int = 20,
    boost_user_preferences: bool = True
) -> Dict[str, Any]:
    """
    Perform semantic search on products using natural language query with RAG
    """
    try:
        logger.info(f"Performing RAG-based semantic search for query: {query}")
        
        # Use the new recommendation service
        recommendation_service = RecommendationService()
        
        # Run async function in event loop
        async def search():
            return await recommendation_service.semantic_search(
                query=query,
                user_id=user_id,
                filters=filters or {},
                limit=limit,
                boost_user_preferences=boost_user_preferences
            )
        
        # Execute async function
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(search())
            return result
        finally:
            loop.close()
        
    except Exception as e:
        logger.error(f"Error performing semantic search for query '{query}': {e}")
        raise


@celery_app.task(bind=True, base=CallbackTask)
def update_user_preferences(
    self,
    user_id: str,
    interaction_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Update user preference profile based on interactions
    """
    try:
        logger.info(f"Updating preferences for user {user_id}")
        
        db = SessionLocal()
        try:
            # This is a placeholder - will be implemented with actual preference learning
            # For now, just log the interaction
            
            logger.info(f"Recorded interaction: {interaction_data}")
            
            return {
                "user_id": user_id,
                "updated": True,
                "interaction_recorded": interaction_data,
                "updated_at": "2024-01-01T00:00:00Z"
            }
            
        finally:
            db.close()
        
    except Exception as e:
        logger.error(f"Error updating preferences for user {user_id}: {e}")
        raise


@celery_app.task(bind=True, base=CallbackTask)
def batch_generate_embeddings(
    self,
    product_ids: List[str]
) -> Dict[str, Any]:
    """
    Generate embeddings for multiple products in batch
    """
    try:
        logger.info(f"Generating embeddings for {len(product_ids)} products")
        
        # This is a placeholder - will be implemented with actual embedding generation
        # For now, simulate the process
        
        results = {
            "processed_products": len(product_ids),
            "successful": len(product_ids),
            "failed": 0,
            "processing_time": len(product_ids) * 0.1,  # Mock processing time
            "errors": []
        }
        
        logger.info(f"Batch embedding generation completed: {results}")
        
        return results
        
    except Exception as e:
        logger.error(f"Error in batch embedding generation: {e}")
        raise


@celery_app.task(bind=True, base=CallbackTask)
def calculate_trending_products(
    self,
    time_window: str = "24h",
    category: Optional[str] = None,
    limit: int = 50,
    boost_factor: float = 1.2
) -> Dict[str, Any]:
    """
    Calculate trending products based on user interactions using real analytics
    """
    try:
        logger.info(f"Calculating trending products for {time_window} in category: {category}")
        
        # Use the new recommendation service
        recommendation_service = RecommendationService()
        
        # Run async function in event loop
        async def get_trending():
            return await recommendation_service.get_trending_recommendations(
                time_window=time_window,
                category=category,
                limit=limit,
                boost_factor=boost_factor
            )
        
        # Execute async function
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(get_trending())
            return result
        finally:
            loop.close()
        
    except Exception as e:
        logger.error(f"Error calculating trending products: {e}")
        raise


@celery_app.task(bind=True, base=CallbackTask)
def optimize_recommendation_model(
    self,
    model_type: str = "collaborative",
    training_data_window: str = "30d"
) -> Dict[str, Any]:
    """
    Optimize recommendation model with recent training data
    """
    try:
        logger.info(f"Optimizing {model_type} recommendation model")
        
        # This is a placeholder - will be implemented with actual model training
        # For now, simulate model optimization
        
        return {
            "model_type": model_type,
            "training_window": training_data_window,
            "optimization_complete": True,
            "performance_metrics": {
                "precision_at_10": 0.85,
                "recall_at_10": 0.72,
                "f1_score": 0.78,
                "auc": 0.91
            },
            "optimized_at": "2024-01-01T00:00:00Z"
        }
        
    except Exception as e:
        logger.error(f"Error optimizing {model_type} model: {e}")
        raise