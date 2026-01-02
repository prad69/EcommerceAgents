from typing import List, Dict, Any, Optional
from celery import Task
from loguru import logger
import asyncio

from src.core.celery import celery_app
from src.core.database import SessionLocal
from src.services.review_analysis import ReviewAnalysisService
from src.services.review_ingestion import ReviewIngestionService
from src.services.sentiment_analysis import SentimentAnalysisService


class CallbackTask(Task):
    """Base task class with database session handling"""
    
    def on_success(self, retval, task_id, args, kwargs):
        logger.info(f"Review task {task_id} completed successfully")
    
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        logger.error(f"Review task {task_id} failed: {exc}")


@celery_app.task(bind=True, base=CallbackTask)
def process_single_review(self, review_id: str) -> Dict[str, Any]:
    """
    Process a single review for sentiment analysis and insights
    """
    try:
        logger.info(f"Processing review {review_id}")
        
        # Initialize service
        review_service = ReviewAnalysisService()
        
        # Run async function in event loop
        async def process():
            return await review_service.process_review(review_id)
        
        # Execute async function
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(process())
            return result
        finally:
            loop.close()
        
    except Exception as e:
        logger.error(f"Error processing review {review_id}: {e}")
        raise


@celery_app.task(bind=True, base=CallbackTask)
def batch_process_reviews(
    self,
    product_id: Optional[str] = None,
    limit: int = 100,
    only_unprocessed: bool = True
) -> Dict[str, Any]:
    """
    Process multiple reviews in batch
    """
    try:
        logger.info(f"Batch processing reviews for product {product_id}, limit {limit}")
        
        # Initialize service
        review_service = ReviewAnalysisService()
        
        # Run async function in event loop
        async def batch_process():
            return await review_service.batch_process_reviews(
                product_id=product_id,
                limit=limit,
                only_unprocessed=only_unprocessed
            )
        
        # Execute async function
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(batch_process())
            return result
        finally:
            loop.close()
        
    except Exception as e:
        logger.error(f"Error in batch review processing: {e}")
        raise


@celery_app.task(bind=True, base=CallbackTask)
def ingest_reviews_from_source(
    self,
    source: str,
    data_path: Optional[str] = None,
    data_url: Optional[str] = None,
    product_filter: Optional[List[str]] = None,
    batch_size: int = 100,
    auto_process: bool = True
) -> Dict[str, Any]:
    """
    Ingest reviews from external data source
    """
    try:
        logger.info(f"Ingesting reviews from {source}")
        
        # Initialize service
        ingestion_service = ReviewIngestionService()
        
        # Run async function in event loop
        async def ingest():
            result = await ingestion_service.ingest_reviews(
                source=source,
                data_path=data_path,
                data_url=data_url,
                product_filter=product_filter,
                batch_size=batch_size
            )
            
            # Auto-process ingested reviews if requested
            if auto_process and result["successful"] > 0:
                logger.info(f"Auto-processing {result['successful']} ingested reviews")
                
                # Trigger batch processing for each product
                if product_filter:
                    for product_id in product_filter:
                        batch_process_reviews.delay(
                            product_id=product_id,
                            only_unprocessed=True
                        )
                else:
                    # Process all unprocessed reviews
                    batch_process_reviews.delay(only_unprocessed=True)
            
            return result
        
        # Execute async function
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(ingest())
            return result
        finally:
            loop.close()
        
    except Exception as e:
        logger.error(f"Error ingesting reviews from {source}: {e}")
        raise


@celery_app.task(bind=True, base=CallbackTask)
def analyze_text_sentiment(
    self,
    text: str,
    include_aspects: bool = True
) -> Dict[str, Any]:
    """
    Analyze sentiment of arbitrary text
    """
    try:
        logger.info(f"Analyzing sentiment for text of length {len(text)}")
        
        # Initialize service
        sentiment_service = SentimentAnalysisService()
        
        # Run async function in event loop
        async def analyze():
            return await sentiment_service.analyze_sentiment(text, include_aspects)
        
        # Execute async function
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(analyze())
            return result
        finally:
            loop.close()
        
    except Exception as e:
        logger.error(f"Error analyzing sentiment: {e}")
        raise


@celery_app.task(bind=True, base=CallbackTask)
def batch_analyze_sentiments(
    self,
    texts: List[str],
    include_aspects: bool = True
) -> List[Dict[str, Any]]:
    """
    Analyze sentiments for multiple texts
    """
    try:
        logger.info(f"Batch analyzing sentiment for {len(texts)} texts")
        
        # Initialize service
        sentiment_service = SentimentAnalysisService()
        
        # Run async function in event loop
        async def analyze():
            return await sentiment_service.batch_analyze_sentiments(texts, include_aspects)
        
        # Execute async function
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(analyze())
            return result
        finally:
            loop.close()
        
    except Exception as e:
        logger.error(f"Error in batch sentiment analysis: {e}")
        raise


@celery_app.task(bind=True, base=CallbackTask)
def update_product_summary(self, product_id: str) -> Dict[str, Any]:
    """
    Update review summary for a specific product
    """
    try:
        logger.info(f"Updating review summary for product {product_id}")
        
        # Initialize service
        review_service = ReviewAnalysisService()
        
        # Run async function in event loop
        async def update():
            await review_service._update_product_summary(product_id)
            return {"product_id": product_id, "updated": True}
        
        # Execute async function
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(update())
            return result
        finally:
            loop.close()
        
    except Exception as e:
        logger.error(f"Error updating product summary for {product_id}: {e}")
        raise


@celery_app.task(bind=True, base=CallbackTask)
def detect_fake_reviews(
    self,
    product_id: Optional[str] = None,
    threshold: float = 0.3
) -> Dict[str, Any]:
    """
    Detect potentially fake reviews based on authenticity scores
    """
    try:
        logger.info(f"Detecting fake reviews for product {product_id} with threshold {threshold}")
        
        db = SessionLocal()
        try:
            from src.models.review import Review, ReviewAnalysis
            from sqlalchemy import and_
            
            # Build query
            query = db.query(Review).join(ReviewAnalysis)
            
            if product_id:
                query = query.filter(Review.product_id == product_id)
            
            # Find reviews with low authenticity scores
            suspicious_reviews = query.filter(
                ReviewAnalysis.authenticity_score < threshold
            ).all()
            
            # Analyze patterns
            suspicious_data = []
            patterns = {
                "short_reviews": 0,
                "extreme_ratings": 0,
                "unverified_purchases": 0,
                "generic_content": 0
            }
            
            for review in suspicious_reviews:
                analysis = review.analysis
                
                # Categorize suspicious patterns
                flags = []
                if len(review.content.split()) < 10:
                    flags.append("very_short")
                    patterns["short_reviews"] += 1
                
                if review.rating in [1.0, 5.0]:
                    flags.append("extreme_rating")
                    patterns["extreme_ratings"] += 1
                
                if not review.verified_purchase:
                    flags.append("unverified")
                    patterns["unverified_purchases"] += 1
                
                suspicious_data.append({
                    "review_id": str(review.id),
                    "authenticity_score": analysis.authenticity_score,
                    "rating": review.rating,
                    "word_count": review.word_count,
                    "verified_purchase": review.verified_purchase,
                    "flags": flags
                })
            
            result = {
                "product_id": product_id,
                "threshold": threshold,
                "total_suspicious": len(suspicious_reviews),
                "patterns": patterns,
                "suspicious_reviews": suspicious_data[:20],  # Limit response size
                "risk_level": "high" if len(suspicious_reviews) > 10 else "medium" if len(suspicious_reviews) > 3 else "low"
            }
            
            return result
            
        finally:
            db.close()
        
    except Exception as e:
        logger.error(f"Error detecting fake reviews: {e}")
        raise


@celery_app.task(bind=True, base=CallbackTask)
def generate_periodic_reports(
    self,
    report_type: str = "weekly",
    product_ids: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Generate periodic review analysis reports
    """
    try:
        logger.info(f"Generating {report_type} review reports")
        
        from datetime import datetime, timedelta
        from src.models.review import Review, ReviewAnalysis, SentimentHistory
        from sqlalchemy import func, desc
        
        db = SessionLocal()
        try:
            # Calculate time window
            if report_type == "daily":
                since = datetime.utcnow() - timedelta(days=1)
            elif report_type == "weekly":
                since = datetime.utcnow() - timedelta(weeks=1)
            elif report_type == "monthly":
                since = datetime.utcnow() - timedelta(days=30)
            else:
                since = datetime.utcnow() - timedelta(weeks=1)  # Default to weekly
            
            # Build base query
            query = db.query(Review).filter(Review.created_at >= since)
            
            if product_ids:
                query = query.filter(Review.product_id.in_(product_ids))
            
            reviews = query.all()
            
            if not reviews:
                return {"message": "No reviews found for the specified period"}
            
            # Calculate metrics
            total_reviews = len(reviews)
            avg_rating = sum(r.rating for r in reviews) / total_reviews
            
            # Sentiment distribution
            sentiment_counts = {"positive": 0, "neutral": 0, "negative": 0}
            analyses = db.query(ReviewAnalysis).filter(
                ReviewAnalysis.review_id.in_([str(r.id) for r in reviews])
            ).all()
            
            for analysis in analyses:
                sentiment_counts[analysis.overall_sentiment] += 1
            
            # Top products by review volume
            product_counts = {}
            for review in reviews:
                pid = str(review.product_id)
                product_counts[pid] = product_counts.get(pid, 0) + 1
            
            top_products = sorted(product_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            
            # Trending sentiment changes
            trending_products = []
            if len(reviews) > 5:  # Only calculate trends with sufficient data
                for product_id in set(str(r.product_id) for r in reviews):
                    # Get sentiment trend for this product
                    sentiment_history = db.query(SentimentHistory).filter(
                        and_(
                            SentimentHistory.product_id == product_id,
                            SentimentHistory.date >= since
                        )
                    ).order_by(SentimentHistory.date).all()
                    
                    if len(sentiment_history) >= 3:
                        # Simple trend calculation
                        recent_scores = [h.sentiment_score for h in sentiment_history[-3:]]
                        older_scores = [h.sentiment_score for h in sentiment_history[:-3]] if len(sentiment_history) > 3 else [0]
                        
                        recent_avg = sum(recent_scores) / len(recent_scores)
                        older_avg = sum(older_scores) / len(older_scores) if older_scores != [0] else recent_avg
                        
                        change = recent_avg - older_avg
                        
                        if abs(change) > 0.2:  # Significant change
                            trending_products.append({
                                "product_id": product_id,
                                "sentiment_change": round(change, 3),
                                "trend": "improving" if change > 0 else "declining"
                            })
            
            # Quality metrics
            verified_ratio = sum(1 for r in reviews if r.verified_purchase) / total_reviews
            avg_word_count = sum(r.word_count for r in reviews) / total_reviews
            
            # Fake review detection summary
            suspicious_count = sum(1 for a in analyses if a.authenticity_score < 0.5)
            
            report = {
                "report_type": report_type,
                "period_start": since.isoformat(),
                "period_end": datetime.utcnow().isoformat(),
                "summary": {
                    "total_reviews": total_reviews,
                    "average_rating": round(avg_rating, 2),
                    "sentiment_distribution": sentiment_counts,
                    "verified_purchase_ratio": round(verified_ratio, 3),
                    "average_word_count": round(avg_word_count, 1),
                    "suspicious_reviews": suspicious_count
                },
                "top_products": [{"product_id": pid, "review_count": count} for pid, count in top_products],
                "trending_products": trending_products[:10],
                "generated_at": datetime.utcnow().isoformat()
            }
            
            return report
            
        finally:
            db.close()
        
    except Exception as e:
        logger.error(f"Error generating {report_type} reports: {e}")
        raise


# Schedule periodic tasks
@celery_app.task(bind=True, base=CallbackTask)
def cleanup_old_sentiment_history(self, days_to_keep: int = 365) -> Dict[str, Any]:
    """
    Clean up old sentiment history records
    """
    try:
        logger.info(f"Cleaning up sentiment history older than {days_to_keep} days")
        
        from datetime import datetime, timedelta
        from src.models.review import SentimentHistory
        
        cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)
        
        db = SessionLocal()
        try:
            # Delete old records
            deleted = db.query(SentimentHistory).filter(
                SentimentHistory.date < cutoff_date
            ).delete()
            
            db.commit()
            
            return {
                "cleaned_up": True,
                "records_deleted": deleted,
                "cutoff_date": cutoff_date.isoformat()
            }
            
        finally:
            db.close()
        
    except Exception as e:
        logger.error(f"Error cleaning up sentiment history: {e}")
        raise


# Legacy task wrappers for backward compatibility
@celery_app.task(bind=True, base=CallbackTask)
def analyze_product_reviews(
    self,
    product_id: str,
    review_texts: List[str]
) -> Dict[str, Any]:
    """
    Legacy wrapper - now processes actual reviews using new system
    """
    try:
        logger.info(f"Processing {len(review_texts)} reviews for product {product_id} (legacy API)")
        
        # Convert to new system by triggering batch processing
        result = batch_process_reviews.delay(
            product_id=product_id,
            limit=len(review_texts),
            only_unprocessed=False
        )
        
        return {
            "product_id": product_id,
            "total_reviews": len(review_texts),
            "processing_task": str(result.id),
            "status": "processing_started",
            "message": "Reviews are being processed with the new analysis system"
        }
        
    except Exception as e:
        logger.error(f"Error in legacy review analysis for product {product_id}: {e}")
        raise


@celery_app.task(bind=True, base=CallbackTask)
def monitor_review_sentiment_trends(
    self,
    product_id: str,
    time_window: str = "7d"
) -> Dict[str, Any]:
    """
    Monitor sentiment trends for a product over time
    """
    try:
        logger.info(f"Monitoring sentiment trends for {product_id} over {time_window}")
        
        from datetime import datetime, timedelta
        from src.models.review import SentimentHistory
        from sqlalchemy import and_
        
        # Parse time window
        if time_window.endswith('d'):
            days = int(time_window[:-1])
        elif time_window.endswith('w'):
            days = int(time_window[:-1]) * 7
        else:
            days = 7  # Default to 7 days
        
        since = datetime.utcnow() - timedelta(days=days)
        
        db = SessionLocal()
        try:
            # Get sentiment history
            sentiment_history = db.query(SentimentHistory).filter(
                and_(
                    SentimentHistory.product_id == product_id,
                    SentimentHistory.date >= since
                )
            ).order_by(SentimentHistory.date).all()
            
            if not sentiment_history:
                return {
                    "product_id": product_id,
                    "time_window": time_window,
                    "message": "No sentiment data available for this time period"
                }
            
            # Calculate daily averages
            daily_sentiment = {}
            for record in sentiment_history:
                date_key = record.date.date().isoformat()
                if date_key not in daily_sentiment:
                    daily_sentiment[date_key] = []
                daily_sentiment[date_key].append(record.sentiment_score)
            
            # Calculate trend
            trend_data = []
            for date, scores in sorted(daily_sentiment.items()):
                avg_score = sum(scores) / len(scores)
                positive_ratio = len([s for s in scores if s > 0.3]) / len(scores)
                negative_ratio = len([s for s in scores if s < -0.3]) / len(scores)
                
                trend_data.append({
                    "date": date,
                    "avg_sentiment": round(avg_score, 3),
                    "positive_ratio": round(positive_ratio, 3),
                    "negative_ratio": round(negative_ratio, 3),
                    "review_count": len(scores)
                })
            
            # Determine overall trend direction
            if len(trend_data) >= 3:
                recent_avg = sum(day["avg_sentiment"] for day in trend_data[-3:]) / 3
                earlier_avg = sum(day["avg_sentiment"] for day in trend_data[:-3]) / len(trend_data[:-3]) if len(trend_data) > 3 else recent_avg
                
                change = recent_avg - earlier_avg
                if change > 0.1:
                    trend_direction = "improving"
                elif change < -0.1:
                    trend_direction = "declining"
                else:
                    trend_direction = "stable"
            else:
                trend_direction = "insufficient_data"
            
            return {
                "product_id": product_id,
                "time_window": time_window,
                "sentiment_trend": trend_data,
                "trend_direction": trend_direction,
                "total_reviews_analyzed": len(sentiment_history),
                "monitored_at": datetime.utcnow().isoformat()
            }
            
        finally:
            db.close()
        
    except Exception as e:
        logger.error(f"Error monitoring sentiment trends for {product_id}: {e}")
        raise