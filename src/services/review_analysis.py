import asyncio
from typing import Dict, List, Any, Optional
from loguru import logger
from sqlalchemy.orm import Session
from sqlalchemy import and_, desc, func
from datetime import datetime, timedelta
import json

from src.core.database import get_db, redis_client
from src.models.product import Product
from src.models.review import Review, ReviewAnalysis, ReviewSummary, SentimentHistory, ReviewAlert
from src.services.sentiment_analysis import SentimentAnalysisService
from src.services.text_processing import TextProcessor
from src.services.review_summarization import ReviewSummarizationService


class ReviewAnalysisService:
    """
    Comprehensive review analysis service
    """
    
    def __init__(self):
        self.sentiment_service = SentimentAnalysisService()
        self.text_processor = TextProcessor()
        self.summarization_service = ReviewSummarizationService()
    
    async def process_review(self, review_id: str) -> Dict[str, Any]:
        """
        Process a single review with comprehensive analysis
        """
        db = next(get_db())
        try:
            # Get review
            review = db.query(Review).filter(Review.id == review_id).first()
            if not review:
                raise ValueError(f"Review {review_id} not found")
            
            # Check if already processed
            existing_analysis = db.query(ReviewAnalysis).filter(
                ReviewAnalysis.review_id == review_id
            ).first()
            
            if existing_analysis:
                logger.info(f"Review {review_id} already processed, updating...")
            
            # Perform comprehensive analysis
            start_time = datetime.utcnow()
            
            # Sentiment analysis
            sentiment_result = await self.sentiment_service.analyze_sentiment(
                review.content, include_aspects=True
            )
            
            # Text processing
            processed_text = await self.text_processor.preprocess_text(review.content)
            themes = await self.text_processor.extract_themes(review.content)
            pros_cons = await self.text_processor.extract_pros_and_cons(review.content)
            keywords = sentiment_result["keywords"]
            
            # Quality scoring
            readability_score = processed_text["readability_score"]
            authenticity_score = await self._calculate_authenticity_score(review)
            helpfulness_predicted = await self._predict_helpfulness(review, sentiment_result)
            
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            # Create or update analysis
            if existing_analysis:
                analysis = existing_analysis
            else:
                analysis = ReviewAnalysis(review_id=review.id)
            
            # Update analysis data
            analysis.overall_sentiment = sentiment_result["overall_sentiment"]
            analysis.sentiment_score = sentiment_result["sentiment_score"]
            analysis.sentiment_confidence = sentiment_result["confidence"]
            analysis.aspects = sentiment_result["aspect_sentiments"]
            analysis.themes = themes
            analysis.keywords = keywords
            analysis.pros = pros_cons["pros"]
            analysis.cons = pros_cons["cons"]
            analysis.readability_score = readability_score
            analysis.authenticity_score = authenticity_score
            analysis.helpfulness_predicted = helpfulness_predicted
            analysis.model_version = "v1.0"
            analysis.processing_time = processing_time
            analysis.processed_at = datetime.utcnow()
            
            if not existing_analysis:
                db.add(analysis)
            
            # Update review status
            review.is_processed = True
            
            # Create sentiment history entry
            sentiment_history = SentimentHistory(
                product_id=review.product_id,
                review_id=review.id,
                sentiment=sentiment_result["overall_sentiment"],
                sentiment_score=sentiment_result["sentiment_score"],
                rating=review.rating,
                date=review.review_date,
                week_start=self._get_week_start(review.review_date),
                month_start=self._get_month_start(review.review_date)
            )
            db.add(sentiment_history)
            
            db.commit()
            
            # Update product summary asynchronously
            asyncio.create_task(self._update_product_summary(str(review.product_id)))
            
            logger.info(f"Review {review_id} processed successfully in {processing_time:.2f}s")
            
            return {
                "review_id": str(review_id),
                "overall_sentiment": sentiment_result["overall_sentiment"],
                "sentiment_score": sentiment_result["sentiment_score"],
                "confidence": sentiment_result["confidence"],
                "themes_count": len(themes),
                "aspects_analyzed": len(sentiment_result["aspect_sentiments"]),
                "processing_time": processing_time,
                "authenticity_score": authenticity_score,
                "helpfulness_predicted": helpfulness_predicted
            }
            
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to process review {review_id}: {e}")
            raise
        finally:
            db.close()
    
    async def batch_process_reviews(
        self, 
        product_id: Optional[str] = None,
        limit: int = 100,
        only_unprocessed: bool = True
    ) -> Dict[str, Any]:
        """
        Process multiple reviews in batch
        """
        db = next(get_db())
        try:
            # Build query
            query = db.query(Review)
            
            if product_id:
                query = query.filter(Review.product_id == product_id)
            
            if only_unprocessed:
                query = query.filter(Review.is_processed == False)
            
            reviews = query.order_by(desc(Review.created_at)).limit(limit).all()
            
            logger.info(f"Processing {len(reviews)} reviews in batch")
            
            # Process reviews concurrently (with limits)
            semaphore = asyncio.Semaphore(5)  # Limit concurrent processing
            
            async def process_with_semaphore(review):
                async with semaphore:
                    try:
                        return await self.process_review(str(review.id))
                    except Exception as e:
                        logger.error(f"Failed to process review {review.id}: {e}")
                        return {"error": str(e), "review_id": str(review.id)}
            
            results = await asyncio.gather(*[
                process_with_semaphore(review) for review in reviews
            ])
            
            # Summarize results
            successful = len([r for r in results if "error" not in r])
            failed = len([r for r in results if "error" in r])
            errors = [r for r in results if "error" in r]
            
            return {
                "total_processed": len(reviews),
                "successful": successful,
                "failed": failed,
                "errors": errors
            }
            
        finally:
            db.close()
    
    async def _calculate_authenticity_score(self, review: Review) -> float:
        """
        Calculate authenticity score to detect fake reviews
        """
        score = 1.0
        
        # Length check (very short or very long reviews might be suspicious)
        word_count = len(review.content.split())
        if word_count < 10:
            score -= 0.3
        elif word_count > 500:
            score -= 0.2
        
        # Verified purchase bonus
        if review.verified_purchase:
            score += 0.2
        
        # Rating extremes (all 1s or 5s might be suspicious)
        if review.rating in [1.0, 5.0]:
            score -= 0.1
        
        # Generic content check (simple heuristic)
        generic_phrases = [
            "great product", "highly recommend", "waste of money", 
            "don't buy", "perfect", "terrible", "amazing"
        ]
        generic_count = sum(1 for phrase in generic_phrases 
                          if phrase in review.content.lower())
        if generic_count > 2:
            score -= 0.2
        
        # Reviewer name check (very generic names might be suspicious)
        if review.reviewer_name and review.reviewer_name.lower() in [
            "anonymous", "customer", "user", "buyer"
        ]:
            score -= 0.1
        
        return max(0.0, min(1.0, score))
    
    async def _predict_helpfulness(
        self, 
        review: Review, 
        sentiment_result: Dict[str, Any]
    ) -> float:
        """
        Predict how helpful this review will be to other customers
        """
        score = 0.5  # Base score
        
        # Length factor (moderate length is usually more helpful)
        word_count = sentiment_result.get("word_count", 0)
        if 50 <= word_count <= 200:
            score += 0.2
        elif word_count < 20:
            score -= 0.2
        
        # Readability factor
        readability = sentiment_result.get("readability_score", 50)
        if readability > 60:
            score += 0.15
        elif readability < 30:
            score -= 0.15
        
        # Specificity (presence of aspects and themes)
        aspects_count = len(sentiment_result.get("aspect_sentiments", {}))
        if aspects_count > 2:
            score += 0.2
        
        # Balanced sentiment (not extremely positive or negative)
        sentiment_score = abs(sentiment_result.get("sentiment_score", 0))
        if 0.2 <= sentiment_score <= 0.8:
            score += 0.1
        
        # Verified purchase
        if review.verified_purchase:
            score += 0.15
        
        return max(0.0, min(1.0, score))
    
    def _get_week_start(self, date: datetime) -> datetime:
        """Get the start of the week for a given date"""
        days_since_monday = date.weekday()
        week_start = date - timedelta(days=days_since_monday)
        return week_start.replace(hour=0, minute=0, second=0, microsecond=0)
    
    def _get_month_start(self, date: datetime) -> datetime:
        """Get the start of the month for a given date"""
        return date.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    
    async def _update_product_summary(self, product_id: str):
        """
        Update product review summary asynchronously
        """
        try:
            db = next(get_db())
            
            # Get all reviews for this product
            reviews = db.query(Review).filter(
                and_(
                    Review.product_id == product_id,
                    Review.is_processed == True
                )
            ).all()
            
            if not reviews:
                return
            
            # Get all analyses
            review_ids = [str(r.id) for r in reviews]
            analyses = db.query(ReviewAnalysis).filter(
                ReviewAnalysis.review_id.in_(review_ids)
            ).all()
            
            # Calculate summary statistics
            total_reviews = len(reviews)
            average_rating = sum(r.rating for r in reviews) / total_reviews
            
            # Rating distribution
            rating_dist = {str(i): 0 for i in range(1, 6)}
            for review in reviews:
                rating_key = str(int(review.rating))
                rating_dist[rating_key] += 1
            
            # Sentiment distribution
            sentiment_counts = {"positive": 0, "neutral": 0, "negative": 0}
            sentiment_scores = []
            
            for analysis in analyses:
                sentiment_counts[analysis.overall_sentiment] += 1
                sentiment_scores.append(analysis.sentiment_score)
            
            sentiment_dist = {
                k: round((v / total_reviews) * 100, 1) 
                for k, v in sentiment_counts.items()
            }
            
            # Overall sentiment
            avg_sentiment_score = sum(sentiment_scores) / len(sentiment_scores)
            if avg_sentiment_score > 0.3:
                overall_sentiment = "positive"
            elif avg_sentiment_score < -0.3:
                overall_sentiment = "negative"
            else:
                overall_sentiment = "neutral"
            
            # Sentiment trend
            sentiment_trend_data = await self.sentiment_service.calculate_sentiment_trends(
                [{"processed_at": a.processed_at.isoformat(), "sentiment_score": a.sentiment_score} 
                 for a in analyses]
            )
            
            # Generate summary text
            summary_text = await self.summarization_service.generate_product_summary(
                product_id, reviews, analyses
            )
            
            # Extract common themes, pros, and cons
            all_themes = []
            all_pros = []
            all_cons = []
            
            for analysis in analyses:
                all_themes.extend(analysis.themes or [])
                all_pros.extend(analysis.pros or [])
                all_cons.extend(analysis.cons or [])
            
            # Get top themes/pros/cons
            from collections import Counter
            top_themes = [item for item, count in Counter(all_themes).most_common(10)]
            common_pros = [item for item, count in Counter(all_pros).most_common(5)]
            common_cons = [item for item, count in Counter(all_cons).most_common(5)]
            
            # Calculate aspect ratings
            aspect_ratings = {}
            aspect_scores = {}
            
            for analysis in analyses:
                for aspect, data in (analysis.aspects or {}).items():
                    if aspect not in aspect_scores:
                        aspect_scores[aspect] = []
                    aspect_scores[aspect].append(data.get("score", 0))
            
            for aspect, scores in aspect_scores.items():
                # Convert sentiment scores to 1-5 rating scale
                avg_score = sum(scores) / len(scores)
                rating = 3 + (avg_score * 2)  # Convert from -1,1 to 1,5
                aspect_ratings[aspect] = round(max(1, min(5, rating)), 1)
            
            # Quality metrics
            verified_ratio = sum(1 for r in reviews if r.verified_purchase) / total_reviews
            avg_helpfulness = sum(r.helpful_votes / max(r.total_votes, 1) for r in reviews if r.total_votes > 0)
            if avg_helpfulness > 0:
                avg_helpfulness = avg_helpfulness / sum(1 for r in reviews if r.total_votes > 0)
            else:
                avg_helpfulness = 0.0
            
            fake_review_ratio = sum(1 for a in analyses if a.authenticity_score < 0.5) / total_reviews
            
            # Update or create summary
            summary = db.query(ReviewSummary).filter(
                ReviewSummary.product_id == product_id
            ).first()
            
            if not summary:
                summary = ReviewSummary(product_id=product_id)
                db.add(summary)
            
            # Update summary data
            summary.total_reviews = total_reviews
            summary.average_rating = round(average_rating, 2)
            summary.rating_distribution = rating_dist
            summary.sentiment_distribution = sentiment_dist
            summary.overall_sentiment = overall_sentiment
            summary.sentiment_trend = sentiment_trend_data["trend"]
            summary.summary_text = summary_text
            summary.top_themes = top_themes
            summary.common_pros = common_pros
            summary.common_cons = common_cons
            summary.aspect_ratings = aspect_ratings
            summary.verified_purchase_ratio = round(verified_ratio, 3)
            summary.average_helpfulness = round(avg_helpfulness, 3)
            summary.fake_review_ratio = round(fake_review_ratio, 3)
            summary.last_updated = datetime.utcnow()
            summary.last_review_date = max(r.review_date for r in reviews)
            
            db.commit()
            
            # Check for alerts
            await self._check_sentiment_alerts(product_id, sentiment_trend_data)
            
            # Cache the summary
            try:
                cache_key = f"review_summary:{product_id}"
                summary_data = {
                    "total_reviews": total_reviews,
                    "average_rating": average_rating,
                    "overall_sentiment": overall_sentiment,
                    "sentiment_trend": sentiment_trend_data["trend"],
                    "top_themes": top_themes[:5],
                    "common_pros": common_pros[:3],
                    "common_cons": common_cons[:3]
                }
                redis_client.setex(cache_key, 3600, json.dumps(summary_data))
            except Exception as e:
                logger.warning(f"Failed to cache review summary: {e}")
            
            logger.info(f"Updated review summary for product {product_id}")
            
        except Exception as e:
            logger.error(f"Failed to update product summary for {product_id}: {e}")
        finally:
            db.close()
    
    async def _check_sentiment_alerts(self, product_id: str, trend_data: Dict[str, Any]):
        """
        Check for sentiment alerts and create them if needed
        """
        try:
            if trend_data["trend"] == "declining" and trend_data["confidence"] > 0.7:
                db = next(get_db())
                
                # Check if alert already exists
                existing_alert = db.query(ReviewAlert).filter(
                    and_(
                        ReviewAlert.product_id == product_id,
                        ReviewAlert.alert_type == "sentiment_decline",
                        ReviewAlert.is_active == True
                    )
                ).first()
                
                if not existing_alert:
                    # Create new alert
                    alert = ReviewAlert(
                        product_id=product_id,
                        alert_type="sentiment_decline",
                        severity="high" if trend_data["confidence"] > 0.9 else "medium",
                        title="Declining Review Sentiment",
                        description=f"Product reviews show declining sentiment with {trend_data['confidence']:.1%} confidence",
                        current_value=trend_data["change"],
                        threshold=-0.1,
                        triggered_at=datetime.utcnow()
                    )
                    db.add(alert)
                    db.commit()
                    logger.info(f"Created sentiment decline alert for product {product_id}")
                
                db.close()
                
        except Exception as e:
            logger.error(f"Failed to check sentiment alerts for {product_id}: {e}")
    
    async def get_product_analysis(self, product_id: str) -> Dict[str, Any]:
        """
        Get comprehensive analysis for a product
        """
        # Try cache first
        cache_key = f"review_summary:{product_id}"
        try:
            cached = redis_client.get(cache_key)
            if cached:
                return json.loads(cached)
        except Exception:
            pass
        
        db = next(get_db())
        try:
            summary = db.query(ReviewSummary).filter(
                ReviewSummary.product_id == product_id
            ).first()
            
            if not summary:
                return {"error": "No analysis available for this product"}
            
            return {
                "total_reviews": summary.total_reviews,
                "average_rating": summary.average_rating,
                "rating_distribution": summary.rating_distribution,
                "sentiment_distribution": summary.sentiment_distribution,
                "overall_sentiment": summary.overall_sentiment,
                "sentiment_trend": summary.sentiment_trend,
                "summary_text": summary.summary_text,
                "top_themes": summary.top_themes,
                "common_pros": summary.common_pros,
                "common_cons": summary.common_cons,
                "aspect_ratings": summary.aspect_ratings,
                "verified_purchase_ratio": summary.verified_purchase_ratio,
                "average_helpfulness": summary.average_helpfulness,
                "fake_review_ratio": summary.fake_review_ratio,
                "last_updated": summary.last_updated.isoformat()
            }
            
        finally:
            db.close()