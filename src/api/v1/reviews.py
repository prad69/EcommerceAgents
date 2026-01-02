from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query
from fastapi.responses import JSONResponse
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from loguru import logger
import asyncio

from src.core.database import get_db
from src.models.review import Review, ReviewAnalysis, ReviewSummary, ReviewAlert
from src.services.review_analysis import ReviewAnalysisService
from src.services.review_ingestion import ReviewIngestionService
from src.services.sentiment_analysis import SentimentAnalysisService
from src.services.review_summarization import ReviewSummarizationService
from src.core.auth import get_current_user
from sqlalchemy.orm import Session
from sqlalchemy import desc, and_

router = APIRouter(prefix="/reviews", tags=["Review Analysis"])

# Initialize services
review_analysis_service = ReviewAnalysisService()
review_ingestion_service = ReviewIngestionService()
sentiment_service = SentimentAnalysisService()
summarization_service = ReviewSummarizationService()


# Pydantic models
class ReviewSubmission(BaseModel):
    product_id: str
    title: Optional[str] = ""
    content: str = Field(..., min_length=10, max_length=5000)
    rating: float = Field(..., ge=1, le=5)
    reviewer_name: Optional[str] = "Anonymous"
    verified_purchase: bool = False

class ReviewIngestionRequest(BaseModel):
    source: str = Field(..., description="Data source: amazon, csv, json, manual")
    data_path: Optional[str] = None
    data_url: Optional[str] = None
    product_filter: Optional[List[str]] = None
    batch_size: int = Field(default=100, ge=1, le=1000)

class SentimentAnalysisRequest(BaseModel):
    text: str = Field(..., min_length=5, max_length=10000)
    include_aspects: bool = True

class ProductComparisonRequest(BaseModel):
    product_ids: List[str] = Field(..., min_items=2, max_items=5)


@router.post("/submit")
async def submit_review(
    review: ReviewSubmission,
    background_tasks: BackgroundTasks,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Submit a new customer review
    """
    try:
        # Create review data
        review_data = {
            "product_external_id": review.product_id,
            "external_id": f"manual_{current_user.id}_{int(asyncio.get_event_loop().time())}",
            "title": review.title,
            "content": review.content,
            "rating": review.rating,
            "reviewer_name": review.reviewer_name,
            "verified_purchase": review.verified_purchase,
            "helpful_votes": 0,
            "total_votes": 0,
            "review_date": asyncio.get_event_loop().time(),
            "data_source": "manual"
        }
        
        # Ingest the review
        result = await review_ingestion_service._ingest_single_review(db, review_data)
        
        if result == "success":
            # Get the created review
            new_review = db.query(Review).filter(
                Review.external_id == review_data["external_id"]
            ).first()
            
            # Process in background
            background_tasks.add_task(
                review_analysis_service.process_review,
                str(new_review.id)
            )
            
            return {
                "message": "Review submitted successfully",
                "review_id": str(new_review.id),
                "status": "processing"
            }
        else:
            raise HTTPException(status_code=400, detail="Failed to submit review")
            
    except Exception as e:
        logger.error(f"Failed to submit review: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ingest")
async def ingest_reviews(
    request: ReviewIngestionRequest,
    background_tasks: BackgroundTasks,
    current_user = Depends(get_current_user)
):
    """
    Ingest reviews from external data source
    """
    try:
        # Start ingestion in background
        background_tasks.add_task(
            review_ingestion_service.ingest_reviews,
            request.source,
            request.data_path,
            request.data_url,
            request.product_filter,
            request.batch_size
        )
        
        return {
            "message": "Review ingestion started",
            "source": request.source,
            "status": "processing"
        }
        
    except Exception as e:
        logger.error(f"Failed to start review ingestion: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/product/{product_id}/summary")
async def get_product_review_summary(
    product_id: str,
    db: Session = Depends(get_db)
):
    """
    Get comprehensive review summary for a product
    """
    try:
        analysis = await review_analysis_service.get_product_analysis(product_id)
        
        if "error" in analysis:
            raise HTTPException(status_code=404, detail=analysis["error"])
        
        return analysis
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get product summary for {product_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/product/{product_id}/reviews")
async def get_product_reviews(
    product_id: str,
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    sentiment_filter: Optional[str] = Query(None, regex="^(positive|negative|neutral)$"),
    sort_by: str = Query("date", regex="^(date|rating|helpfulness)$"),
    verified_only: bool = Query(False),
    db: Session = Depends(get_db)
):
    """
    Get paginated reviews for a product with filtering and sorting
    """
    try:
        # Build query
        query = db.query(Review).filter(Review.product_id == product_id)
        
        # Apply filters
        if verified_only:
            query = query.filter(Review.verified_purchase == True)
        
        # Apply sentiment filter if specified
        if sentiment_filter:
            query = query.join(ReviewAnalysis).filter(
                ReviewAnalysis.overall_sentiment == sentiment_filter
            )
        
        # Apply sorting
        if sort_by == "date":
            query = query.order_by(desc(Review.review_date))
        elif sort_by == "rating":
            query = query.order_by(desc(Review.rating))
        elif sort_by == "helpfulness":
            query = query.join(ReviewAnalysis).order_by(
                desc(ReviewAnalysis.helpfulness_predicted)
            )
        
        # Get total count
        total = query.count()
        
        # Apply pagination
        reviews = query.offset(skip).limit(limit).all()
        
        # Format response
        result = {
            "total": total,
            "reviews": []
        }
        
        for review in reviews:
            review_data = {
                "id": str(review.id),
                "title": review.title,
                "content": review.content,
                "rating": review.rating,
                "reviewer_name": review.reviewer_name,
                "verified_purchase": review.verified_purchase,
                "helpful_votes": review.helpful_votes,
                "total_votes": review.total_votes,
                "review_date": review.review_date.isoformat(),
                "word_count": review.word_count
            }
            
            # Add analysis if available
            if review.analysis:
                review_data["analysis"] = {
                    "sentiment": review.analysis.overall_sentiment,
                    "sentiment_score": review.analysis.sentiment_score,
                    "confidence": review.analysis.sentiment_confidence,
                    "themes": review.analysis.themes,
                    "helpfulness_predicted": review.analysis.helpfulness_predicted,
                    "authenticity_score": review.analysis.authenticity_score
                }
            
            result["reviews"].append(review_data)
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to get reviews for product {product_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/product/{product_id}/highlights")
async def get_review_highlights(
    product_id: str,
    max_highlights: int = Query(5, ge=1, le=10),
    db: Session = Depends(get_db)
):
    """
    Get key review highlights for a product
    """
    try:
        # Get reviews and analyses
        reviews = db.query(Review).filter(
            and_(Review.product_id == product_id, Review.is_processed == True)
        ).all()
        
        if not reviews:
            return {"highlights": []}
        
        review_ids = [str(r.id) for r in reviews]
        analyses = db.query(ReviewAnalysis).filter(
            ReviewAnalysis.review_id.in_(review_ids)
        ).all()
        
        # Generate highlights
        highlights = await summarization_service.generate_review_highlights(
            reviews, analyses, max_highlights
        )
        
        return {"highlights": highlights}
        
    except Exception as e:
        logger.error(f"Failed to get highlights for product {product_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze-sentiment")
async def analyze_text_sentiment(request: SentimentAnalysisRequest):
    """
    Analyze sentiment of arbitrary text
    """
    try:
        result = await sentiment_service.analyze_sentiment(
            request.text, 
            include_aspects=request.include_aspects
        )
        return result
        
    except Exception as e:
        logger.error(f"Failed to analyze sentiment: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/process-batch")
async def process_reviews_batch(
    product_id: Optional[str] = None,
    limit: int = Query(100, ge=1, le=1000),
    only_unprocessed: bool = Query(True),
    background_tasks: BackgroundTasks = None,
    current_user = Depends(get_current_user)
):
    """
    Process reviews in batch for analysis
    """
    try:
        # Start batch processing in background
        background_tasks.add_task(
            review_analysis_service.batch_process_reviews,
            product_id,
            limit,
            only_unprocessed
        )
        
        return {
            "message": "Batch processing started",
            "product_id": product_id,
            "limit": limit,
            "only_unprocessed": only_unprocessed
        }
        
    except Exception as e:
        logger.error(f"Failed to start batch processing: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sentiment/trends/{product_id}")
async def get_sentiment_trends(
    product_id: str,
    time_window: str = Query("daily", regex="^(daily|weekly|monthly)$"),
    days_back: int = Query(30, ge=1, le=365),
    db: Session = Depends(get_db)
):
    """
    Get sentiment trends for a product over time
    """
    try:
        from datetime import datetime, timedelta
        
        # Calculate date range
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days_back)
        
        # Get sentiment history
        sentiment_history = db.query(SentimentHistory).filter(
            and_(
                SentimentHistory.product_id == product_id,
                SentimentHistory.date >= start_date,
                SentimentHistory.date <= end_date
            )
        ).order_by(SentimentHistory.date).all()
        
        if not sentiment_history:
            return {"trends": [], "summary": "No data available"}
        
        # Group by time window
        trends = {}
        for record in sentiment_history:
            if time_window == "daily":
                key = record.date.date().isoformat()
            elif time_window == "weekly":
                week_start = record.week_start.date().isoformat()
                key = f"Week of {week_start}"
            else:  # monthly
                month_start = record.month_start.date().isoformat()
                key = f"Month of {month_start}"
            
            if key not in trends:
                trends[key] = {
                    "period": key,
                    "sentiments": [],
                    "ratings": []
                }
            
            trends[key]["sentiments"].append(record.sentiment_score)
            if record.rating:
                trends[key]["ratings"].append(record.rating)
        
        # Calculate averages
        trend_data = []
        for period, data in trends.items():
            avg_sentiment = sum(data["sentiments"]) / len(data["sentiments"])
            avg_rating = sum(data["ratings"]) / len(data["ratings"]) if data["ratings"] else None
            
            trend_data.append({
                "period": period,
                "avg_sentiment_score": round(avg_sentiment, 3),
                "avg_rating": round(avg_rating, 2) if avg_rating else None,
                "review_count": len(data["sentiments"])
            })
        
        # Calculate overall trend
        sentiment_values = [{"processed_at": h.date.isoformat(), "sentiment_score": h.sentiment_score} 
                          for h in sentiment_history]
        overall_trend = await sentiment_service.calculate_sentiment_trends(sentiment_values, time_window)
        
        return {
            "trends": trend_data,
            "overall_trend": overall_trend,
            "time_window": time_window,
            "days_analyzed": days_back
        }
        
    except Exception as e:
        logger.error(f"Failed to get sentiment trends for {product_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/compare-products")
async def compare_products(request: ProductComparisonRequest):
    """
    Compare multiple products based on their review analysis
    """
    try:
        # Get summaries for all products
        product_summaries = []
        
        for product_id in request.product_ids:
            summary = await review_analysis_service.get_product_analysis(product_id)
            if "error" not in summary:
                summary["product_id"] = product_id
                product_summaries.append(summary)
        
        if len(product_summaries) < 2:
            raise HTTPException(
                status_code=400, 
                detail="Need at least 2 products with review data for comparison"
            )
        
        # Generate comparison summary
        comparison_text = await summarization_service.generate_comparison_summary(
            product_summaries
        )
        
        return {
            "comparison_summary": comparison_text,
            "products": product_summaries,
            "compared_at": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to compare products: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/alerts")
async def get_review_alerts(
    active_only: bool = Query(True),
    severity: Optional[str] = Query(None, regex="^(low|medium|high|critical)$"),
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """
    Get review-related alerts
    """
    try:
        # Build query
        query = db.query(ReviewAlert)
        
        if active_only:
            query = query.filter(ReviewAlert.is_active == True)
        
        if severity:
            query = query.filter(ReviewAlert.severity == severity)
        
        # Get total and paginated results
        total = query.count()
        alerts = query.order_by(desc(ReviewAlert.triggered_at)).offset(skip).limit(limit).all()
        
        # Format response
        result = {
            "total": total,
            "alerts": []
        }
        
        for alert in alerts:
            result["alerts"].append({
                "id": str(alert.id),
                "product_id": str(alert.product_id),
                "alert_type": alert.alert_type,
                "severity": alert.severity,
                "title": alert.title,
                "description": alert.description,
                "current_value": alert.current_value,
                "previous_value": alert.previous_value,
                "threshold": alert.threshold,
                "is_active": alert.is_active,
                "is_acknowledged": alert.is_acknowledged,
                "triggered_at": alert.triggered_at.isoformat(),
                "resolved_at": alert.resolved_at.isoformat() if alert.resolved_at else None
            })
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to get alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(
    alert_id: str,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """
    Acknowledge a review alert
    """
    try:
        alert = db.query(ReviewAlert).filter(ReviewAlert.id == alert_id).first()
        
        if not alert:
            raise HTTPException(status_code=404, detail="Alert not found")
        
        alert.is_acknowledged = True
        alert.acknowledged_by = current_user.id
        alert.acknowledged_at = datetime.utcnow()
        
        db.commit()
        
        return {"message": "Alert acknowledged successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to acknowledge alert {alert_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))