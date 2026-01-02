import asyncio
import json
import pandas as pd
from typing import List, Dict, Any, Optional
from loguru import logger
from sqlalchemy.orm import Session
import httpx
from datetime import datetime, timezone
import re
from dateutil import parser

from src.core.database import get_db
from src.core.config import settings
from src.models.product import Product
from src.models.review import Review, ReviewAnalysis
from src.services.text_processing import TextProcessor


class ReviewIngestionService:
    """
    Service for ingesting review data from various sources
    """
    
    def __init__(self):
        self.text_processor = TextProcessor()
        self.supported_sources = {
            "amazon": self._process_amazon_reviews,
            "csv": self._process_csv_reviews,
            "json": self._process_json_reviews,
            "manual": self._process_manual_reviews
        }
    
    async def ingest_reviews(
        self,
        source: str,
        data_path: Optional[str] = None,
        data_url: Optional[str] = None,
        product_filter: Optional[List[str]] = None,
        batch_size: int = 100
    ) -> Dict[str, Any]:
        """
        Main method to ingest reviews from various sources
        """
        if source not in self.supported_sources:
            raise ValueError(f"Unsupported data source: {source}")
        
        logger.info(f"Starting review ingestion from {source}")
        start_time = datetime.now()
        
        try:
            # Load raw data
            raw_data = await self._load_data(source, data_path, data_url)
            
            # Process data according to source format
            processed_reviews = await self.supported_sources[source](raw_data, product_filter)
            
            # Ingest in batches
            results = await self._batch_ingest(processed_reviews, batch_size)
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            logger.info(f"Review ingestion completed in {duration:.2f} seconds")
            
            return {
                "source": source,
                "total_processed": len(processed_reviews),
                "successful": results["successful"],
                "failed": results["failed"],
                "duplicates_skipped": results["duplicates_skipped"],
                "duration_seconds": duration,
                "errors": results["errors"]
            }
            
        except Exception as e:
            logger.error(f"Review ingestion failed: {e}")
            raise
    
    async def _load_data(
        self,
        source: str,
        data_path: Optional[str] = None,
        data_url: Optional[str] = None
    ) -> Any:
        """
        Load data from file or URL
        """
        if data_url:
            async with httpx.AsyncClient() as client:
                response = await client.get(data_url, timeout=60.0)
                response.raise_for_status()
                
                if data_url.endswith('.csv'):
                    from io import StringIO
                    return pd.read_csv(StringIO(response.text))
                elif data_url.endswith('.json'):
                    return response.json()
                else:
                    return response.text
        
        elif data_path:
            if data_path.endswith('.csv'):
                return pd.read_csv(data_path)
            elif data_path.endswith('.json'):
                with open(data_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                with open(data_path, 'r', encoding='utf-8') as f:
                    return f.read()
        
        else:
            raise ValueError("Either data_path or data_url must be provided")
    
    async def _process_amazon_reviews(
        self, 
        raw_data: Any, 
        product_filter: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Process Amazon review dataset
        Expected format: CSV or JSON with Amazon review fields
        """
        reviews = []
        
        if isinstance(raw_data, pd.DataFrame):
            for _, row in raw_data.iterrows():
                # Filter by product if specified
                product_asin = row.get("asin") or row.get("product_id")
                if product_filter and product_asin not in product_filter:
                    continue
                
                # Parse review date
                review_date = self._parse_date(
                    row.get("reviewTime") or row.get("date") or row.get("timestamp")
                )
                
                # Clean and validate content
                content = self._clean_review_text(row.get("reviewText", ""))
                title = self._clean_review_text(row.get("summary", ""))
                
                if not content or len(content.strip()) < 10:
                    continue  # Skip reviews with no meaningful content
                
                reviews.append({
                    "product_external_id": product_asin,
                    "external_id": row.get("reviewerID", "") + "_" + str(row.get("unixReviewTime", "")),
                    "title": title,
                    "content": content,
                    "rating": float(row.get("overall", 0)),
                    "reviewer_name": row.get("reviewerName", "Anonymous"),
                    "verified_purchase": bool(row.get("verified", False)),
                    "helpful_votes": int(row.get("helpful", [0, 0])[0]) if isinstance(row.get("helpful"), list) else 0,
                    "total_votes": int(row.get("helpful", [0, 0])[1]) if isinstance(row.get("helpful"), list) else 0,
                    "review_date": review_date,
                    "data_source": "amazon"
                })
        
        elif isinstance(raw_data, list):
            for item in raw_data:
                product_asin = item.get("asin") or item.get("product_id")
                if product_filter and product_asin not in product_filter:
                    continue
                
                review_date = self._parse_date(
                    item.get("reviewTime") or item.get("date") or item.get("timestamp")
                )
                
                content = self._clean_review_text(item.get("reviewText", ""))
                title = self._clean_review_text(item.get("summary", ""))
                
                if not content or len(content.strip()) < 10:
                    continue
                
                reviews.append({
                    "product_external_id": product_asin,
                    "external_id": item.get("reviewerID", "") + "_" + str(item.get("unixReviewTime", "")),
                    "title": title,
                    "content": content,
                    "rating": float(item.get("overall", 0)),
                    "reviewer_name": item.get("reviewerName", "Anonymous"),
                    "verified_purchase": bool(item.get("verified", False)),
                    "helpful_votes": item.get("helpful", {}).get("yes", 0) if isinstance(item.get("helpful"), dict) else 0,
                    "total_votes": item.get("helpful", {}).get("total", 0) if isinstance(item.get("helpful"), dict) else 0,
                    "review_date": review_date,
                    "data_source": "amazon"
                })
        
        return reviews
    
    async def _process_csv_reviews(
        self, 
        raw_data: pd.DataFrame, 
        product_filter: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Process generic CSV review data
        """
        reviews = []
        
        for _, row in raw_data.iterrows():
            product_id = row.get("product_id") or row.get("asin") or row.get("sku")
            if product_filter and product_id not in product_filter:
                continue
            
            content = self._clean_review_text(row.get("content") or row.get("review_text", ""))
            title = self._clean_review_text(row.get("title") or row.get("summary", ""))
            
            if not content or len(content.strip()) < 10:
                continue
            
            review_date = self._parse_date(
                row.get("date") or row.get("review_date") or row.get("timestamp")
            )
            
            reviews.append({
                "product_external_id": product_id,
                "external_id": row.get("review_id", f"csv_{len(reviews)}"),
                "title": title,
                "content": content,
                "rating": float(row.get("rating", 0)),
                "reviewer_name": row.get("reviewer_name", "Anonymous"),
                "verified_purchase": bool(row.get("verified_purchase", False)),
                "helpful_votes": int(row.get("helpful_votes", 0)),
                "total_votes": int(row.get("total_votes", 0)),
                "review_date": review_date,
                "data_source": "csv"
            })
        
        return reviews
    
    async def _process_json_reviews(
        self, 
        raw_data: Any, 
        product_filter: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Process JSON review data
        """
        if isinstance(raw_data, list):
            return await self._process_review_list(raw_data, product_filter)
        elif isinstance(raw_data, dict):
            if "reviews" in raw_data:
                return await self._process_review_list(raw_data["reviews"], product_filter)
            else:
                return await self._process_review_list([raw_data], product_filter)
        else:
            return []
    
    async def _process_review_list(
        self, 
        review_list: List[Dict], 
        product_filter: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Process a list of review dictionaries
        """
        reviews = []
        
        for item in review_list:
            product_id = item.get("product_id") or item.get("asin") or item.get("sku")
            if product_filter and product_id not in product_filter:
                continue
            
            content = self._clean_review_text(item.get("content") or item.get("text", ""))
            title = self._clean_review_text(item.get("title") or item.get("summary", ""))
            
            if not content or len(content.strip()) < 10:
                continue
            
            review_date = self._parse_date(
                item.get("date") or item.get("review_date") or item.get("timestamp")
            )
            
            reviews.append({
                "product_external_id": product_id,
                "external_id": item.get("review_id", f"json_{len(reviews)}"),
                "title": title,
                "content": content,
                "rating": float(item.get("rating", 0)),
                "reviewer_name": item.get("reviewer_name", "Anonymous"),
                "verified_purchase": bool(item.get("verified_purchase", False)),
                "helpful_votes": int(item.get("helpful_votes", 0)),
                "total_votes": int(item.get("total_votes", 0)),
                "review_date": review_date,
                "data_source": "json"
            })
        
        return reviews
    
    async def _process_manual_reviews(
        self, 
        raw_data: Any, 
        product_filter: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Process manually submitted reviews
        """
        return await self._process_json_reviews(raw_data, product_filter)
    
    def _clean_review_text(self, text: str) -> str:
        """
        Clean and normalize review text
        """
        if not text:
            return ""
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove very long lines (likely spam/malformed)
        lines = text.split('\n')
        cleaned_lines = [line for line in lines if len(line) < 1000]
        text = '\n'.join(cleaned_lines)
        
        return text
    
    def _parse_date(self, date_str: Any) -> datetime:
        """
        Parse various date formats
        """
        if not date_str:
            return datetime.now(timezone.utc)
        
        # Handle Unix timestamp
        if isinstance(date_str, (int, float)):
            return datetime.fromtimestamp(date_str, tz=timezone.utc)
        
        # Handle string dates
        if isinstance(date_str, str):
            try:
                # Try common formats
                for fmt in ['%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%Y-%m-%d %H:%M:%S']:
                    try:
                        return datetime.strptime(date_str, fmt).replace(tzinfo=timezone.utc)
                    except ValueError:
                        continue
                
                # Use dateutil parser as fallback
                parsed = parser.parse(date_str)
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=timezone.utc)
                return parsed
                
            except Exception:
                logger.warning(f"Could not parse date: {date_str}")
                return datetime.now(timezone.utc)
        
        return datetime.now(timezone.utc)
    
    async def _batch_ingest(self, reviews: List[Dict[str, Any]], batch_size: int) -> Dict[str, Any]:
        """
        Ingest reviews in batches
        """
        successful = 0
        failed = 0
        duplicates_skipped = 0
        errors = []
        
        db = next(get_db())
        
        try:
            for i in range(0, len(reviews), batch_size):
                batch = reviews[i:i + batch_size]
                
                for review_data in batch:
                    try:
                        result = await self._ingest_single_review(db, review_data)
                        if result == "success":
                            successful += 1
                        elif result == "duplicate":
                            duplicates_skipped += 1
                        else:
                            failed += 1
                    except Exception as e:
                        failed += 1
                        errors.append({
                            "external_id": review_data.get("external_id", "unknown"),
                            "error": str(e)
                        })
                        logger.error(f"Failed to ingest review {review_data.get('external_id')}: {e}")
                
                # Commit batch
                db.commit()
                logger.info(f"Processed batch {i//batch_size + 1}, reviews {i+1}-{min(i+batch_size, len(reviews))}")
        
        finally:
            db.close()
        
        return {
            "successful": successful,
            "failed": failed,
            "duplicates_skipped": duplicates_skipped,
            "errors": errors
        }
    
    async def _ingest_single_review(self, db: Session, review_data: Dict[str, Any]) -> str:
        """
        Ingest a single review into the database
        """
        # Check if review already exists
        existing_review = db.query(Review).filter(
            Review.external_id == review_data["external_id"]
        ).first()
        
        if existing_review:
            return "duplicate"
        
        # Find product by external ID
        product = db.query(Product).filter(
            Product.external_id == review_data["product_external_id"]
        ).first()
        
        if not product:
            # Try to find by SKU as fallback
            product = db.query(Product).filter(
                Product.sku == review_data["product_external_id"]
            ).first()
        
        if not product:
            logger.warning(f"Product not found for external_id: {review_data['product_external_id']}")
            raise ValueError(f"Product not found: {review_data['product_external_id']}")
        
        # Create review
        review = Review(
            product_id=product.id,
            external_id=review_data["external_id"],
            title=review_data["title"],
            content=review_data["content"],
            rating=review_data["rating"],
            reviewer_name=review_data["reviewer_name"],
            verified_purchase=review_data["verified_purchase"],
            helpful_votes=review_data["helpful_votes"],
            total_votes=review_data["total_votes"],
            review_date=review_data["review_date"],
            data_source=review_data["data_source"],
            word_count=len(review_data["content"].split()),
            language="en"  # TODO: Add language detection
        )
        
        db.add(review)
        db.flush()  # Get the ID
        
        return "success"