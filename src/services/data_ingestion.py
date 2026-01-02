import asyncio
import json
import pandas as pd
from typing import List, Dict, Any, Optional
from loguru import logger
from sqlalchemy.orm import Session
import httpx
from datetime import datetime

from src.core.database import get_db, vector_db
from src.core.config import settings
from src.models.product import Product, Category, Brand
from src.services.embedding import EmbeddingService


class DataIngestionService:
    """
    Service for ingesting product data from various sources
    """
    
    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.supported_sources = {
            "amazon": self._process_amazon_data,
            "bestbuy": self._process_bestbuy_data,
            "csv": self._process_csv_data,
            "json": self._process_json_data
        }
    
    async def ingest_products(
        self,
        source: str,
        data_path: Optional[str] = None,
        data_url: Optional[str] = None,
        batch_size: int = 100
    ) -> Dict[str, Any]:
        """
        Main method to ingest products from various sources
        """
        if source not in self.supported_sources:
            raise ValueError(f"Unsupported data source: {source}")
        
        logger.info(f"Starting data ingestion from {source}")
        start_time = datetime.now()
        
        try:
            # Load raw data
            raw_data = await self._load_data(source, data_path, data_url)
            
            # Process data according to source format
            processed_data = await self.supported_sources[source](raw_data)
            
            # Ingest in batches
            results = await self._batch_ingest(processed_data, batch_size)
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            logger.info(f"Data ingestion completed in {duration:.2f} seconds")
            
            return {
                "source": source,
                "total_processed": len(processed_data),
                "successful": results["successful"],
                "failed": results["failed"],
                "duration_seconds": duration,
                "errors": results["errors"]
            }
            
        except Exception as e:
            logger.error(f"Data ingestion failed: {e}")
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
                response = await client.get(data_url)
                response.raise_for_status()
                
                if data_url.endswith('.csv'):
                    return pd.read_csv(data_url)
                elif data_url.endswith('.json'):
                    return response.json()
                else:
                    return response.text
        
        elif data_path:
            if data_path.endswith('.csv'):
                return pd.read_csv(data_path)
            elif data_path.endswith('.json'):
                with open(data_path, 'r') as f:
                    return json.load(f)
            else:
                with open(data_path, 'r') as f:
                    return f.read()
        
        else:
            raise ValueError("Either data_path or data_url must be provided")
    
    async def _process_amazon_data(self, raw_data: Any) -> List[Dict[str, Any]]:
        """
        Process Amazon product dataset
        Expected format: CSV or JSON with specific Amazon fields
        """
        products = []
        
        if isinstance(raw_data, pd.DataFrame):
            for _, row in raw_data.iterrows():
                products.append({
                    "title": row.get("title", ""),
                    "description": row.get("description", ""),
                    "price": float(row.get("price", 0)) if row.get("price") else 0.0,
                    "category": row.get("category", ""),
                    "brand": row.get("brand", ""),
                    "sku": row.get("asin", f"amz_{row.name}"),
                    "external_id": row.get("asin", ""),
                    "data_source": "amazon",
                    "images": [row.get("image_url")] if row.get("image_url") else [],
                    "average_rating": float(row.get("rating", 0)) if row.get("rating") else 0.0,
                    "review_count": int(row.get("num_reviews", 0)) if row.get("num_reviews") else 0,
                    "features": row.get("features", "").split(",") if row.get("features") else []
                })
        
        elif isinstance(raw_data, list):
            for item in raw_data:
                products.append({
                    "title": item.get("title", ""),
                    "description": item.get("description", ""),
                    "price": float(item.get("price", 0)),
                    "category": item.get("category", ""),
                    "brand": item.get("brand", ""),
                    "sku": item.get("asin", f"amz_{len(products)}"),
                    "external_id": item.get("asin", ""),
                    "data_source": "amazon",
                    "images": item.get("images", []),
                    "average_rating": float(item.get("rating", 0)),
                    "review_count": int(item.get("num_reviews", 0)),
                    "features": item.get("features", [])
                })
        
        return products
    
    async def _process_bestbuy_data(self, raw_data: Any) -> List[Dict[str, Any]]:
        """
        Process Best Buy product data
        """
        products = []
        
        if isinstance(raw_data, dict) and "products" in raw_data:
            for item in raw_data["products"]:
                products.append({
                    "title": item.get("name", ""),
                    "description": item.get("longDescription", ""),
                    "short_description": item.get("shortDescription", ""),
                    "price": float(item.get("salePrice", 0)),
                    "original_price": float(item.get("regularPrice", 0)),
                    "category": item.get("categoryPath", ""),
                    "brand": item.get("manufacturer", ""),
                    "sku": item.get("sku", f"bb_{len(products)}"),
                    "external_id": item.get("sku", ""),
                    "data_source": "bestbuy",
                    "images": [item.get("image")] if item.get("image") else [],
                    "average_rating": float(item.get("customerReviewAverage", 0)),
                    "review_count": int(item.get("customerReviewCount", 0)),
                    "is_in_stock": item.get("onlineAvailability", False)
                })
        
        return products
    
    async def _process_csv_data(self, raw_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Process generic CSV data
        """
        products = []
        
        for _, row in raw_data.iterrows():
            products.append({
                "title": row.get("title") or row.get("name", ""),
                "description": row.get("description", ""),
                "price": float(row.get("price", 0)) if row.get("price") else 0.0,
                "category": row.get("category", ""),
                "brand": row.get("brand", ""),
                "sku": row.get("sku", f"csv_{row.name}"),
                "data_source": "csv",
                "stock_quantity": int(row.get("stock", 0)) if row.get("stock") else 0
            })
        
        return products
    
    async def _process_json_data(self, raw_data: Any) -> List[Dict[str, Any]]:
        """
        Process generic JSON data
        """
        if isinstance(raw_data, list):
            return raw_data
        elif isinstance(raw_data, dict) and "products" in raw_data:
            return raw_data["products"]
        else:
            return [raw_data]
    
    async def _batch_ingest(self, products: List[Dict[str, Any]], batch_size: int) -> Dict[str, Any]:
        """
        Ingest products in batches
        """
        successful = 0
        failed = 0
        errors = []
        
        db = next(get_db())
        
        try:
            for i in range(0, len(products), batch_size):
                batch = products[i:i + batch_size]
                
                for product_data in batch:
                    try:
                        await self._ingest_single_product(db, product_data)
                        successful += 1
                    except Exception as e:
                        failed += 1
                        errors.append({
                            "product_sku": product_data.get("sku", "unknown"),
                            "error": str(e)
                        })
                        logger.error(f"Failed to ingest product {product_data.get('sku')}: {e}")
                
                # Commit batch
                db.commit()
                logger.info(f"Processed batch {i//batch_size + 1}, products {i+1}-{min(i+batch_size, len(products))}")
        
        finally:
            db.close()
        
        return {
            "successful": successful,
            "failed": failed,
            "errors": errors
        }
    
    async def _ingest_single_product(self, db: Session, product_data: Dict[str, Any]):
        """
        Ingest a single product into the database
        """
        # Create or get category
        category = None
        if product_data.get("category"):
            category = self._get_or_create_category(db, product_data["category"])
        
        # Create or get brand
        brand = None
        if product_data.get("brand"):
            brand = self._get_or_create_brand(db, product_data["brand"])
        
        # Generate slug
        slug = self._generate_slug(product_data["title"])
        
        # Create product
        product = Product(
            title=product_data["title"],
            description=product_data.get("description"),
            short_description=product_data.get("short_description"),
            sku=product_data["sku"],
            price=product_data.get("price", 0.0),
            original_price=product_data.get("original_price"),
            stock_quantity=product_data.get("stock_quantity", 0),
            is_in_stock=product_data.get("is_in_stock", True),
            category_id=category.id if category else None,
            brand_id=brand.id if brand else None,
            images=product_data.get("images", []),
            primary_image=product_data.get("images", [None])[0],
            slug=slug,
            average_rating=product_data.get("average_rating", 0.0),
            review_count=product_data.get("review_count", 0),
            features=product_data.get("features", []),
            tags=product_data.get("tags", []),
            external_id=product_data.get("external_id"),
            data_source=product_data.get("data_source"),
            last_synced=datetime.now()
        )
        
        db.add(product)
        db.flush()  # Get the ID
        
        # Generate embeddings
        await self._generate_product_embeddings(product)
    
    def _get_or_create_category(self, db: Session, category_name: str) -> Category:
        """
        Get existing category or create new one
        """
        category = db.query(Category).filter(Category.name == category_name).first()
        
        if not category:
            slug = self._generate_slug(category_name)
            category = Category(
                name=category_name,
                slug=slug,
                level=0  # TODO: Implement category hierarchy
            )
            db.add(category)
            db.flush()
        
        return category
    
    def _get_or_create_brand(self, db: Session, brand_name: str) -> Brand:
        """
        Get existing brand or create new one
        """
        brand = db.query(Brand).filter(Brand.name == brand_name).first()
        
        if not brand:
            brand = Brand(name=brand_name)
            db.add(brand)
            db.flush()
        
        return brand
    
    def _generate_slug(self, text: str) -> str:
        """
        Generate URL-friendly slug from text
        """
        import re
        slug = re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())
        slug = re.sub(r'\s+', '-', slug.strip())
        return slug[:50]  # Limit length
    
    async def _generate_product_embeddings(self, product: Product):
        """
        Generate and store vector embeddings for the product using enhanced methods
        """
        try:
            # Generate enhanced product embedding
            embedding = await self.embedding_service.generate_product_embedding(
                title=product.title,
                description=product.description or "",
                features=product.features or [],
                category=product.category.name if product.category else "",
                brand=product.brand.name if product.brand else ""
            )
            
            # Store in vector database with enhanced metadata
            embedding_id = await vector_db.store_product_embedding(
                product_id=str(product.id),
                embedding=embedding,
                metadata={
                    "title": product.title,
                    "category": product.category.name if product.category else "",
                    "brand": product.brand.name if product.brand else "",
                    "price": product.price,
                    "features": product.features or [],
                    "sku": product.sku,
                    "data_source": product.data_source or "unknown"
                }
            )
            
            product.embedding_id = embedding_id
            
        except Exception as e:
            logger.warning(f"Failed to generate embeddings for product {product.sku}: {e}")