from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from sqlalchemy.exc import SQLAlchemyError
from loguru import logger
import redis
from typing import AsyncGenerator

from .config import settings

# PostgreSQL Database Setup
engine = create_engine(
    settings.database_url,
    poolclass=StaticPool,
    pool_pre_ping=True,
    pool_recycle=300,
    echo=settings.debug
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    """Dependency to get database session"""
    db = SessionLocal()
    try:
        yield db
    except SQLAlchemyError as e:
        logger.error(f"Database error: {e}")
        db.rollback()
        raise
    finally:
        db.close()


# Redis Setup
def get_redis_client():
    """Get Redis client for caching and message queue"""
    try:
        if settings.redis_password:
            redis_client = redis.from_url(
                settings.redis_url,
                password=settings.redis_password,
                decode_responses=True
            )
        else:
            redis_client = redis.from_url(
                settings.redis_url,
                decode_responses=True
            )
        
        # Test connection
        redis_client.ping()
        logger.info("Redis connection established")
        return redis_client
    except redis.ConnectionError as e:
        logger.error(f"Redis connection failed: {e}")
        raise


# Vector Database Setup
class VectorDB:
    """Vector database client wrapper for product recommendations"""
    
    def __init__(self):
        self.pinecone_client = None
        self.weaviate_client = None
        self.setup_vector_db()
    
    def setup_vector_db(self):
        """Initialize vector database clients"""
        # Try Pinecone first
        if settings.pinecone_api_key and settings.pinecone_environment:
            try:
                import pinecone
                pinecone.init(
                    api_key=settings.pinecone_api_key,
                    environment=settings.pinecone_environment
                )
                self.pinecone_client = pinecone
                
                # Create index if it doesn't exist
                if settings.pinecone_index_name not in pinecone.list_indexes():
                    pinecone.create_index(
                        settings.pinecone_index_name,
                        dimension=1536,  # OpenAI embedding dimension
                        metric="cosine"
                    )
                logger.info("Pinecone client initialized")
            except Exception as e:
                logger.warning(f"Pinecone initialization failed: {e}")
        
        # Setup Weaviate as fallback
        try:
            import weaviate
            auth_config = None
            if settings.weaviate_api_key:
                auth_config = weaviate.AuthApiKey(api_key=settings.weaviate_api_key)
            
            self.weaviate_client = weaviate.Client(
                url=settings.weaviate_url,
                auth_client_secret=auth_config
            )
            
            # Test connection and setup schema
            if self.weaviate_client.is_ready():
                self._setup_weaviate_schema()
                logger.info("Weaviate client initialized")
            else:
                logger.warning("Weaviate client not ready")
                
        except Exception as e:
            logger.warning(f"Weaviate initialization failed: {e}")
    
    def _setup_weaviate_schema(self):
        """Setup Weaviate schema for products"""
        try:
            # Check if Product class exists
            existing_classes = self.weaviate_client.schema.get()["classes"]
            product_class_exists = any(cls["class"] == "Product" for cls in existing_classes)
            
            if not product_class_exists:
                product_schema = {
                    "class": "Product",
                    "description": "Product embeddings for recommendations",
                    "vectorizer": "none",  # We'll provide vectors manually
                    "properties": [
                        {
                            "name": "productId",
                            "dataType": ["string"],
                            "description": "Product ID"
                        },
                        {
                            "name": "title",
                            "dataType": ["string"],
                            "description": "Product title"
                        },
                        {
                            "name": "category",
                            "dataType": ["string"],
                            "description": "Product category"
                        },
                        {
                            "name": "brand",
                            "dataType": ["string"],
                            "description": "Product brand"
                        },
                        {
                            "name": "price",
                            "dataType": ["number"],
                            "description": "Product price"
                        },
                        {
                            "name": "features",
                            "dataType": ["string[]"],
                            "description": "Product features"
                        }
                    ]
                }
                self.weaviate_client.schema.create_class(product_schema)
                logger.info("Created Weaviate Product schema")
        except Exception as e:
            logger.warning(f"Failed to setup Weaviate schema: {e}")
    
    async def store_product_embedding(
        self,
        product_id: str,
        embedding: list[float],
        metadata: dict
    ) -> str:
        """Store product embedding in vector database"""
        try:
            if self.pinecone_client:
                return await self._store_pinecone_embedding(product_id, embedding, metadata)
            elif self.weaviate_client:
                return await self._store_weaviate_embedding(product_id, embedding, metadata)
            else:
                raise Exception("No vector database available")
        except Exception as e:
            logger.error(f"Failed to store embedding for product {product_id}: {e}")
            raise
    
    async def _store_pinecone_embedding(
        self,
        product_id: str,
        embedding: list[float],
        metadata: dict
    ) -> str:
        """Store embedding in Pinecone"""
        index = self.pinecone_client.Index(settings.pinecone_index_name)
        
        # Upsert the embedding
        index.upsert([
            {
                "id": product_id,
                "values": embedding,
                "metadata": metadata
            }
        ])
        
        return product_id
    
    async def _store_weaviate_embedding(
        self,
        product_id: str,
        embedding: list[float],
        metadata: dict
    ) -> str:
        """Store embedding in Weaviate"""
        # Prepare data object
        data_object = {
            "productId": product_id,
            "title": metadata.get("title", ""),
            "category": metadata.get("category", ""),
            "brand": metadata.get("brand", ""),
            "price": metadata.get("price", 0),
            "features": metadata.get("features", [])
        }
        
        # Add object with vector
        result = self.weaviate_client.data_object.create(
            data_object=data_object,
            class_name="Product",
            vector=embedding,
            uuid=product_id
        )
        
        return result
    
    async def search_similar_products(
        self,
        query_embedding: list[float],
        limit: int = 10,
        filters: dict = None
    ) -> list[dict]:
        """Search for similar products using vector similarity"""
        try:
            if self.pinecone_client:
                return await self._search_pinecone(query_embedding, limit, filters)
            elif self.weaviate_client:
                return await self._search_weaviate(query_embedding, limit, filters)
            else:
                raise Exception("No vector database available")
        except Exception as e:
            logger.error(f"Failed to search similar products: {e}")
            raise
    
    async def _search_pinecone(
        self,
        query_embedding: list[float],
        limit: int,
        filters: dict = None
    ) -> list[dict]:
        """Search similar products in Pinecone"""
        index = self.pinecone_client.Index(settings.pinecone_index_name)
        
        # Build filter if provided
        pinecone_filter = None
        if filters:
            pinecone_filter = {}
            for key, value in filters.items():
                if isinstance(value, list):
                    pinecone_filter[key] = {"$in": value}
                else:
                    pinecone_filter[key] = {"$eq": value}
        
        # Query similar vectors
        results = index.query(
            vector=query_embedding,
            top_k=limit,
            include_metadata=True,
            filter=pinecone_filter
        )
        
        # Format results
        return [
            {
                "product_id": match["id"],
                "similarity_score": match["score"],
                "metadata": match["metadata"]
            }
            for match in results["matches"]
        ]
    
    async def _search_weaviate(
        self,
        query_embedding: list[float],
        limit: int,
        filters: dict = None
    ) -> list[dict]:
        """Search similar products in Weaviate"""
        # Build where filter
        where_filter = None
        if filters:
            conditions = []
            for key, value in filters.items():
                if isinstance(value, list):
                    for v in value:
                        conditions.append({
                            "path": [key],
                            "operator": "Equal",
                            "valueString": str(v)
                        })
                else:
                    conditions.append({
                        "path": [key],
                        "operator": "Equal",
                        "valueString": str(value)
                    })
            
            if conditions:
                if len(conditions) == 1:
                    where_filter = conditions[0]
                else:
                    where_filter = {
                        "operator": "And",
                        "operands": conditions
                    }
        
        # Perform near vector search
        result = (
            self.weaviate_client.query
            .get("Product", ["productId", "title", "category", "brand", "price", "features"])
            .with_near_vector({"vector": query_embedding})
            .with_limit(limit)
            .with_additional(["distance"])
        )
        
        if where_filter:
            result = result.with_where(where_filter)
        
        response = result.do()
        
        # Format results
        products = []
        if "data" in response and "Get" in response["data"] and "Product" in response["data"]["Get"]:
            for item in response["data"]["Get"]["Product"]:
                similarity_score = 1 - item["_additional"]["distance"]  # Convert distance to similarity
                products.append({
                    "product_id": item["productId"],
                    "similarity_score": similarity_score,
                    "metadata": {
                        "title": item.get("title"),
                        "category": item.get("category"),
                        "brand": item.get("brand"),
                        "price": item.get("price"),
                        "features": item.get("features", [])
                    }
                })
        
        return products
    
    def get_client(self):
        """Get available vector database client"""
        if self.pinecone_client:
            return self.pinecone_client
        elif self.weaviate_client:
            return self.weaviate_client
        else:
            raise Exception("No vector database available")


# Global instances - Initialize lazily to avoid startup errors in demo
redis_client = None
vector_db = None

def init_redis():
    """Initialize Redis client lazily"""
    global redis_client
    try:
        redis_client = get_redis_client()
    except Exception as e:
        logger.warning(f"Redis not available for demo: {e}")
        # Use mock Redis for demo
        class MockRedis:
            def ping(self): return "PONG"
            def get(self, key): return None
            def set(self, key, value, ex=None): return True
            def delete(self, key): return True
            def lpush(self, key, value): return True
            def lrange(self, key, start, end): return []
            def ltrim(self, key, start, end): return True
            def setex(self, key, time, value): return True
        redis_client = MockRedis()

def init_vector_db():
    """Initialize vector database lazily"""
    global vector_db
    try:
        vector_db = VectorDB()
    except Exception as e:
        logger.warning(f"Vector database not available for demo: {e}")
        vector_db = None


def init_database():
    """Initialize database tables"""
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
        
        # Initialize Redis and Vector DB
        init_redis()
        init_vector_db()
        
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise