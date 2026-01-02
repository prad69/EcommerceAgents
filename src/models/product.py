from sqlalchemy import Column, Integer, String, Boolean, DateTime, Text, JSON, Float, ForeignKey
from sqlalchemy.sql import func
from sqlalchemy.dialects.postgresql import UUID, ARRAY
from sqlalchemy.orm import relationship
import uuid

from src.core.database import Base


class Category(Base):
    __tablename__ = "categories"
    __table_args__ = {'schema': 'products'}
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String, unique=True, index=True, nullable=False)
    description = Column(Text, nullable=True)
    parent_id = Column(UUID(as_uuid=True), ForeignKey("products.categories.id"), nullable=True)
    
    # Hierarchy and metadata
    level = Column(Integer, default=0)
    path = Column(String, nullable=True)  # e.g., "Electronics > Smartphones > Android"
    is_active = Column(Boolean, default=True)
    
    # SEO and display
    slug = Column(String, unique=True, index=True, nullable=False)
    image_url = Column(String, nullable=True)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    children = relationship("Category", backref="parent", remote_side=[id])
    
    def __repr__(self):
        return f"<Category(name='{self.name}', level={self.level})>"


class Brand(Base):
    __tablename__ = "brands"
    __table_args__ = {'schema': 'products'}
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String, unique=True, index=True, nullable=False)
    description = Column(Text, nullable=True)
    logo_url = Column(String, nullable=True)
    website_url = Column(String, nullable=True)
    is_active = Column(Boolean, default=True)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    def __repr__(self):
        return f"<Brand(name='{self.name}')>"


class Product(Base):
    __tablename__ = "products"
    __table_args__ = {'schema': 'products'}
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Basic product information
    title = Column(String, nullable=False, index=True)
    description = Column(Text, nullable=True)
    short_description = Column(Text, nullable=True)
    sku = Column(String, unique=True, index=True, nullable=False)
    
    # Categorization
    category_id = Column(UUID(as_uuid=True), ForeignKey("products.categories.id"), nullable=True)
    brand_id = Column(UUID(as_uuid=True), ForeignKey("products.brands.id"), nullable=True)
    
    # Pricing
    price = Column(Float, nullable=False)
    original_price = Column(Float, nullable=True)
    currency = Column(String, default="USD")
    
    # Inventory
    stock_quantity = Column(Integer, default=0)
    is_in_stock = Column(Boolean, default=True)
    
    # Product attributes
    attributes = Column(JSON, nullable=True, default={})  # Flexible attributes
    specifications = Column(JSON, nullable=True, default={})
    features = Column(ARRAY(String), nullable=True)
    tags = Column(ARRAY(String), nullable=True)
    
    # Media
    images = Column(JSON, nullable=True, default=[])  # Array of image URLs
    primary_image = Column(String, nullable=True)
    
    # SEO and display
    slug = Column(String, unique=True, index=True, nullable=False)
    meta_title = Column(String, nullable=True)
    meta_description = Column(Text, nullable=True)
    
    # Status and visibility
    is_active = Column(Boolean, default=True)
    is_featured = Column(Boolean, default=False)
    is_digital = Column(Boolean, default=False)
    
    # Ratings and reviews (denormalized for performance)
    average_rating = Column(Float, default=0.0)
    review_count = Column(Integer, default=0)
    
    # Data sources and sync
    external_id = Column(String, nullable=True, index=True)  # ID from external source
    data_source = Column(String, nullable=True)  # amazon, bestbuy, etc.
    last_synced = Column(DateTime(timezone=True), nullable=True)
    
    # Vector embeddings (for RAG)
    embedding_id = Column(String, nullable=True)  # Reference to vector DB
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    category = relationship("Category", back_populates=None)
    brand = relationship("Brand", back_populates=None)
    
    def __repr__(self):
        return f"<Product(title='{self.title}', sku='{self.sku}', price={self.price})>"


class ProductVariant(Base):
    __tablename__ = "product_variants"
    __table_args__ = {'schema': 'products'}
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    product_id = Column(UUID(as_uuid=True), ForeignKey("products.products.id"), nullable=False)
    
    # Variant details
    title = Column(String, nullable=False)
    sku = Column(String, unique=True, index=True, nullable=False)
    
    # Variant attributes (size, color, etc.)
    attributes = Column(JSON, nullable=False, default={})
    
    # Pricing and inventory
    price = Column(Float, nullable=False)
    stock_quantity = Column(Integer, default=0)
    is_in_stock = Column(Boolean, default=True)
    
    # Media
    image_url = Column(String, nullable=True)
    
    # Status
    is_active = Column(Boolean, default=True)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    product = relationship("Product", back_populates=None)
    
    def __repr__(self):
        return f"<ProductVariant(product_id='{self.product_id}', sku='{self.sku}')>"