from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import uuid
from loguru import logger

from src.services.product_analyzer import ProductAnalyzerService
from src.services.description_generator import DescriptionGeneratorService, GenerationRequest
from src.services.seo_optimizer import SEOOptimizerService
from src.services.ab_testing_service import ABTestingService, ABTestConfig
from src.models.product_description import DescriptionType, GenerationMethod, DescriptionStatus
from src.core.auth import get_current_user
from src.models.user import User
from src.core.database import get_db


# Pydantic models for API
class ProductAnalysisRequest(BaseModel):
    product_id: str

class ProductAnalysisResponse(BaseModel):
    product_id: str
    analysis_date: str
    specifications: Dict[str, Any]
    market_analysis: Dict[str, Any]
    seo_analysis: Dict[str, Any]
    content_recommendations: Dict[str, Any]
    quality_scores: Dict[str, float]

class DescriptionGenerationRequest(BaseModel):
    product_id: str
    description_types: List[str] = Field(..., description="List of description types to generate")
    target_keywords: Optional[List[str]] = Field(None, description="Target SEO keywords")
    target_word_count: Optional[int] = Field(None, description="Target word count")
    tone: str = Field("professional", description="Content tone")
    brand_guidelines_id: Optional[str] = Field(None, description="Brand guidelines ID")
    template_id: Optional[str] = Field(None, description="Custom template ID")
    custom_prompt: Optional[str] = Field(None, description="Custom generation prompt")
    include_seo: bool = Field(True, description="Include SEO optimization")
    include_specifications: bool = Field(True, description="Include product specifications")
    target_audience: Optional[str] = Field(None, description="Target audience")

class DescriptionGenerationResponse(BaseModel):
    product_id: str
    generation_id: str
    descriptions: Dict[str, Any]
    metadata: Dict[str, Any]
    recommendations: Dict[str, Any]
    generated_at: str

class SEOAnalysisRequest(BaseModel):
    description_id: str
    target_keywords: Optional[List[str]] = None

class SEOAnalysisResponse(BaseModel):
    seo_score: float
    keyword_density: Dict[str, float]
    title_optimization: Dict[str, Any]
    meta_description_optimization: Dict[str, Any]
    content_optimization: Dict[str, Any]
    recommendations: List[str]
    technical_issues: List[str]
    opportunities: List[str]

class SEOOptimizationRequest(BaseModel):
    content: str
    target_keywords: List[str]
    optimization_level: str = Field("moderate", description="light, moderate, or aggressive")

class SEOOptimizationResponse(BaseModel):
    original_content: str
    optimized_content: str
    improvements: Dict[str, Any]
    recommendations: List[str]

class ABTestRequest(BaseModel):
    test_name: str
    product_id: str
    variants: List[Dict[str, Any]]
    traffic_allocation: Dict[str, float]
    success_metrics: List[str]
    duration_days: int
    confidence_threshold: float = 0.95
    auto_start: bool = False

class ABTestResponse(BaseModel):
    test_id: str
    status: str
    message: str

class BulkGenerationRequest(BaseModel):
    product_ids: List[str]
    description_types: List[str]
    generation_settings: Dict[str, Any]

class BulkGenerationResponse(BaseModel):
    job_id: str
    total_products: int
    estimated_time_minutes: int
    status: str


# Router setup
router = APIRouter(prefix="/product-descriptions", tags=["product-descriptions"])

# Initialize services
product_analyzer = ProductAnalyzerService()
description_generator = DescriptionGeneratorService()
seo_optimizer = SEOOptimizerService()
ab_testing = ABTestingService()


@router.post("/analyze", response_model=ProductAnalysisResponse)
async def analyze_product_specifications(
    request: ProductAnalysisRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Analyze product specifications and market positioning
    """
    try:
        logger.info(f"Analyzing product {request.product_id} for user {current_user.email}")
        
        # Generate comprehensive analysis report
        analysis_report = await product_analyzer.generate_product_analysis_report(
            request.product_id
        )
        
        return ProductAnalysisResponse(
            product_id=analysis_report["product_id"],
            analysis_date=analysis_report["analysis_date"],
            specifications=analysis_report["specifications"],
            market_analysis=analysis_report["market_analysis"],
            seo_analysis=analysis_report["seo_analysis"],
            content_recommendations=analysis_report["content_recommendations"],
            quality_scores=analysis_report["quality_scores"]
        )
        
    except Exception as e:
        logger.error(f"Product analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate", response_model=DescriptionGenerationResponse)
async def generate_descriptions(
    request: DescriptionGenerationRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user)
):
    """
    Generate product descriptions in multiple formats
    """
    try:
        logger.info(f"Generating descriptions for product {request.product_id}")
        
        # Convert string types to enum
        description_types = []
        for desc_type_str in request.description_types:
            try:
                description_types.append(DescriptionType(desc_type_str))
            except ValueError:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Invalid description type: {desc_type_str}"
                )
        
        # Create generation request
        generation_request = GenerationRequest(
            product_id=request.product_id,
            description_types=description_types,
            target_keywords=request.target_keywords,
            target_word_count=request.target_word_count,
            tone=request.tone,
            brand_guidelines_id=request.brand_guidelines_id,
            template_id=request.template_id,
            custom_prompt=request.custom_prompt,
            include_seo=request.include_seo,
            include_specifications=request.include_specifications,
            target_audience=request.target_audience
        )
        
        # Generate descriptions
        result = await description_generator.generate_descriptions(generation_request)
        
        # Store results in database (background task)
        background_tasks.add_task(
            _store_generated_descriptions,
            result,
            current_user.id
        )
        
        return DescriptionGenerationResponse(
            product_id=result["product_id"],
            generation_id=result["generation_id"],
            descriptions=result["descriptions"],
            metadata=result["metadata"],
            recommendations=result["recommendations"],
            generated_at=result["generated_at"]
        )
        
    except Exception as e:
        logger.error(f"Description generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def _store_generated_descriptions(
    result: Dict[str, Any],
    user_id: str
):
    """
    Store generated descriptions in database (background task)
    """
    try:
        from src.models.product_description import ProductDescription, ContentGeneration
        
        db = next(get_db())
        
        # Store each generated description
        for desc_type, desc_data in result["descriptions"].items():
            # Create ProductDescription record
            description = ProductDescription(
                product_id=result["product_id"],
                description_type=DescriptionType(desc_type),
                title=f"{desc_type.title()} Description",
                content=desc_data["content"],
                generation_method=GenerationMethod(desc_data["generation_method"]),
                quality_score=desc_data.get("quality_score", 0.0),
                seo_score=desc_data.get("seo_metadata", {}).get("score", 0.0),
                created_by=user_id,
                status=DescriptionStatus.DRAFT
            )
            
            # Add SEO metadata if available
            seo_metadata = desc_data.get("seo_metadata", {})
            if seo_metadata:
                description.meta_title = seo_metadata.get("meta_title")
                description.meta_description = seo_metadata.get("meta_description")
                description.keywords = seo_metadata.get("keywords", [])
            
            db.add(description)
        
        # Store generation record
        generation_record = ContentGeneration(
            product_id=result["product_id"],
            description_type=DescriptionType.MEDIUM,  # Default
            generation_method=GenerationMethod.HYBRID,
            generated_content=str(result["descriptions"]),
            quality_score=result["metadata"]["generation_stats"]["average_quality_score"],
            generation_time_ms=result["metadata"]["generation_stats"]["total_time_ms"],
            requested_by=user_id,
            status="completed"
        )
        
        db.add(generation_record)
        db.commit()
        db.close()
        
        logger.info(f"Stored generated descriptions for product {result['product_id']}")
        
    except Exception as e:
        logger.error(f"Failed to store generated descriptions: {e}")


@router.post("/seo/analyze", response_model=SEOAnalysisResponse)
async def analyze_seo_performance(
    request: SEOAnalysisRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Analyze SEO performance of a product description
    """
    try:
        logger.info(f"Analyzing SEO for description {request.description_id}")
        
        # Perform SEO analysis
        analysis = await seo_optimizer.analyze_seo_performance(
            description_id=request.description_id,
            target_keywords=request.target_keywords
        )
        
        return SEOAnalysisResponse(
            seo_score=analysis.seo_score,
            keyword_density=analysis.keyword_density,
            title_optimization=analysis.title_optimization,
            meta_description_optimization=analysis.meta_description_optimization,
            content_optimization=analysis.content_optimization,
            recommendations=analysis.recommendations,
            technical_issues=analysis.technical_issues,
            opportunities=analysis.opportunities
        )
        
    except Exception as e:
        logger.error(f"SEO analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/seo/optimize", response_model=SEOOptimizationResponse)
async def optimize_content_seo(
    request: SEOOptimizationRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Optimize content for SEO using AI
    """
    try:
        logger.info(f"Optimizing content for SEO with {len(request.target_keywords)} keywords")
        
        # Optimize content
        result = await seo_optimizer.optimize_content_for_seo(
            content=request.content,
            target_keywords=request.target_keywords,
            optimization_level=request.optimization_level
        )
        
        return SEOOptimizationResponse(
            original_content=result["original_content"],
            optimized_content=result["optimized_content"],
            improvements=result["improvements"],
            recommendations=result["recommendations"]
        )
        
    except Exception as e:
        logger.error(f"SEO optimization failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ab-test/create", response_model=ABTestResponse)
async def create_ab_test(
    request: ABTestRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Create A/B test for product descriptions
    """
    try:
        logger.info(f"Creating A/B test: {request.test_name}")
        
        # Create test configuration
        test_config = ABTestConfig(
            test_name=request.test_name,
            product_id=request.product_id,
            variants=request.variants,
            traffic_allocation=request.traffic_allocation,
            success_metrics=request.success_metrics,
            duration_days=request.duration_days,
            confidence_threshold=request.confidence_threshold
        )
        
        # Create test
        test_id = await ab_testing.create_ab_test(
            config=test_config,
            auto_start=request.auto_start
        )
        
        return ABTestResponse(
            test_id=test_id,
            status="running" if request.auto_start else "draft",
            message=f"A/B test {'started' if request.auto_start else 'created'} successfully"
        )
        
    except Exception as e:
        logger.error(f"A/B test creation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ab-test/{test_id}/start")
async def start_ab_test(
    test_id: str,
    current_user: User = Depends(get_current_user)
):
    """
    Start an A/B test
    """
    try:
        success = await ab_testing.start_test(test_id)
        
        if success:
            return {"message": "Test started successfully", "test_id": test_id}
        else:
            raise HTTPException(status_code=404, detail="Test not found or already running")
            
    except Exception as e:
        logger.error(f"Failed to start A/B test: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ab-test/{test_id}/stop")
async def stop_ab_test(
    test_id: str,
    current_user: User = Depends(get_current_user)
):
    """
    Stop an A/B test
    """
    try:
        success = await ab_testing.stop_test(test_id, "manual_stop")
        
        if success:
            return {"message": "Test stopped successfully", "test_id": test_id}
        else:
            raise HTTPException(status_code=404, detail="Test not found or not running")
            
    except Exception as e:
        logger.error(f"Failed to stop A/B test: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/ab-test/{test_id}/results")
async def get_test_results(
    test_id: str,
    current_user: User = Depends(get_current_user)
):
    """
    Get A/B test results and analysis
    """
    try:
        results = await ab_testing.get_test_results(test_id)
        
        if results:
            return {
                "test_id": results.test_id,
                "status": results.status,
                "duration_days": results.duration_days,
                "total_participants": results.total_participants,
                "confidence_level": results.confidence_level,
                "winner": results.winner,
                "statistical_significance": results.statistical_significance,
                "effect_size": results.effect_size,
                "conversion_rates": results.conversion_rates,
                "sample_sizes": results.sample_sizes,
                "recommendations": results.recommendations
            }
        else:
            raise HTTPException(status_code=404, detail="Test results not found")
            
    except Exception as e:
        logger.error(f"Failed to get test results: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/ab-test/active")
async def get_active_tests(
    current_user: User = Depends(get_current_user)
):
    """
    Get list of active A/B tests
    """
    try:
        active_tests = await ab_testing.get_active_tests()
        return {"active_tests": active_tests}
        
    except Exception as e:
        logger.error(f"Failed to get active tests: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/bulk-generate", response_model=BulkGenerationResponse)
async def bulk_generate_descriptions(
    request: BulkGenerationRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user)
):
    """
    Generate descriptions for multiple products in bulk
    """
    try:
        logger.info(f"Starting bulk generation for {len(request.product_ids)} products")
        
        # Validate product count
        if len(request.product_ids) > 100:
            raise HTTPException(
                status_code=400, 
                detail="Maximum 100 products per bulk generation"
            )
        
        # Generate job ID
        job_id = f"bulk_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        
        # Estimate time (rough calculation)
        estimated_time_per_product = 30  # seconds
        estimated_time_minutes = (len(request.product_ids) * estimated_time_per_product) // 60
        
        # Start bulk generation as background task
        background_tasks.add_task(
            _process_bulk_generation,
            job_id,
            request.product_ids,
            request.description_types,
            request.generation_settings,
            current_user.id
        )
        
        return BulkGenerationResponse(
            job_id=job_id,
            total_products=len(request.product_ids),
            estimated_time_minutes=estimated_time_minutes,
            status="processing"
        )
        
    except Exception as e:
        logger.error(f"Bulk generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def _process_bulk_generation(
    job_id: str,
    product_ids: List[str],
    description_types: List[str],
    generation_settings: Dict[str, Any],
    user_id: str
):
    """
    Process bulk generation in background
    """
    try:
        logger.info(f"Processing bulk generation job {job_id}")
        
        # Convert description types
        desc_types = []
        for desc_type_str in description_types:
            try:
                desc_types.append(DescriptionType(desc_type_str))
            except ValueError:
                logger.warning(f"Invalid description type: {desc_type_str}")
                continue
        
        if not desc_types:
            logger.error("No valid description types provided")
            return
        
        # Process each product
        successful = 0
        failed = 0
        
        for product_id in product_ids:
            try:
                # Create generation request
                generation_request = GenerationRequest(
                    product_id=product_id,
                    description_types=desc_types,
                    target_keywords=generation_settings.get("target_keywords"),
                    target_word_count=generation_settings.get("target_word_count"),
                    tone=generation_settings.get("tone", "professional"),
                    brand_guidelines_id=generation_settings.get("brand_guidelines_id"),
                    include_seo=generation_settings.get("include_seo", True),
                    target_audience=generation_settings.get("target_audience")
                )
                
                # Generate descriptions
                result = await description_generator.generate_descriptions(generation_request)
                
                # Store results
                await _store_generated_descriptions(result, user_id)
                
                successful += 1
                logger.debug(f"Generated descriptions for product {product_id}")
                
            except Exception as e:
                logger.error(f"Failed to generate descriptions for product {product_id}: {e}")
                failed += 1
        
        logger.info(f"Bulk generation job {job_id} completed: {successful} successful, {failed} failed")
        
    except Exception as e:
        logger.error(f"Bulk generation job {job_id} failed: {e}")


@router.get("/bulk-generate/{job_id}/status")
async def get_bulk_generation_status(
    job_id: str,
    current_user: User = Depends(get_current_user)
):
    """
    Get status of bulk generation job
    """
    # This would typically check a job queue or database
    # For now, return a simple response
    return {
        "job_id": job_id,
        "status": "completed",  # Would be dynamic in real implementation
        "progress": {
            "completed": 100,
            "total": 100,
            "failed": 0
        },
        "estimated_time_remaining": 0
    }


@router.get("/templates")
async def get_description_templates(
    category: Optional[str] = Query(None),
    description_type: Optional[str] = Query(None),
    current_user: User = Depends(get_current_user)
):
    """
    Get available description templates
    """
    try:
        from src.models.product_description import DescriptionTemplate
        
        db = next(get_db())
        
        query = db.query(DescriptionTemplate).filter(
            DescriptionTemplate.is_active == True
        )
        
        if category:
            query = query.filter(DescriptionTemplate.category == category)
        
        if description_type:
            try:
                desc_type = DescriptionType(description_type)
                query = query.filter(DescriptionTemplate.description_type == desc_type)
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid description type")
        
        templates = query.all()
        
        template_list = []
        for template in templates:
            template_list.append({
                "id": str(template.id),
                "name": template.name,
                "description_type": template.description_type.value,
                "category": template.category,
                "tone": template.tone,
                "target_audience": template.target_audience,
                "average_quality_score": template.average_quality_score,
                "usage_count": template.usage_count
            })
        
        db.close()
        
        return {"templates": template_list}
        
    except Exception as e:
        logger.error(f"Failed to get templates: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analytics")
async def get_description_analytics(
    product_id: Optional[str] = Query(None),
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    current_user: User = Depends(get_current_user)
):
    """
    Get analytics for product descriptions
    """
    try:
        from src.models.product_description import ProductDescription, DescriptionAnalytics
        
        db = next(get_db())
        
        # Build query
        query = db.query(ProductDescription)
        
        if product_id:
            query = query.filter(ProductDescription.product_id == product_id)
        
        if start_date and end_date:
            query = query.filter(
                ProductDescription.created_at >= start_date,
                ProductDescription.created_at <= end_date
            )
        
        descriptions = query.all()
        
        # Calculate analytics
        total_descriptions = len(descriptions)
        avg_quality_score = sum(d.quality_score or 0.0 for d in descriptions) / max(total_descriptions, 1)
        avg_seo_score = sum(d.seo_score or 0.0 for d in descriptions) / max(total_descriptions, 1)
        
        # Description type distribution
        type_distribution = {}
        for desc in descriptions:
            desc_type = desc.description_type.value
            type_distribution[desc_type] = type_distribution.get(desc_type, 0) + 1
        
        # Generation method distribution
        method_distribution = {}
        for desc in descriptions:
            method = desc.generation_method.value
            method_distribution[method] = method_distribution.get(method, 0) + 1
        
        # Get A/B test analytics
        ab_analytics = await ab_testing.get_test_analytics(
            product_id=product_id,
            date_range=(start_date, end_date) if start_date and end_date else None
        )
        
        db.close()
        
        return {
            "description_analytics": {
                "total_descriptions": total_descriptions,
                "average_quality_score": avg_quality_score,
                "average_seo_score": avg_seo_score,
                "type_distribution": type_distribution,
                "method_distribution": method_distribution
            },
            "ab_test_analytics": ab_analytics,
            "analysis_period": {
                "start_date": start_date.isoformat() if start_date else None,
                "end_date": end_date.isoformat() if end_date else None
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def description_service_health():
    """
    Health check endpoint for description services
    """
    try:
        # Check service availability
        health_status = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "services": {
                "product_analyzer": True,
                "description_generator": True,
                "seo_optimizer": True,
                "ab_testing": True
            },
            "version": "1.0.0"
        }
        
        return health_status
        
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }