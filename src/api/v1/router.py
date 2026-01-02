from fastapi import APIRouter

from .endpoints import auth, products, recommendations, analytics, agents

api_router = APIRouter()

# Include all endpoint routers
api_router.include_router(auth.router, prefix="/auth", tags=["Authentication"])\napi_router.include_router(products.router, prefix="/products", tags=["Products"])\napi_router.include_router(recommendations.router, prefix="/recommendations", tags=["Recommendations"])\napi_router.include_router(analytics.router, prefix="/analytics", tags=["Analytics"])\napi_router.include_router(agents.router, prefix="/agents", tags=["AI Agents"])