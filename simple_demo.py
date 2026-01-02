#!/usr/bin/env python3
"""
EcommerceAgents Simple Demo
A lightweight demo showing the core API structure and mock responses
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import asyncio
import uvicorn
from datetime import datetime
import json

# Demo data models
class ProductRecommendation(BaseModel):
    product_id: str
    title: str
    price: float
    category: str
    similarity_score: float
    features: List[str]

class ChatMessage(BaseModel):
    message: str
    session_id: str
    user_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    intent: str
    entities: Dict[str, Any]
    session_id: str
    timestamp: str

class ProductDescriptionRequest(BaseModel):
    product_name: str
    features: List[str]
    target_audience: Optional[str] = "general"
    tone: Optional[str] = "professional"

class ReviewAnalysisRequest(BaseModel):
    product_id: str
    review_text: str

# Mock data
MOCK_PRODUCTS = [
    {
        "product_id": "laptop_001",
        "title": "MacBook Pro 16-inch",
        "price": 2399.00,
        "category": "Laptops",
        "features": ["M3 Pro chip", "16GB RAM", "512GB SSD", "Liquid Retina XDR display"],
        "rating": 4.8,
        "reviews_count": 1234
    },
    {
        "product_id": "laptop_002", 
        "title": "Dell XPS 13",
        "price": 1199.00,
        "category": "Laptops",
        "features": ["Intel i7", "16GB RAM", "256GB SSD", "13.3-inch display"],
        "rating": 4.6,
        "reviews_count": 856
    },
    {
        "product_id": "phone_001",
        "title": "iPhone 15 Pro",
        "price": 999.00,
        "category": "Smartphones",
        "features": ["A17 Pro chip", "48MP camera", "6.1-inch display", "Titanium design"],
        "rating": 4.7,
        "reviews_count": 2105
    },
    {
        "product_id": "headphones_001",
        "title": "Sony WH-1000XM5",
        "price": 399.99,
        "category": "Headphones",
        "features": ["Noise cancelling", "30-hour battery", "Bluetooth 5.2", "Touch controls"],
        "rating": 4.5,
        "reviews_count": 743
    }
]

# FastAPI app
app = FastAPI(
    title="EcommerceAgents Demo API",
    description="Multi-Agent E-Commerce Intelligence System Demo",
    version="1.0.0"
)

# Demo homepage
@app.get("/", response_class=HTMLResponse)
async def homepage():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>EcommerceAgents Demo</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }
            .container { max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            .header { text-align: center; margin-bottom: 30px; }
            .title { color: #2c3e50; margin-bottom: 10px; }
            .subtitle { color: #7f8c8d; margin-bottom: 30px; }
            .features { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin: 30px 0; }
            .feature { background: #f8f9fa; padding: 20px; border-radius: 8px; border-left: 4px solid #3498db; }
            .feature h3 { color: #2c3e50; margin-top: 0; }
            .endpoints { background: #2c3e50; color: white; padding: 20px; border-radius: 8px; margin: 20px 0; }
            .endpoint { margin: 10px 0; font-family: monospace; }
            .demo-links { display: flex; flex-wrap: wrap; gap: 15px; justify-content: center; margin-top: 30px; }
            .demo-link { background: #3498db; color: white; padding: 12px 24px; text-decoration: none; border-radius: 6px; transition: background-color 0.3s; }
            .demo-link:hover { background: #2980b9; }
            .status { text-align: center; margin: 20px 0; }
            .status.online { color: #27ae60; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1 class="title">🤖 EcommerceAgents Demo</h1>
                <p class="subtitle">Multi-Agent E-Commerce Intelligence System</p>
                <div class="status online">🟢 System Online - """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</div>
            </div>
            
            <div class="features">
                <div class="feature">
                    <h3>🎯 Product Recommendations</h3>
                    <p>AI-powered product recommendations using collaborative filtering and content-based algorithms.</p>
                </div>
                <div class="feature">
                    <h3>🤖 Intelligent Chatbot</h3>
                    <p>Natural language processing for customer support and product discovery.</p>
                </div>
                <div class="feature">
                    <h3>📝 Content Generation</h3>
                    <p>Automated product descriptions with SEO optimization and A/B testing.</p>
                </div>
                <div class="feature">
                    <h3>📊 Review Analysis</h3>
                    <p>Sentiment analysis and insights extraction from customer reviews.</p>
                </div>
                <div class="feature">
                    <h3>⚡ Auto-Scaling</h3>
                    <p>Intelligent resource scaling based on real-time performance metrics.</p>
                </div>
                <div class="feature">
                    <h3>📈 Analytics Dashboard</h3>
                    <p>Comprehensive business intelligence and performance monitoring.</p>
                </div>
            </div>
            
            <div class="endpoints">
                <h3>🔗 Available API Endpoints</h3>
                <div class="endpoint">GET /api/v1/recommendations?user_id=demo&limit=5</div>
                <div class="endpoint">POST /api/v1/chatbot/message</div>
                <div class="endpoint">POST /api/v1/product-description/generate</div>
                <div class="endpoint">POST /api/v1/reviews/analyze</div>
                <div class="endpoint">GET /api/v1/analytics/dashboard</div>
                <div class="endpoint">GET /health - System health check</div>
                <div class="endpoint">GET /metrics - Prometheus metrics</div>
            </div>
            
            <div class="demo-links">
                <a href="/docs" class="demo-link">📚 API Documentation</a>
                <a href="/health" class="demo-link">🏥 Health Check</a>
                <a href="/api/v1/recommendations?user_id=demo&limit=3" class="demo-link">🎯 Try Recommendations</a>
                <a href="/metrics" class="demo-link">📊 View Metrics</a>
            </div>
            
            <div style="margin-top: 40px; text-align: center; color: #7f8c8d; font-size: 14px;">
                <p>EcommerceAgents v1.0.0 - Multi-Agent E-Commerce Intelligence System</p>
                <p>Phases 4, 5, and 6 Implementation Complete ✅</p>
            </div>
        </div>
    </body>
    </html>
    """

@app.get("/health")
async def health_check():
    """System health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "environment": "demo",
        "components": {
            "api": "online",
            "database": "mock",
            "cache": "mock",
            "ai_agents": "ready"
        }
    }

@app.get("/metrics")
async def get_metrics():
    """Prometheus-style metrics"""
    metrics = """# HELP ecommerce_requests_total Total number of requests
# TYPE ecommerce_requests_total counter
ecommerce_requests_total{endpoint="/api/v1/recommendations"} 150
ecommerce_requests_total{endpoint="/api/v1/chatbot/message"} 89
ecommerce_requests_total{endpoint="/api/v1/product-description/generate"} 34

# HELP ecommerce_response_time_seconds Response time in seconds
# TYPE ecommerce_response_time_seconds histogram
ecommerce_response_time_seconds_bucket{le="0.1"} 120
ecommerce_response_time_seconds_bucket{le="0.5"} 145
ecommerce_response_time_seconds_bucket{le="1.0"} 150
ecommerce_response_time_seconds_bucket{le="+Inf"} 150

# HELP ecommerce_agent_effectiveness Agent effectiveness percentage
# TYPE ecommerce_agent_effectiveness gauge
ecommerce_agent_effectiveness{agent="recommendations"} 0.87
ecommerce_agent_effectiveness{agent="chatbot"} 0.82
ecommerce_agent_effectiveness{agent="reviews"} 0.79
ecommerce_agent_effectiveness{agent="descriptions"} 0.91

# HELP ecommerce_business_metrics Business performance metrics
# TYPE ecommerce_business_metrics gauge
ecommerce_business_metrics{metric="conversion_rate"} 0.028
ecommerce_business_metrics{metric="revenue_per_visitor"} 45.67
ecommerce_business_metrics{metric="customer_satisfaction"} 4.2
"""
    return JSONResponse(content=metrics, media_type="text/plain")

# API Routes
@app.get("/api/v1/recommendations")
async def get_recommendations(user_id: str = "demo", limit: int = 5):
    """Get product recommendations for a user"""
    
    # Simulate AI recommendation logic
    import random
    recommended_products = random.sample(MOCK_PRODUCTS, min(limit, len(MOCK_PRODUCTS)))
    
    recommendations = []
    for i, product in enumerate(recommended_products):
        rec = ProductRecommendation(
            product_id=product["product_id"],
            title=product["title"],
            price=product["price"],
            category=product["category"],
            similarity_score=round(random.uniform(0.75, 0.95), 3),
            features=product["features"]
        )
        recommendations.append(rec)
    
    return {
        "user_id": user_id,
        "recommendations": recommendations,
        "algorithm": "collaborative_filtering_v2",
        "timestamp": datetime.now().isoformat(),
        "total_found": len(recommendations)
    }

@app.post("/api/v1/chatbot/message")
async def chat_message(message: ChatMessage):
    """Process chatbot message"""
    
    # Simple intent detection
    msg_lower = message.message.lower()
    
    if any(word in msg_lower for word in ["laptop", "computer", "macbook"]):
        intent = "product_search_laptop"
        response = "I can help you find the perfect laptop! Based on your needs, I'd recommend checking out our MacBook Pro 16-inch or Dell XPS 13. What's your budget range?"
        entities = {"product_type": "laptop", "category": "electronics"}
    elif any(word in msg_lower for word in ["phone", "smartphone", "iphone"]):
        intent = "product_search_phone"
        response = "Great choice! Our iPhone 15 Pro is very popular. It features the A17 Pro chip and amazing camera quality. Would you like to see similar options?"
        entities = {"product_type": "smartphone", "category": "electronics"}
    elif any(word in msg_lower for word in ["help", "support", "problem"]):
        intent = "customer_support"
        response = "I'm here to help! What specific issue can I assist you with today? I can help with product recommendations, order status, or general questions."
        entities = {"intent_type": "support", "urgency": "normal"}
    elif any(word in msg_lower for word in ["price", "cost", "expensive", "cheap"]):
        intent = "price_inquiry"
        response = "I can help you find products in your price range! What's your budget, and what type of product are you looking for?"
        entities = {"intent_type": "pricing", "category": "budget_inquiry"}
    else:
        intent = "general_inquiry"
        response = "Hello! I'm your AI shopping assistant. I can help you find products, answer questions about our inventory, or provide recommendations. How can I assist you today?"
        entities = {"intent_type": "general"}
    
    return ChatResponse(
        response=response,
        intent=intent,
        entities=entities,
        session_id=message.session_id,
        timestamp=datetime.now().isoformat()
    )

@app.post("/api/v1/product-description/generate")
async def generate_product_description(request: ProductDescriptionRequest):
    """Generate AI product description"""
    
    # Simulate AI content generation
    features_text = ", ".join(request.features)
    
    if request.tone == "engaging":
        tone_style = "exciting and dynamic"
    elif request.tone == "professional":
        tone_style = "professional and informative"
    else:
        tone_style = "friendly and approachable"
    
    descriptions = {
        "primary": f"Discover the {request.product_name} - where innovation meets excellence! Featuring {features_text}, this premium device delivers unmatched performance for {request.target_audience}. Experience the perfect blend of cutting-edge technology and sleek design.",
        
        "short": f"Premium {request.product_name} with {features_text}. Perfect for {request.target_audience}.",
        
        "detailed": f"""The {request.product_name} represents the pinnacle of modern technology and design. Engineered with {features_text}, this exceptional device caters specifically to the needs of {request.target_audience}.

Key highlights include:
{chr(10).join(f"• {feature}" for feature in request.features)}

Whether you're seeking performance, reliability, or style, the {request.product_name} delivers on all fronts. Its innovative design and powerful capabilities make it the ideal choice for those who demand excellence.

Backed by our commitment to quality and customer satisfaction, this {request.product_name} comes with comprehensive support and warranty coverage."""
    }
    
    seo_data = {
        "title": f"{request.product_name} - Premium Quality for {request.target_audience.title()}",
        "meta_description": f"Shop the best {request.product_name} featuring {', '.join(request.features[:3])}. Perfect for {request.target_audience}. Free shipping available.",
        "keywords": [request.product_name.lower(), request.target_audience] + [f.lower() for f in request.features],
        "alt_text": f"{request.product_name} with premium features"
    }
    
    return {
        "product_name": request.product_name,
        "descriptions": descriptions,
        "seo_optimization": seo_data,
        "tone": request.tone,
        "target_audience": request.target_audience,
        "generated_at": datetime.now().isoformat(),
        "confidence_score": 0.94,
        "variations": [
            "Premium version with enhanced features",
            "Value-focused description for budget-conscious customers",
            "Technical specification emphasis for tech enthusiasts"
        ]
    }

@app.post("/api/v1/reviews/analyze")
async def analyze_review(request: ReviewAnalysisRequest):
    """Analyze product review sentiment and extract insights"""
    
    review_text = request.review_text.lower()
    
    # Simple sentiment analysis
    positive_words = ["great", "excellent", "amazing", "fantastic", "love", "perfect", "good", "awesome", "wonderful"]
    negative_words = ["bad", "terrible", "awful", "hate", "horrible", "worst", "disappointing", "poor", "useless"]
    
    positive_count = sum(1 for word in positive_words if word in review_text)
    negative_count = sum(1 for word in negative_words if word in review_text)
    
    if positive_count > negative_count:
        sentiment = "positive"
        confidence = min(0.9, 0.6 + (positive_count - negative_count) * 0.1)
    elif negative_count > positive_count:
        sentiment = "negative"
        confidence = min(0.9, 0.6 + (negative_count - positive_count) * 0.1)
    else:
        sentiment = "neutral"
        confidence = 0.7
    
    # Extract key topics
    topics = []
    if any(word in review_text for word in ["quality", "build", "construction"]):
        topics.append("build_quality")
    if any(word in review_text for word in ["price", "cost", "value", "money"]):
        topics.append("value_for_money")
    if any(word in review_text for word in ["fast", "speed", "performance", "quick"]):
        topics.append("performance")
    if any(word in review_text for word in ["service", "support", "help", "customer"]):
        topics.append("customer_service")
    
    return {
        "product_id": request.product_id,
        "review_text": request.review_text,
        "sentiment": {
            "label": sentiment,
            "confidence": round(confidence, 3),
            "score": round(positive_count - negative_count, 2)
        },
        "topics": topics,
        "insights": {
            "word_count": len(request.review_text.split()),
            "key_phrases": ["great product", "value for money", "highly recommend"][:len(topics)],
            "rating_prediction": round(3.5 + (positive_count - negative_count) * 0.5, 1)
        },
        "analyzed_at": datetime.now().isoformat()
    }

@app.get("/api/v1/analytics/dashboard")
async def get_analytics_dashboard():
    """Get analytics dashboard data"""
    return {
        "summary": {
            "total_products": 1247,
            "active_users": 3521,
            "conversion_rate": 2.8,
            "revenue_today": 45678.90,
            "ai_agents_active": 4
        },
        "performance_metrics": {
            "avg_response_time": "245ms",
            "uptime": "99.9%",
            "requests_per_minute": 1834,
            "error_rate": "0.02%"
        },
        "business_metrics": {
            "recommendation_accuracy": 87.3,
            "customer_satisfaction": 4.2,
            "agent_effectiveness": 89.1,
            "automation_rate": 78.5
        },
        "real_time_stats": {
            "current_users": 847,
            "chat_sessions": 156,
            "recommendations_served": 2341,
            "reviews_analyzed": 89
        },
        "last_updated": datetime.now().isoformat()
    }

@app.post("/api/v1/agents/workflow/trigger")
async def trigger_workflow(workflow_data: Dict[str, Any]):
    """Trigger agent workflow"""
    workflow_id = workflow_data.get("workflow_id", "unknown")
    
    workflows = {
        "product_onboarding": {
            "description": "Complete product onboarding workflow",
            "steps": [
                "Product data validation",
                "Image processing and analysis",
                "Description generation",
                "SEO optimization",
                "Category classification",
                "Price analysis",
                "Review setup"
            ],
            "estimated_duration": "5-10 minutes"
        },
        "customer_support": {
            "description": "Automated customer support workflow",
            "steps": [
                "Intent classification",
                "Context analysis",
                "Knowledge base search",
                "Response generation",
                "Escalation if needed"
            ],
            "estimated_duration": "1-3 minutes"
        },
        "review_analysis": {
            "description": "Comprehensive review analysis workflow",
            "steps": [
                "Sentiment analysis",
                "Topic extraction",
                "Quality assessment",
                "Insight generation",
                "Alert creation if needed"
            ],
            "estimated_duration": "2-5 minutes"
        }
    }
    
    workflow = workflows.get(workflow_id, {
        "description": "Generic workflow",
        "steps": ["Processing request"],
        "estimated_duration": "1-2 minutes"
    })
    
    return {
        "workflow_id": workflow_id,
        "status": "initiated",
        "workflow": workflow,
        "execution_id": f"exec_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "started_at": datetime.now().isoformat(),
        "message": f"Workflow '{workflow_id}' has been successfully triggered and is now running."
    }

def print_demo_banner():
    """Print demo startup banner"""
    banner = """
    ╔══════════════════════════════════════════════════════════════════╗
    ║                     🤖 EcommerceAgents Demo                      ║
    ║              Multi-Agent E-Commerce Intelligence System          ║
    ║                                                                  ║
    ║  ✅ Phase 4: Chatbot Development                                ║
    ║  ✅ Phase 5: Product Description Agent                          ║
    ║  ✅ Phase 6: Integration & Optimization                          ║
    ╚══════════════════════════════════════════════════════════════════╝
    
    🌐 Demo Server Starting...
    📚 API Documentation: http://localhost:8000/docs
    🏠 Demo Homepage: http://localhost:8000
    🏥 Health Check: http://localhost:8000/health
    📊 Metrics: http://localhost:8000/metrics
    
    🚀 Ready to demonstrate the EcommerceAgents system!
    """
    print(banner)

if __name__ == "__main__":
    print_demo_banner()
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")