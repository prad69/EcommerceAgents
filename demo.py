#!/usr/bin/env python3
"""
EcommerceAgents Demo Script
This script demonstrates the key features of the multi-agent e-commerce system
"""

import asyncio
import uvicorn
from datetime import datetime
from pathlib import Path

# Add src to Python path
import sys
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.main import app

def print_banner():
    """Print demo banner"""
    banner = """
    ╔══════════════════════════════════════════════════════════════════╗
    ║                     EcommerceAgents Demo                         ║
    ║              Multi-Agent E-Commerce Intelligence System          ║
    ║                                                                  ║
    ║  🤖 Phase 4: Chatbot Development                                ║
    ║  📝 Phase 5: Product Description Agent                          ║
    ║  ⚡ Phase 6: Integration & Optimization                          ║
    ║                                                                  ║
    ║  Features:                                                       ║
    ║  • Intelligent Product Recommendations                          ║
    ║  • Automated Review Analysis                                     ║
    ║  • AI-Powered Chatbot                                          ║
    ║  • Product Description Generation                               ║
    ║  • Performance Monitoring                                       ║
    ║  • Auto-Scaling & Analytics                                     ║
    ╚══════════════════════════════════════════════════════════════════╝
    """
    print(banner)

def print_demo_info():
    """Print demo information"""
    info = f"""
    🚀 Starting EcommerceAgents Demo Server...
    
    📅 Demo Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    
    🌐 API Endpoints:
    • Main API: http://localhost:8000
    • API Documentation: http://localhost:8000/docs
    • Health Check: http://localhost:8000/health
    • Metrics: http://localhost:8000/metrics
    
    🎯 Demo Features Available:
    
    1. PRODUCT RECOMMENDATIONS API
       GET /api/v1/recommendations?user_id=123&limit=5
    
    2. REVIEW ANALYSIS API
       POST /api/v1/reviews/analyze
       Body: {{"product_id": "prod123", "review_text": "Great product!"}}
    
    3. CHATBOT API
       POST /api/v1/chatbot/message
       Body: {{"message": "I need help finding a laptop", "session_id": "demo123"}}
    
    4. PRODUCT DESCRIPTION GENERATION
       POST /api/v1/product-description/generate
       Body: {{"product_name": "Laptop", "features": ["Intel i7", "16GB RAM"]}}
    
    5. ANALYTICS DASHBOARD
       GET /api/v1/analytics/dashboard
    
    6. AGENT ORCHESTRATOR
       POST /api/v1/agents/workflow/trigger
       Body: {{"workflow_id": "product_onboarding", "product_data": {{}}}}
    
    💡 Quick Demo Commands:
    
    # Test health endpoint
    curl http://localhost:8000/health
    
    # Get product recommendations
    curl "http://localhost:8000/api/v1/recommendations?user_id=demo_user&limit=3"
    
    # Chat with the AI assistant
    curl -X POST http://localhost:8000/api/v1/chatbot/message \\
         -H "Content-Type: application/json" \\
         -d '{{"message": "Hi, I need help finding a smartphone", "session_id": "demo123"}}'
    
    # Generate product description
    curl -X POST http://localhost:8000/api/v1/product-description/generate \\
         -H "Content-Type: application/json" \\
         -d '{{"product_name": "Gaming Laptop", "features": ["RTX 4080", "32GB RAM", "1TB SSD"]}}'
    
    📊 Monitor the system:
    • Check /metrics for Prometheus metrics
    • View /docs for interactive API documentation
    
    🛑 To stop the demo: Press Ctrl+C
    
    """
    print(info)

def print_demo_scenarios():
    """Print demo scenarios"""
    scenarios = """
    🎬 DEMO SCENARIOS TO TRY:
    
    Scenario 1: Product Discovery
    1. Use chatbot to ask for product recommendations
    2. Get AI-powered suggestions based on preferences
    3. View detailed product analysis
    
    Scenario 2: Review Intelligence
    1. Submit product reviews for analysis
    2. Get sentiment analysis and insights
    3. See aggregated review summaries
    
    Scenario 3: Content Generation
    1. Generate product descriptions from basic info
    2. Get SEO-optimized content
    3. Create A/B test variations
    
    Scenario 4: System Monitoring
    1. Check system health and performance
    2. View analytics dashboard
    3. Monitor agent effectiveness
    
    📱 Frontend (if available):
    If you have the React frontend running on port 3000:
    http://localhost:3000
    
    """
    print(scenarios)

async def demo_api_calls():
    """Demonstrate API calls programmatically"""
    import httpx
    
    print("\n🔄 Running Demo API Calls...\n")
    
    base_url = "http://localhost:8000"
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            
            # 1. Health check
            print("1. Testing Health Check...")
            response = await client.get(f"{base_url}/health")
            print(f"   Status: {response.status_code}")
            print(f"   Response: {response.json()}\n")
            
            # 2. Root endpoint
            print("2. Testing Root Endpoint...")
            response = await client.get(f"{base_url}/")
            print(f"   Status: {response.status_code}")
            print(f"   Response: {response.json()}\n")
            
            # 3. Recommendations (mock data)
            print("3. Testing Product Recommendations...")
            response = await client.get(f"{base_url}/api/v1/recommendations?user_id=demo_user&limit=3")
            print(f"   Status: {response.status_code}")
            if response.status_code == 200:
                print(f"   Response: {response.json()}\n")
            else:
                print(f"   Error: {response.text}\n")
            
            # 4. Chatbot
            print("4. Testing Chatbot...")
            chatbot_payload = {
                "message": "Hi, I'm looking for a good laptop for programming",
                "session_id": "demo_session_123"
            }
            response = await client.post(f"{base_url}/api/v1/chatbot/message", json=chatbot_payload)
            print(f"   Status: {response.status_code}")
            if response.status_code == 200:
                print(f"   Response: {response.json()}\n")
            else:
                print(f"   Error: {response.text}\n")
            
            # 5. Product Description Generation
            print("5. Testing Product Description Generation...")
            description_payload = {
                "product_name": "Gaming Laptop",
                "features": ["Intel Core i9", "RTX 4080", "32GB RAM", "2TB SSD"],
                "target_audience": "gamers",
                "tone": "engaging"
            }
            response = await client.post(f"{base_url}/api/v1/product-description/generate", json=description_payload)
            print(f"   Status: {response.status_code}")
            if response.status_code == 200:
                print(f"   Response: {response.json()}\n")
            else:
                print(f"   Error: {response.text}\n")
            
            print("✅ Demo API calls completed!")
            
    except Exception as e:
        print(f"❌ Error during demo API calls: {e}")

def main():
    """Main demo function"""
    print_banner()
    print_demo_info()
    print_demo_scenarios()
    
    print("🚀 Starting the EcommerceAgents server...\n")
    
    # Start the FastAPI server
    try:
        # Run server with auto-reload for demo
        uvicorn.run(
            "src.main:app",
            host="0.0.0.0",
            port=8000,
            reload=False,  # Set to False to avoid issues in demo
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n\n👋 Demo stopped by user. Thank you for trying EcommerceAgents!")
    except Exception as e:
        print(f"\n❌ Error starting server: {e}")
        print("💡 Make sure all dependencies are installed:")
        print("   pip install -r requirements.txt")

if __name__ == "__main__":
    main()