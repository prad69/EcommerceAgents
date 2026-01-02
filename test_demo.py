#!/usr/bin/env python3
"""
Test script to demonstrate EcommerceAgents API functionality
"""

import asyncio
import json
import sys

async def test_api():
    """Test the running EcommerceAgents API"""
    try:
        import httpx
    except ImportError:
        print("Installing httpx for API testing...")
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "httpx"])
        import httpx

    print("🧪 Testing EcommerceAgents API...")
    print("=" * 60)
    
    base_url = "http://localhost:8000"
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        
        # Test 1: Health Check
        print("\n1. 🏥 Health Check")
        try:
            response = await client.get(f"{base_url}/health")
            print(f"   Status: {response.status_code}")
            print(f"   Response: {json.dumps(response.json(), indent=2)}")
        except Exception as e:
            print(f"   ❌ Error: {e}")
        
        # Test 2: Product Recommendations
        print("\n2. 🎯 Product Recommendations")
        try:
            response = await client.get(f"{base_url}/api/v1/recommendations?user_id=demo_user&limit=3")
            print(f"   Status: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                print(f"   User ID: {data['user_id']}")
                print(f"   Algorithm: {data['algorithm']}")
                print(f"   Found {data['total_found']} recommendations:")
                for rec in data['recommendations']:
                    print(f"     📦 {rec['title']} - ${rec['price']} (Score: {rec['similarity_score']})")
            else:
                print(f"   ❌ Error: {response.text}")
        except Exception as e:
            print(f"   ❌ Error: {e}")
        
        # Test 3: Chatbot Interaction
        print("\n3. 🤖 Chatbot Interaction")
        try:
            chatbot_payload = {
                "message": "Hi, I'm looking for a good laptop for programming and gaming",
                "session_id": "demo_session_123"
            }
            response = await client.post(f"{base_url}/api/v1/chatbot/message", json=chatbot_payload)
            print(f"   Status: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                print(f"   User Message: {chatbot_payload['message']}")
                print(f"   Bot Response: {data['response']}")
                print(f"   Detected Intent: {data['intent']}")
                print(f"   Entities: {data['entities']}")
            else:
                print(f"   ❌ Error: {response.text}")
        except Exception as e:
            print(f"   ❌ Error: {e}")
        
        # Test 4: Product Description Generation
        print("\n4. 📝 Product Description Generation")
        try:
            description_payload = {
                "product_name": "Gaming Laptop Pro",
                "features": ["Intel Core i9-13900H", "NVIDIA RTX 4080", "32GB DDR5 RAM", "2TB NVMe SSD"],
                "target_audience": "gamers",
                "tone": "engaging"
            }
            response = await client.post(f"{base_url}/api/v1/product-description/generate", json=description_payload)
            print(f"   Status: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                print(f"   Product: {data['product_name']}")
                print(f"   Primary Description: {data['descriptions']['primary'][:100]}...")
                print(f"   SEO Title: {data['seo_optimization']['title']}")
                print(f"   Confidence Score: {data['confidence_score']}")
            else:
                print(f"   ❌ Error: {response.text}")
        except Exception as e:
            print(f"   ❌ Error: {e}")
        
        # Test 5: Review Analysis
        print("\n5. 📊 Review Analysis")
        try:
            review_payload = {
                "product_id": "laptop_001",
                "review_text": "This laptop is absolutely amazing! Great build quality, fantastic performance, and excellent value for money. Highly recommend it!"
            }
            response = await client.post(f"{base_url}/api/v1/reviews/analyze", json=review_payload)
            print(f"   Status: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                print(f"   Review: {data['review_text'][:80]}...")
                print(f"   Sentiment: {data['sentiment']['label']} (Confidence: {data['sentiment']['confidence']})")
                print(f"   Topics: {', '.join(data['topics'])}")
                print(f"   Predicted Rating: {data['insights']['rating_prediction']}/5")
            else:
                print(f"   ❌ Error: {response.text}")
        except Exception as e:
            print(f"   ❌ Error: {e}")
        
        # Test 6: Analytics Dashboard
        print("\n6. 📈 Analytics Dashboard")
        try:
            response = await client.get(f"{base_url}/api/v1/analytics/dashboard")
            print(f"   Status: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                print(f"   Total Products: {data['summary']['total_products']}")
                print(f"   Active Users: {data['summary']['active_users']}")
                print(f"   Conversion Rate: {data['summary']['conversion_rate']}%")
                print(f"   Revenue Today: ${data['summary']['revenue_today']:,}")
                print(f"   System Uptime: {data['performance_metrics']['uptime']}")
                print(f"   AI Agent Effectiveness: {data['business_metrics']['agent_effectiveness']}%")
            else:
                print(f"   ❌ Error: {response.text}")
        except Exception as e:
            print(f"   ❌ Error: {e}")
        
        # Test 7: Workflow Trigger
        print("\n7. ⚙️ Agent Workflow Trigger")
        try:
            workflow_payload = {
                "workflow_id": "product_onboarding",
                "product_data": {
                    "name": "Smart Watch Pro",
                    "category": "Electronics",
                    "price": 299.99
                }
            }
            response = await client.post(f"{base_url}/api/v1/agents/workflow/trigger", json=workflow_payload)
            print(f"   Status: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                print(f"   Workflow: {data['workflow_id']}")
                print(f"   Status: {data['status']}")
                print(f"   Execution ID: {data['execution_id']}")
                print(f"   Steps: {len(data['workflow']['steps'])} steps")
                print(f"   Duration: {data['workflow']['estimated_duration']}")
            else:
                print(f"   ❌ Error: {response.text}")
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print("\n" + "=" * 60)
    print("✅ Demo testing completed!")
    print("\n🌐 Try the interactive API documentation at: http://localhost:8000/docs")
    print("🏠 View the demo homepage at: http://localhost:8000")
    print("📊 Check system metrics at: http://localhost:8000/metrics")

if __name__ == "__main__":
    asyncio.run(test_api())