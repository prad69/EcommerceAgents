#!/usr/bin/env python3
"""
Test script to verify the chatbot web UI is working
"""

import asyncio
import websockets
import json
from datetime import datetime

async def test_chatbot():
    """Test the chatbot WebSocket connection"""
    uri = "ws://localhost:8001/ws"
    
    print("🧪 Testing EcommerceAgents Chatbot Web UI...")
    print("=" * 50)
    
    try:
        async with websockets.connect(uri) as websocket:
            print("✅ Connected to chatbot WebSocket")
            
            # Wait for session start message
            session_msg = await websocket.recv()
            session_data = json.loads(session_msg)
            print(f"📱 Session ID: {session_data.get('session_id', 'N/A')}")
            
            # Test messages to send
            test_messages = [
                "Hi, I need help finding a laptop for programming",
                "My budget is around $2500",
                "What are the reviews like?"
            ]
            
            for i, message in enumerate(test_messages, 1):
                print(f"\n{i}. 👤 Sending: '{message}'")
                
                # Send message
                await websocket.send(json.dumps({
                    "type": "message",
                    "content": message,
                    "session_id": session_data.get('session_id')
                }))
                
                # Wait for response
                response_msg = await websocket.recv()
                response_data = json.loads(response_msg)
                
                if response_data.get("type") == "bot_response":
                    bot_text = response_data["response"]["text"]
                    intent = response_data["response"]["debug"]["intent"]
                    confidence = response_data["response"]["debug"]["confidence"]
                    
                    print(f"   🤖 Bot responded: {bot_text[:100]}...")
                    print(f"   🧠 Intent: {intent} (confidence: {confidence:.1%})")
                    
                    if response_data["response"].get("products"):
                        print(f"   📦 Products shown: {len(response_data['response']['products'])}")
                    
                    if response_data["response"].get("quick_replies"):
                        print(f"   💬 Quick replies: {len(response_data['response']['quick_replies'])}")
                
                await asyncio.sleep(1)
            
            print(f"\n✅ Chatbot test completed successfully!")
            print(f"🌐 Web UI is running at: http://localhost:8001")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_chatbot())