#!/usr/bin/env python3
"""
EcommerceAgents Web Chatbot Interface
A web-based UI for the chatbot using FastAPI and HTML/JavaScript
"""

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import json
import uuid
from datetime import datetime
from typing import Dict, Any, List
import re
import asyncio

# Import the chatbot logic from our previous implementation
class WebChatbot:
    """Web-enabled chatbot with the same intelligence as the CLI version"""
    
    def __init__(self):
        self.products = [
            {
                "id": "laptop_001",
                "name": "MacBook Pro 16-inch M3 Pro",
                "price": 2399.00,
                "category": "Laptops",
                "features": ["M3 Pro chip", "16GB RAM", "512GB SSD", "Liquid Retina XDR"],
                "rating": 4.8,
                "reviews": 1247,
                "use_cases": ["programming", "design", "video editing", "machine learning"],
                "image": "https://via.placeholder.com/300x200/007ACC/fff?text=MacBook+Pro"
            },
            {
                "id": "laptop_002",
                "name": "Dell XPS 13 Developer Edition",
                "price": 1299.00,
                "category": "Laptops", 
                "features": ["Intel i7-13700H", "16GB RAM", "512GB SSD", "Ubuntu 22.04"],
                "rating": 4.6,
                "reviews": 856,
                "use_cases": ["programming", "development", "linux", "coding"],
                "image": "https://via.placeholder.com/300x200/0078D4/fff?text=Dell+XPS+13"
            },
            {
                "id": "laptop_003",
                "name": "ASUS ROG Strix G16 Gaming",
                "price": 1899.00,
                "category": "Laptops",
                "features": ["Intel i9-13980HX", "RTX 4070", "32GB RAM", "1TB SSD"],
                "rating": 4.7,
                "reviews": 943,
                "use_cases": ["gaming", "streaming", "content creation", "video editing"],
                "image": "https://via.placeholder.com/300x200/FF6B00/fff?text=ASUS+ROG"
            },
            {
                "id": "phone_001",
                "name": "iPhone 15 Pro",
                "price": 999.00,
                "category": "Smartphones",
                "features": ["A17 Pro chip", "48MP camera", "6.1-inch display", "Titanium design"],
                "rating": 4.7,
                "reviews": 2105,
                "use_cases": ["photography", "mobile", "apps", "communication"],
                "image": "https://via.placeholder.com/300x200/000000/fff?text=iPhone+15+Pro"
            },
            {
                "id": "headphones_001",
                "name": "Sony WH-1000XM5",
                "price": 399.99,
                "category": "Headphones",
                "features": ["Noise cancelling", "30-hour battery", "Bluetooth 5.2", "Touch controls"],
                "rating": 4.5,
                "reviews": 743,
                "use_cases": ["music", "calls", "travel", "focus"],
                "image": "https://via.placeholder.com/300x200/000000/fff?text=Sony+WH-1000XM5"
            }
        ]
    
    def analyze_intent(self, message: str) -> Dict[str, Any]:
        """Analyze user intent from message"""
        message_lower = message.lower()
        
        intents = {
            "product_search": {
                "keywords": ["looking for", "need", "want", "find", "search", "buy", "purchase", "show me"],
                "categories": {
                    "laptop": ["laptop", "computer", "macbook", "pc", "programming", "coding"],
                    "phone": ["phone", "smartphone", "iphone", "mobile", "cell"],
                    "headphones": ["headphones", "earphones", "audio", "music", "sound"],
                    "monitor": ["monitor", "display", "screen", "ultrawide"]
                }
            },
            "price_inquiry": {
                "keywords": ["price", "cost", "expensive", "cheap", "budget", "afford", "how much"]
            },
            "review_inquiry": {
                "keywords": ["review", "rating", "opinion", "feedback", "experience", "quality"]
            },
            "comparison": {
                "keywords": ["compare", "vs", "versus", "difference", "better", "best"]
            },
            "greeting": {
                "keywords": ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"]
            },
            "help": {
                "keywords": ["help", "assist", "support", "guide", "how", "what can you"]
            }
        }
        
        detected_intent = "general_inquiry"
        confidence = 0.5
        entities = {}
        
        # Check for greetings first
        if any(word in message_lower for word in intents["greeting"]["keywords"]):
            detected_intent = "greeting"
            confidence = 0.9
        
        # Check for help requests
        elif any(word in message_lower for word in intents["help"]["keywords"]):
            detected_intent = "help"
            confidence = 0.85
        
        # Check for product search
        elif any(word in message_lower for word in intents["product_search"]["keywords"]):
            detected_intent = "product_search"
            confidence = 0.8
            
            # Detect product category
            for category, category_words in intents["product_search"]["categories"].items():
                if any(word in message_lower for word in category_words):
                    entities["product_category"] = category
                    confidence = 0.9
                    break
        
        # Check for price inquiries
        elif any(word in message_lower for word in intents["price_inquiry"]["keywords"]):
            detected_intent = "price_inquiry"
            confidence = 0.85
        
        # Check for review inquiries
        elif any(word in message_lower for word in intents["review_inquiry"]["keywords"]):
            detected_intent = "review_inquiry" 
            confidence = 0.8
        
        # Extract budget information
        budget_match = re.search(r'\$?(\d{1,4}(?:,\d{3})*(?:\.\d{2})?)', message)
        if budget_match:
            entities["budget"] = float(budget_match.group(1).replace(',', ''))
        
        # Extract use cases
        use_cases = []
        use_case_patterns = {
            "programming": ["programming", "coding", "development", "dev", "software"],
            "gaming": ["gaming", "games", "esports", "streaming"],
            "design": ["design", "graphics", "photoshop", "creative"],
            "business": ["business", "work", "office", "productivity"],
            "student": ["student", "school", "university", "education"]
        }
        
        for use_case, patterns in use_case_patterns.items():
            if any(pattern in message_lower for pattern in patterns):
                use_cases.append(use_case)
        
        if use_cases:
            entities["use_cases"] = use_cases
        
        return {
            "intent": detected_intent,
            "confidence": confidence,
            "entities": entities,
            "original_message": message
        }
    
    def find_matching_products(self, intent_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find products matching user intent"""
        entities = intent_data.get("entities", {})
        category = entities.get("product_category")
        budget = entities.get("budget")
        use_cases = entities.get("use_cases", [])
        
        matching_products = []
        
        for product in self.products:
            match_score = 0
            
            # Category match
            if category and category in product["category"].lower():
                match_score += 0.4
            
            # Budget match
            if budget:
                if product["price"] <= budget * 1.1:
                    match_score += 0.3
                elif product["price"] <= budget * 1.3:
                    match_score += 0.1
            
            # Use case match
            if use_cases:
                for use_case in use_cases:
                    if use_case in product["use_cases"]:
                        match_score += 0.2
            
            # General relevance
            if match_score == 0:
                match_score = 0.1
            
            if match_score > 0.1:
                product_with_score = product.copy()
                product_with_score["match_score"] = match_score
                matching_products.append(product_with_score)
        
        # Sort by match score and rating
        matching_products.sort(key=lambda x: (x["match_score"], x["rating"]), reverse=True)
        
        return matching_products[:3]
    
    def generate_response(self, intent_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate chatbot response with rich content"""
        intent = intent_data["intent"]
        entities = intent_data.get("entities", {})
        message = intent_data["original_message"]
        
        response_data = {
            "text": "",
            "products": [],
            "quick_replies": [],
            "type": "text"
        }
        
        if intent == "greeting":
            response_data["text"] = "Hello! 👋 I'm your AI shopping assistant. I can help you find the perfect products based on your needs, budget, and preferences. What are you looking for today?"
            response_data["quick_replies"] = [
                "Show me laptops",
                "I need a gaming setup",
                "Budget under $1000",
                "Best smartphones"
            ]
        
        elif intent == "help":
            response_data["text"] = """I can help you with:

🎯 **Product Search**: "Show me gaming laptops" or "I need a phone for photography"
💰 **Budget Shopping**: "Laptops under $2000" or "Best value headphones"
⭐ **Reviews & Ratings**: "Reviews for MacBook Pro"
🔍 **Comparisons**: "Compare iPhone vs Samsung"
💡 **Recommendations**: Just tell me your needs and I'll find the perfect match!

Try asking: "I need a laptop for programming with a budget of $2500" """
            response_data["quick_replies"] = [
                "Show me all products",
                "Gaming laptops",
                "Budget recommendations",
                "Best rated products"
            ]
        
        elif intent == "product_search":
            matching_products = self.find_matching_products(intent_data)
            
            if not matching_products:
                response_data["text"] = "I couldn't find specific products matching your request. Could you be more specific about the type of product, your budget, and how you plan to use it?"
                response_data["quick_replies"] = ["Show all laptops", "Show all phones", "Set my budget"]
            else:
                response_data["text"] = f"Great! I found {len(matching_products)} products that match your needs:"
                response_data["products"] = matching_products
                response_data["type"] = "product_list"
                response_data["quick_replies"] = ["Tell me more", "Compare these", "Show alternatives", "What's your recommendation?"]
        
        elif intent == "price_inquiry":
            if entities.get("budget"):
                budget = entities["budget"]
                matching_products = [p for p in self.products if p["price"] <= budget * 1.1]
                response_data["text"] = f"Here are excellent products within your ${budget:,.0f} budget:"
                response_data["products"] = matching_products[:3]
                response_data["type"] = "product_list"
            else:
                response_data["text"] = "I'd be happy to help with pricing! What's your budget range, or which specific product are you interested in?"
                response_data["quick_replies"] = ["Under $1000", "Under $2000", "$2000-3000", "Show all prices"]
        
        elif intent == "review_inquiry":
            # Show reviews for a relevant product
            product = None
            for p in self.products:
                if any(word in p["name"].lower() for word in message.lower().split()):
                    product = p
                    break
            
            if not product:
                product = self.products[0]  # Default to first product
            
            response_data["text"] = f"**{product['name']}** has excellent reviews!\n\n⭐ {product['rating']}/5 from {product['reviews']:,} customers\n\n**Recent highlights:**\n• \"Amazing performance and build quality!\"\n• \"Best purchase I've made this year\"\n• \"Highly recommend for {', '.join(product['use_cases'][:2])}\"\n\n94% of customers would recommend this product!"
            response_data["products"] = [product]
            response_data["quick_replies"] = ["Show similar products", "Add to cart", "Compare with others"]
        
        else:  # general_inquiry
            response_data["text"] = "I'm here to help you find the perfect products! You can ask me about specific items, set your budget, or tell me what you're looking for."
            response_data["quick_replies"] = [
                "Show popular products",
                "I have a specific budget",
                "Help me choose",
                "What's trending?"
            ]
        
        return response_data

# Create FastAPI app
app = FastAPI(title="EcommerceAgents Chatbot UI")

# Global chatbot instance
chatbot = WebChatbot()

# Store active connections
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.sessions: Dict[str, Dict] = {}

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = {
            "websocket": websocket,
            "conversation": [],
            "created_at": datetime.now()
        }
        await websocket.send_text(json.dumps({
            "type": "session_start",
            "session_id": session_id,
            "message": "Connected to EcommerceAgents AI Assistant!"
        }))
        return session_id

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        # Remove session
        for session_id, session in list(self.sessions.items()):
            if session["websocket"] == websocket:
                del self.sessions[session_id]
                break

    async def send_message(self, websocket: WebSocket, message: dict):
        await websocket.send_text(json.dumps(message))

manager = ConnectionManager()

@app.get("/", response_class=HTMLResponse)
async def get_chatbot_ui():
    """Serve the chatbot UI"""
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>EcommerceAgents AI Chatbot</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }
        
        .chat-container {
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            width: 100%;
            max-width: 800px;
            height: 90vh;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }
        
        .chat-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            text-align: center;
            border-radius: 20px 20px 0 0;
        }
        
        .chat-header h1 {
            font-size: 1.5em;
            margin-bottom: 5px;
        }
        
        .chat-header .subtitle {
            opacity: 0.9;
            font-size: 0.9em;
        }
        
        .status {
            background: #00C851;
            color: white;
            padding: 8px 16px;
            border-radius: 15px;
            font-size: 0.8em;
            display: inline-block;
            margin-top: 10px;
        }
        
        .chat-messages {
            flex: 1;
            overflow-y: auto;
            padding: 20px;
            background: #f8f9fa;
        }
        
        .message {
            margin-bottom: 15px;
            display: flex;
            align-items: flex-start;
        }
        
        .message.user {
            justify-content: flex-end;
        }
        
        .message.bot {
            justify-content: flex-start;
        }
        
        .message-content {
            max-width: 70%;
            padding: 12px 16px;
            border-radius: 18px;
            font-size: 14px;
            line-height: 1.4;
        }
        
        .message.user .message-content {
            background: #007bff;
            color: white;
            border-bottom-right-radius: 4px;
        }
        
        .message.bot .message-content {
            background: white;
            color: #333;
            border: 1px solid #e9ecef;
            border-bottom-left-radius: 4px;
        }
        
        .avatar {
            width: 32px;
            height: 32px;
            border-radius: 50%;
            margin: 0 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 16px;
        }
        
        .avatar.user {
            background: #007bff;
            color: white;
        }
        
        .avatar.bot {
            background: #28a745;
            color: white;
        }
        
        .product-card {
            background: white;
            border: 1px solid #e9ecef;
            border-radius: 10px;
            margin: 10px 0;
            overflow: hidden;
            transition: transform 0.2s;
        }
        
        .product-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }
        
        .product-info {
            padding: 15px;
        }
        
        .product-name {
            font-weight: bold;
            color: #333;
            margin-bottom: 5px;
        }
        
        .product-price {
            color: #007bff;
            font-size: 1.2em;
            font-weight: bold;
            margin-bottom: 8px;
        }
        
        .product-rating {
            color: #ffa500;
            margin-bottom: 8px;
        }
        
        .product-features {
            font-size: 0.9em;
            color: #666;
            margin-bottom: 10px;
        }
        
        .match-score {
            background: #28a745;
            color: white;
            padding: 4px 8px;
            border-radius: 10px;
            font-size: 0.8em;
            display: inline-block;
        }
        
        .quick-replies {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 10px;
        }
        
        .quick-reply {
            background: #e9ecef;
            border: none;
            padding: 8px 12px;
            border-radius: 15px;
            cursor: pointer;
            font-size: 0.9em;
            transition: background-color 0.2s;
        }
        
        .quick-reply:hover {
            background: #dee2e6;
        }
        
        .chat-input {
            background: white;
            border-top: 1px solid #e9ecef;
            padding: 20px;
            display: flex;
            gap: 10px;
        }
        
        #messageInput {
            flex: 1;
            border: 2px solid #e9ecef;
            border-radius: 25px;
            padding: 12px 20px;
            font-size: 14px;
            outline: none;
            transition: border-color 0.2s;
        }
        
        #messageInput:focus {
            border-color: #007bff;
        }
        
        #sendButton {
            background: #007bff;
            color: white;
            border: none;
            border-radius: 50%;
            width: 48px;
            height: 48px;
            cursor: pointer;
            font-size: 18px;
            transition: background-color 0.2s;
        }
        
        #sendButton:hover {
            background: #0056b3;
        }
        
        .typing-indicator {
            display: flex;
            align-items: center;
            margin: 10px 0;
            opacity: 0;
            transition: opacity 0.3s;
        }
        
        .typing-indicator.show {
            opacity: 1;
        }
        
        .typing-dots {
            display: flex;
            gap: 4px;
            margin-left: 10px;
        }
        
        .typing-dot {
            width: 8px;
            height: 8px;
            background: #999;
            border-radius: 50%;
            animation: typing 1.4s infinite ease-in-out;
        }
        
        .typing-dot:nth-child(1) { animation-delay: -0.32s; }
        .typing-dot:nth-child(2) { animation-delay: -0.16s; }
        
        @keyframes typing {
            0%, 80%, 100% { transform: scale(0); }
            40% { transform: scale(1); }
        }
        
        .connection-status {
            text-align: center;
            padding: 10px;
            font-size: 0.9em;
            color: #666;
        }
        
        .connected { color: #28a745; }
        .disconnected { color: #dc3545; }
    </style>
</head>
<body>
    <div class="chat-container">
        <div class="chat-header">
            <h1>🤖 EcommerceAgents AI Assistant</h1>
            <p class="subtitle">Your Intelligent Shopping Companion</p>
            <div class="status" id="connectionStatus">🔄 Connecting...</div>
        </div>
        
        <div class="chat-messages" id="chatMessages">
            <div class="connection-status">
                <em>Starting conversation with AI assistant...</em>
            </div>
        </div>
        
        <div class="typing-indicator" id="typingIndicator">
            <div class="avatar bot">🤖</div>
            <div class="typing-dots">
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
            </div>
        </div>
        
        <div class="chat-input">
            <input type="text" id="messageInput" placeholder="Ask me about products, prices, or anything else..." maxlength="500">
            <button id="sendButton">➤</button>
        </div>
    </div>

    <script>
        class ChatbotUI {
            constructor() {
                this.websocket = null;
                this.sessionId = null;
                this.messageInput = document.getElementById('messageInput');
                this.sendButton = document.getElementById('sendButton');
                this.chatMessages = document.getElementById('chatMessages');
                this.connectionStatus = document.getElementById('connectionStatus');
                this.typingIndicator = document.getElementById('typingIndicator');
                
                this.connect();
                this.setupEventListeners();
            }
            
            connect() {
                const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
                this.websocket = new WebSocket(`${protocol}//${location.host}/ws`);
                
                this.websocket.onopen = () => {
                    this.connectionStatus.innerHTML = '🟢 Connected';
                    this.connectionStatus.className = 'status connected';
                };
                
                this.websocket.onmessage = (event) => {
                    const data = JSON.parse(event.data);
                    this.handleMessage(data);
                };
                
                this.websocket.onclose = () => {
                    this.connectionStatus.innerHTML = '🔴 Disconnected';
                    this.connectionStatus.className = 'status disconnected';
                };
                
                this.websocket.onerror = (error) => {
                    console.error('WebSocket error:', error);
                };
            }
            
            setupEventListeners() {
                this.messageInput.addEventListener('keypress', (e) => {
                    if (e.key === 'Enter') {
                        this.sendMessage();
                    }
                });
                
                this.sendButton.addEventListener('click', () => {
                    this.sendMessage();
                });
            }
            
            sendMessage() {
                const message = this.messageInput.value.trim();
                if (!message) return;
                
                // Display user message
                this.addMessage(message, 'user');
                
                // Clear input
                this.messageInput.value = '';
                
                // Show typing indicator
                this.showTyping();
                
                // Send to backend
                this.websocket.send(JSON.stringify({
                    type: 'message',
                    content: message,
                    session_id: this.sessionId
                }));
            }
            
            addMessage(content, sender) {
                const messageDiv = document.createElement('div');
                messageDiv.className = `message ${sender}`;
                
                if (sender === 'bot') {
                    messageDiv.innerHTML = `
                        <div class="avatar bot">🤖</div>
                        <div class="message-content">${content}</div>
                    `;
                } else {
                    messageDiv.innerHTML = `
                        <div class="message-content">${content}</div>
                        <div class="avatar user">👤</div>
                    `;
                }
                
                this.chatMessages.appendChild(messageDiv);
                this.scrollToBottom();
            }
            
            addProductCards(products) {
                const messageDiv = document.createElement('div');
                messageDiv.className = 'message bot';
                
                let productsHTML = '';
                products.forEach(product => {
                    const matchScore = product.match_score ? `<div class="match-score">${Math.round(product.match_score * 100)}% match</div>` : '';
                    
                    productsHTML += `
                        <div class="product-card">
                            <div class="product-info">
                                <div class="product-name">${product.name}</div>
                                <div class="product-price">$${product.price.toLocaleString()}</div>
                                <div class="product-rating">⭐ ${product.rating}/5 (${product.reviews.toLocaleString()} reviews)</div>
                                <div class="product-features">Features: ${product.features.join(', ')}</div>
                                ${matchScore}
                            </div>
                        </div>
                    `;
                });
                
                messageDiv.innerHTML = `
                    <div class="avatar bot">🤖</div>
                    <div class="message-content">${productsHTML}</div>
                `;
                
                this.chatMessages.appendChild(messageDiv);
                this.scrollToBottom();
            }
            
            addQuickReplies(replies) {
                const repliesDiv = document.createElement('div');
                repliesDiv.className = 'quick-replies';
                
                replies.forEach(reply => {
                    const button = document.createElement('button');
                    button.className = 'quick-reply';
                    button.textContent = reply;
                    button.onclick = () => {
                        this.messageInput.value = reply;
                        this.sendMessage();
                    };
                    repliesDiv.appendChild(button);
                });
                
                this.chatMessages.appendChild(repliesDiv);
                this.scrollToBottom();
            }
            
            handleMessage(data) {
                this.hideTyping();
                
                if (data.type === 'session_start') {
                    this.sessionId = data.session_id;
                    this.addMessage("Hello! I'm your AI shopping assistant. I can help you find products, compare prices, read reviews, and provide personalized recommendations. What are you looking for today? 😊", 'bot');
                    return;
                }
                
                if (data.type === 'bot_response') {
                    // Add text response
                    if (data.response.text) {
                        this.addMessage(data.response.text.replace(/\\n/g, '<br>'), 'bot');
                    }
                    
                    // Add product cards
                    if (data.response.products && data.response.products.length > 0) {
                        this.addProductCards(data.response.products);
                    }
                    
                    // Add quick replies
                    if (data.response.quick_replies && data.response.quick_replies.length > 0) {
                        this.addQuickReplies(data.response.quick_replies);
                    }
                }
            }
            
            showTyping() {
                this.typingIndicator.classList.add('show');
                this.scrollToBottom();
            }
            
            hideTyping() {
                this.typingIndicator.classList.remove('show');
            }
            
            scrollToBottom() {
                setTimeout(() => {
                    this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
                }, 100);
            }
        }
        
        // Initialize chatbot when page loads
        document.addEventListener('DOMContentLoaded', () => {
            new ChatbotUI();
        });
    </script>
</body>
</html>
    """
    return HTMLResponse(content=html_content)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    session_id = await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            if message_data["type"] == "message":
                user_message = message_data["content"]
                
                # Process with chatbot
                intent_data = chatbot.analyze_intent(user_message)
                response = chatbot.generate_response(intent_data)
                
                # Add debug info
                response["debug"] = {
                    "intent": intent_data["intent"],
                    "confidence": intent_data["confidence"],
                    "entities": intent_data.get("entities", {})
                }
                
                # Send response
                await manager.send_message(websocket, {
                    "type": "bot_response",
                    "response": response,
                    "timestamp": datetime.now().isoformat()
                })
                
                # Store in session history
                if session_id in manager.sessions:
                    manager.sessions[session_id]["conversation"].append({
                        "user": user_message,
                        "bot": response,
                        "timestamp": datetime.now().isoformat(),
                        "intent": intent_data
                    })
                    
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "EcommerceAgents Chatbot UI",
        "active_connections": len(manager.active_connections),
        "active_sessions": len(manager.sessions)
    }

if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║                🌐 EcommerceAgents Chatbot Web UI                 ║
    ║                                                                  ║
    ║  🚀 Starting web server...                                       ║
    ║  📱 Open your browser to: http://localhost:8001                  ║
    ║  💬 Chat with the AI assistant in your browser!                  ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)
    
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")