#!/usr/bin/env python3
"""
EcommerceAgents Interactive Chatbot Demo
"""

import asyncio
import json
import uuid
from datetime import datetime
from typing import Dict, Any, List
import re

class EcommerceChatbot:
    """Interactive chatbot for the EcommerceAgents system"""
    
    def __init__(self):
        self.session_id = str(uuid.uuid4())
        self.conversation_history = []
        self.user_context = {
            "preferences": {},
            "budget_range": None,
            "product_interests": [],
            "current_intent": None
        }
        
        # Product catalog for recommendations
        self.products = [
            {
                "id": "laptop_001",
                "name": "MacBook Pro 16-inch M3 Pro",
                "price": 2399.00,
                "category": "Laptops",
                "features": ["M3 Pro chip", "16GB RAM", "512GB SSD", "Liquid Retina XDR"],
                "rating": 4.8,
                "reviews": 1247,
                "use_cases": ["programming", "design", "video editing", "machine learning"]
            },
            {
                "id": "laptop_002",
                "name": "Dell XPS 13 Developer Edition",
                "price": 1299.00,
                "category": "Laptops", 
                "features": ["Intel i7-13700H", "16GB RAM", "512GB SSD", "Ubuntu 22.04"],
                "rating": 4.6,
                "reviews": 856,
                "use_cases": ["programming", "development", "linux", "coding"]
            },
            {
                "id": "laptop_003",
                "name": "ASUS ROG Strix G16 Gaming",
                "price": 1899.00,
                "category": "Laptops",
                "features": ["Intel i9-13980HX", "RTX 4070", "32GB RAM", "1TB SSD"],
                "rating": 4.7,
                "reviews": 943,
                "use_cases": ["gaming", "streaming", "content creation", "video editing"]
            },
            {
                "id": "phone_001",
                "name": "iPhone 15 Pro",
                "price": 999.00,
                "category": "Smartphones",
                "features": ["A17 Pro chip", "48MP camera", "6.1-inch display", "Titanium design"],
                "rating": 4.7,
                "reviews": 2105,
                "use_cases": ["photography", "mobile", "apps", "communication"]
            },
            {
                "id": "headphones_001",
                "name": "Sony WH-1000XM5",
                "price": 399.99,
                "category": "Headphones",
                "features": ["Noise cancelling", "30-hour battery", "Bluetooth 5.2", "Touch controls"],
                "rating": 4.5,
                "reviews": 743,
                "use_cases": ["music", "calls", "travel", "focus"]
            },
            {
                "id": "monitor_001",
                "name": "LG UltraWide 34WK95U",
                "price": 899.00,
                "category": "Monitors",
                "features": ["34-inch 5K", "USB-C Hub", "HDR600", "Thunderbolt 3"],
                "rating": 4.5,
                "reviews": 521,
                "use_cases": ["programming", "design", "productivity", "multitasking"]
            }
        ]
        
    def print_banner(self):
        """Print chatbot banner"""
        banner = """
    ╔══════════════════════════════════════════════════════════════════╗
    ║                🤖 EcommerceAgents AI Chatbot                     ║
    ║                  Your Intelligent Shopping Assistant             ║
    ╚══════════════════════════════════════════════════════════════════╝
    
    👋 Hello! I'm your AI shopping assistant. I can help you:
    🎯 Find the perfect products based on your needs
    💰 Compare prices and features
    ⭐ Check reviews and ratings
    🔍 Answer questions about our products
    💡 Provide personalized recommendations
    
    💬 Just type your message and press Enter to chat!
    📝 Type 'help' for more commands, 'quit' to exit
    
    🕐 Session started: """ + datetime.now().strftime("%H:%M:%S") + """
    🆔 Session ID: """ + self.session_id[:8] + """...
        """
        print(banner)
    
    def analyze_intent(self, message: str) -> Dict[str, Any]:
        """Analyze user intent from message"""
        message_lower = message.lower()
        
        intents = {
            "product_search": {
                "keywords": ["looking for", "need", "want", "find", "search", "buy", "purchase"],
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
            "technical_specs": {
                "keywords": ["specs", "specification", "features", "technical", "details"]
            },
            "greeting": {
                "keywords": ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"]
            },
            "help": {
                "keywords": ["help", "assist", "support", "guide", "how"]
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
        
        # Check for comparisons
        elif any(word in message_lower for word in intents["comparison"]["keywords"]):
            detected_intent = "comparison"
            confidence = 0.8
        
        # Check for technical specs
        elif any(word in message_lower for word in intents["technical_specs"]["keywords"]):
            detected_intent = "technical_specs"
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
            "student": ["student", "school", "university", "education"],
            "music": ["music", "audio", "listening", "sound"],
            "video": ["video", "editing", "youtube", "content creation"]
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
                if product["price"] <= budget * 1.1:  # 10% tolerance
                    match_score += 0.3
                elif product["price"] <= budget * 1.3:  # 30% tolerance
                    match_score += 0.1
            
            # Use case match
            if use_cases:
                for use_case in use_cases:
                    if use_case in product["use_cases"]:
                        match_score += 0.2
            
            # General relevance
            if match_score == 0:
                match_score = 0.1  # Base relevance
            
            if match_score > 0.1:
                product_with_score = product.copy()
                product_with_score["match_score"] = match_score
                matching_products.append(product_with_score)
        
        # Sort by match score and rating
        matching_products.sort(key=lambda x: (x["match_score"], x["rating"]), reverse=True)
        
        return matching_products[:3]  # Return top 3 matches
    
    def generate_response(self, intent_data: Dict[str, Any]) -> str:
        """Generate chatbot response based on intent"""
        intent = intent_data["intent"]
        entities = intent_data.get("entities", {})
        message = intent_data["original_message"]
        
        if intent == "greeting":
            return ("Hello! 👋 I'm your AI shopping assistant. I'm here to help you find the perfect products. "
                   "What are you looking for today? You can tell me about your needs, budget, or the type of product you want!")
        
        elif intent == "help":
            return """I can help you with:
🎯 **Product Search**: "I need a laptop for programming" or "Looking for gaming headphones"
💰 **Price & Budget**: "Show me laptops under $2000" or "What's the price of iPhone 15?"
⭐ **Reviews & Ratings**: "What are the reviews for MacBook Pro?" 
🔍 **Comparisons**: "Compare iPhone vs Samsung" or "MacBook vs Dell XPS"
📋 **Specifications**: "What are the specs of the gaming laptop?"
💡 **Recommendations**: Just describe your needs and I'll suggest the best products!

Try asking something like: "I need a laptop for machine learning with a budget of $3000" """
        
        elif intent == "product_search":
            matching_products = self.find_matching_products(intent_data)
            
            if not matching_products:
                return ("I couldn't find specific products matching your request. Could you be more specific? "
                       "For example, tell me: the type of product, your budget, and how you plan to use it.")
            
            # Update user context
            self.user_context["current_intent"] = "product_search"
            if entities.get("budget"):
                self.user_context["budget_range"] = entities["budget"]
            if entities.get("product_category"):
                self.user_context["product_interests"].append(entities["product_category"])
            
            response = f"Great! I found {len(matching_products)} products that match your needs:\n\n"
            
            for i, product in enumerate(matching_products, 1):
                response += f"**{i}. {product['name']}** - ${product['price']:,.2f}\n"
                response += f"   ⭐ {product['rating']}/5 ({product['reviews']:,} reviews)\n"
                response += f"   🔧 Features: {', '.join(product['features'][:3])}\n"
                response += f"   🎯 Match Score: {product['match_score']*100:.0f}%\n\n"
            
            response += "Would you like more details about any of these products, or would you like to refine your search?"
            return response
        
        elif intent == "price_inquiry":
            if entities.get("budget"):
                budget = entities["budget"]
                matching_products = [p for p in self.products if p["price"] <= budget * 1.1]
                response = f"Here are great products within your ${budget:,.0f} budget:\n\n"
                
                for product in matching_products[:3]:
                    response += f"• **{product['name']}** - ${product['price']:,.2f}\n"
                
                return response
            else:
                return ("I'd be happy to help with pricing! Could you tell me which specific product you're interested in, "
                       "or what your budget range is? For example: 'What laptops are under $2000?' or 'Price of iPhone 15?'")
        
        elif intent == "review_inquiry":
            # For demo, show reviews for first matching product
            if "macbook" in message.lower():
                product = next((p for p in self.products if "macbook" in p["name"].lower()), None)
            elif "iphone" in message.lower():
                product = next((p for p in self.products if "iphone" in p["name"].lower()), None)
            else:
                product = self.products[0]  # Default to first product
            
            if product:
                return f"""**{product['name']}** has excellent reviews! ⭐ {product['rating']}/5

**Recent Customer Feedback:**
👍 "Amazing performance and build quality - exactly what I needed!"
👍 "Best purchase I've made this year. Highly recommend!"
👍 "Fast, reliable, and worth every penny."
👍 "Great for {', '.join(product['use_cases'][:2])}"

**Summary from {product['reviews']:,} reviews:**
• 94% would recommend this product
• Average satisfaction: {product['rating']}/5
• Top mentioned: Performance, Quality, Value

Would you like to see more detailed reviews or information about this product?"""
        
        elif intent == "comparison":
            return ("I'd be happy to compare products for you! Could you tell me which specific products you'd like to compare? "
                   "For example: 'Compare MacBook Pro vs Dell XPS' or 'iPhone vs Samsung Galaxy'")
        
        elif intent == "technical_specs":
            return ("I can provide detailed specifications! Which product would you like to know more about? "
                   "You can ask: 'Specs for MacBook Pro' or 'Features of gaming laptop'")
        
        else:  # general_inquiry
            return ("I'm here to help you find the perfect products! You can ask me about:\n"
                   "🔍 Specific products: 'Show me gaming laptops'\n"
                   "💰 Budget options: 'Best phone under $800'\n"
                   "⭐ Reviews: 'Reviews for Sony headphones'\n"
                   "🆚 Comparisons: 'MacBook vs Windows laptop'\n\n"
                   "What would you like to explore?")
    
    def add_to_conversation(self, user_message: str, bot_response: str, intent_data: Dict[str, Any]):
        """Add exchange to conversation history"""
        self.conversation_history.append({
            "timestamp": datetime.now().isoformat(),
            "user_message": user_message,
            "bot_response": bot_response,
            "intent": intent_data["intent"],
            "confidence": intent_data["confidence"],
            "entities": intent_data.get("entities", {})
        })
    
    def get_conversation_stats(self):
        """Get conversation statistics"""
        if not self.conversation_history:
            return "No conversation yet."
        
        intents = [exchange["intent"] for exchange in self.conversation_history]
        intent_counts = {}
        for intent in intents:
            intent_counts[intent] = intent_counts.get(intent, 0) + 1
        
        avg_confidence = sum(exchange["confidence"] for exchange in self.conversation_history) / len(self.conversation_history)
        
        return f"""
📊 **Conversation Statistics:**
• Messages exchanged: {len(self.conversation_history)}
• Average confidence: {avg_confidence:.1%}
• Top intents: {', '.join(list(intent_counts.keys())[:3])}
• Session duration: {datetime.now().strftime("%H:%M:%S")}
        """
    
    async def start_chat(self):
        """Start interactive chat session"""
        self.print_banner()
        
        try:
            while True:
                print("\n" + "="*60)
                user_input = input("👤 You: ").strip()
                
                if not user_input:
                    continue
                
                # Handle special commands
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("\n🤖 Assistant: Thank you for chatting! Have a great day! 👋")
                    print(self.get_conversation_stats())
                    break
                
                elif user_input.lower() in ['help', '?']:
                    intent_data = {"intent": "help", "confidence": 1.0, "entities": {}, "original_message": user_input}
                    response = self.generate_response(intent_data)
                    print(f"\n🤖 Assistant: {response}")
                    continue
                
                elif user_input.lower() == 'stats':
                    print(self.get_conversation_stats())
                    continue
                
                elif user_input.lower() == 'clear':
                    self.conversation_history.clear()
                    print("\n🤖 Assistant: Conversation history cleared! How can I help you?")
                    continue
                
                # Process user message
                print("\n🧠 Processing your message...")
                await asyncio.sleep(0.5)  # Simulate processing time
                
                # Analyze intent
                intent_data = self.analyze_intent(user_input)
                
                # Generate response
                response = self.generate_response(intent_data)
                
                # Display response
                print(f"\n🤖 Assistant: {response}")
                
                # Show debug info
                print(f"\n🔍 Debug Info:")
                print(f"   Intent: {intent_data['intent']} (confidence: {intent_data['confidence']:.1%})")
                if intent_data.get("entities"):
                    print(f"   Entities: {intent_data['entities']}")
                
                # Save to conversation history
                self.add_to_conversation(user_input, response, intent_data)
                
        except KeyboardInterrupt:
            print("\n\n👋 Chat session ended. Thanks for using EcommerceAgents chatbot!")
            print(self.get_conversation_stats())

def main():
    """Main function to start the chatbot"""
    chatbot = EcommerceChatbot()
    
    print("🚀 Starting EcommerceAgents Chatbot...")
    print("💡 Tip: Try asking about laptops, phones, or any products you need!")
    
    # Run the async chat
    asyncio.run(chatbot.start_chat())

if __name__ == "__main__":
    main()