import asyncio
from typing import Dict, List, Any, Optional, Tuple
from loguru import logger
from datetime import datetime
import numpy as np
import json

from src.core.config import settings
from src.services.text_processing import TextProcessor


class SentimentAnalysisService:
    """
    Advanced sentiment analysis service using multiple approaches
    """
    
    def __init__(self):
        self.text_processor = TextProcessor()
        self.transformers_model = None
        self.openai_client = None
        self._setup_models()
    
    def _setup_models(self):
        """
        Initialize sentiment analysis models
        """
        # Setup Transformers model
        try:
            from transformers import pipeline
            self.transformers_model = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                return_all_scores=True
            )
            logger.info("Transformers sentiment model loaded")
        except ImportError:
            logger.warning("Transformers library not available")
        except Exception as e:
            logger.warning(f"Failed to load transformers model: {e}")
        
        # Setup OpenAI as fallback
        if settings.openai_api_key:
            try:
                import openai
                self.openai_client = openai.AsyncOpenAI(
                    api_key=settings.openai_api_key
                )
                logger.info("OpenAI sentiment client initialized")
            except ImportError:
                logger.warning("OpenAI library not available")
    
    async def analyze_sentiment(
        self, 
        text: str, 
        include_aspects: bool = True
    ) -> Dict[str, Any]:
        """
        Comprehensive sentiment analysis
        """
        if not text or len(text.strip()) < 5:
            return self._empty_sentiment_result()
        
        try:
            # Preprocess text
            processed = await self.text_processor.preprocess_text(text, for_sentiment=True)
            
            # Get overall sentiment
            overall_sentiment = await self._get_overall_sentiment(text)
            
            # Extract aspects if requested
            aspect_sentiments = {}
            if include_aspects:
                aspect_sentiments = await self._analyze_aspect_sentiments(text)
            
            # Extract sentiment keywords
            keywords = await self.text_processor.extract_keywords(text)
            
            # Calculate confidence scores
            confidence = await self._calculate_confidence(text, overall_sentiment)
            
            result = {
                "overall_sentiment": overall_sentiment["label"],
                "sentiment_score": overall_sentiment["score"],
                "confidence": confidence,
                "aspect_sentiments": aspect_sentiments,
                "keywords": keywords,
                "text_quality": await self.text_processor.calculate_text_quality_score(text),
                "word_count": processed["word_count"],
                "readability_score": processed["readability_score"],
                "model_used": overall_sentiment.get("model", "unknown"),
                "processed_at": datetime.utcnow().isoformat()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            return self._empty_sentiment_result()
    
    async def _get_overall_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Get overall sentiment using the best available model
        """
        # Try Transformers first
        if self.transformers_model:
            try:
                return await self._analyze_with_transformers(text)
            except Exception as e:
                logger.warning(f"Transformers sentiment failed: {e}")
        
        # Try OpenAI as fallback
        if self.openai_client:
            try:
                return await self._analyze_with_openai(text)
            except Exception as e:
                logger.warning(f"OpenAI sentiment failed: {e}")
        
        # Use rule-based fallback
        return await self._analyze_with_rules(text)
    
    async def _analyze_with_transformers(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment using Transformers model
        """
        def _predict():
            # Truncate text if too long
            max_length = 512
            if len(text) > max_length:
                text_truncated = text[:max_length]
            else:
                text_truncated = text
            
            results = self.transformers_model(text_truncated)
            
            # Convert to standard format
            sentiment_map = {
                'LABEL_0': 'negative',
                'LABEL_1': 'neutral', 
                'LABEL_2': 'positive',
                'NEGATIVE': 'negative',
                'NEUTRAL': 'neutral',
                'POSITIVE': 'positive'
            }
            
            best_result = max(results[0], key=lambda x: x['score'])
            label = sentiment_map.get(best_result['label'], best_result['label'].lower())
            
            # Convert score to -1 to 1 range
            if label == 'positive':
                score = best_result['score']
            elif label == 'negative':
                score = -best_result['score']
            else:  # neutral
                score = 0.0
            
            return {
                "label": label,
                "score": score,
                "confidence": best_result['score'],
                "model": "transformers"
            }
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _predict)
    
    async def _analyze_with_openai(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment using OpenAI API
        """
        try:
            prompt = f"""
            Analyze the sentiment of the following text and respond with a JSON object containing:
            - sentiment: "positive", "negative", or "neutral"
            - score: a number between -1 (very negative) and 1 (very positive)
            - confidence: a number between 0 and 1 indicating confidence
            
            Text: "{text[:1000]}"
            
            Response (JSON only):
            """
            
            response = await self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=0.1
            )
            
            result_text = response.choices[0].message.content.strip()
            
            # Parse JSON response
            try:
                result = json.loads(result_text)
                return {
                    "label": result.get("sentiment", "neutral"),
                    "score": float(result.get("score", 0.0)),
                    "confidence": float(result.get("confidence", 0.5)),
                    "model": "openai"
                }
            except json.JSONDecodeError:
                # Fallback parsing
                if "positive" in result_text.lower():
                    return {"label": "positive", "score": 0.5, "confidence": 0.7, "model": "openai"}
                elif "negative" in result_text.lower():
                    return {"label": "negative", "score": -0.5, "confidence": 0.7, "model": "openai"}
                else:
                    return {"label": "neutral", "score": 0.0, "confidence": 0.5, "model": "openai"}
                    
        except Exception as e:
            logger.error(f"OpenAI sentiment analysis failed: {e}")
            raise
    
    async def _analyze_with_rules(self, text: str) -> Dict[str, Any]:
        """
        Rule-based sentiment analysis fallback
        """
        # Extract keywords
        keywords = await self.text_processor.extract_keywords(text)
        
        positive_count = len(keywords.get('positive', []))
        negative_count = len(keywords.get('negative', []))
        
        # Calculate sentiment
        if positive_count > negative_count:
            label = "positive"
            score = min(0.8, positive_count / (positive_count + negative_count + 1))
        elif negative_count > positive_count:
            label = "negative"
            score = -min(0.8, negative_count / (positive_count + negative_count + 1))
        else:
            label = "neutral"
            score = 0.0
        
        confidence = abs(score) if score != 0 else 0.3
        
        return {
            "label": label,
            "score": score,
            "confidence": confidence,
            "model": "rule_based"
        }
    
    async def _analyze_aspect_sentiments(self, text: str) -> Dict[str, Dict[str, Any]]:
        """
        Analyze sentiment for specific product aspects
        """
        # Extract aspects mentioned in the text
        aspects = await self.text_processor.extract_aspects(text)
        aspect_sentiments = {}
        
        for aspect, keywords in aspects.items():
            # Find sentences mentioning this aspect
            aspect_sentences = []
            sentences = text.split('.')
            
            for sentence in sentences:
                sentence_lower = sentence.lower()
                if any(keyword in sentence_lower for keyword in keywords):
                    aspect_sentences.append(sentence.strip())
            
            if aspect_sentences:
                # Analyze sentiment of aspect-related sentences
                aspect_text = '. '.join(aspect_sentences)
                try:
                    aspect_sentiment = await self._get_overall_sentiment(aspect_text)
                    aspect_sentiments[aspect] = {
                        "sentiment": aspect_sentiment["label"],
                        "score": aspect_sentiment["score"],
                        "confidence": aspect_sentiment["confidence"],
                        "mentions": len(aspect_sentences),
                        "keywords": keywords
                    }
                except Exception as e:
                    logger.warning(f"Failed to analyze aspect sentiment for {aspect}: {e}")
        
        return aspect_sentiments
    
    async def _calculate_confidence(self, text: str, sentiment_result: Dict[str, Any]) -> float:
        """
        Calculate confidence in the sentiment analysis
        """
        base_confidence = sentiment_result.get("confidence", 0.5)
        
        # Adjust confidence based on text quality
        text_quality = await self.text_processor.calculate_text_quality_score(text)
        
        # Adjust confidence based on text length
        word_count = len(text.split())
        length_factor = min(1.0, word_count / 50)  # Full confidence at 50+ words
        
        # Combine factors
        final_confidence = base_confidence * 0.7 + text_quality * 0.2 + length_factor * 0.1
        
        return min(1.0, max(0.1, final_confidence))
    
    def _empty_sentiment_result(self) -> Dict[str, Any]:
        """
        Return empty sentiment result
        """
        return {
            "overall_sentiment": "neutral",
            "sentiment_score": 0.0,
            "confidence": 0.0,
            "aspect_sentiments": {},
            "keywords": {"positive": [], "negative": [], "neutral": []},
            "text_quality": 0.0,
            "word_count": 0,
            "readability_score": 0.0,
            "model_used": "none",
            "processed_at": datetime.utcnow().isoformat()
        }
    
    async def batch_analyze_sentiments(
        self, 
        texts: List[str], 
        include_aspects: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Analyze sentiments for multiple texts
        """
        tasks = [
            self.analyze_sentiment(text, include_aspects) 
            for text in texts
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Batch sentiment analysis failed for text {i}: {result}")
                processed_results.append(self._empty_sentiment_result())
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def calculate_sentiment_trends(
        self, 
        sentiments: List[Dict[str, Any]], 
        time_window: str = "daily"
    ) -> Dict[str, Any]:
        """
        Calculate sentiment trends over time
        """
        if not sentiments:
            return {"trend": "stable", "change": 0.0, "confidence": 0.0}
        
        # Group sentiments by time window
        grouped = {}
        for sentiment in sentiments:
            try:
                timestamp = datetime.fromisoformat(sentiment["processed_at"])
                if time_window == "daily":
                    key = timestamp.date()
                elif time_window == "weekly":
                    week = timestamp.isocalendar()[1]
                    key = f"{timestamp.year}-W{week}"
                else:  # monthly
                    key = f"{timestamp.year}-{timestamp.month}"
                
                if key not in grouped:
                    grouped[key] = []
                grouped[key].append(sentiment["sentiment_score"])
            except Exception:
                continue
        
        if len(grouped) < 2:
            return {"trend": "stable", "change": 0.0, "confidence": 0.0}
        
        # Calculate average sentiment for each time period
        time_averages = []
        for period in sorted(grouped.keys()):
            avg_sentiment = sum(grouped[period]) / len(grouped[period])
            time_averages.append(avg_sentiment)
        
        # Calculate trend
        if len(time_averages) >= 3:
            # Simple linear regression
            x = list(range(len(time_averages)))
            y = time_averages
            
            n = len(x)
            sum_x = sum(x)
            sum_y = sum(y)
            sum_xy = sum(xi * yi for xi, yi in zip(x, y))
            sum_x2 = sum(xi * xi for xi in x)
            
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
            
            if slope > 0.1:
                trend = "improving"
            elif slope < -0.1:
                trend = "declining"
            else:
                trend = "stable"
            
            confidence = min(1.0, abs(slope) * 10)
            
        else:
            # Simple comparison of first and last
            change = time_averages[-1] - time_averages[0]
            if change > 0.2:
                trend = "improving"
            elif change < -0.2:
                trend = "declining"
            else:
                trend = "stable"
            
            confidence = min(1.0, abs(change) * 2)
            slope = change
        
        return {
            "trend": trend,
            "change": slope,
            "confidence": confidence,
            "periods_analyzed": len(grouped)
        }