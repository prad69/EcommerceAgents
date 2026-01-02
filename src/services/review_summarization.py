import asyncio
from typing import List, Dict, Any, Optional
from loguru import logger
import json

from src.core.config import settings
from src.models.review import Review, ReviewAnalysis


class ReviewSummarizationService:
    """
    Service for generating review summaries and insights
    """
    
    def __init__(self):
        self.openai_client = None
        self._setup_clients()
    
    def _setup_clients(self):
        """
        Initialize summarization clients
        """
        if settings.openai_api_key:
            try:
                import openai
                self.openai_client = openai.AsyncOpenAI(
                    api_key=settings.openai_api_key
                )
                logger.info("OpenAI summarization client initialized")
            except ImportError:
                logger.warning("OpenAI library not available for summarization")
    
    async def generate_product_summary(
        self,
        product_id: str,
        reviews: List[Review],
        analyses: List[ReviewAnalysis]
    ) -> str:
        """
        Generate a comprehensive product summary from reviews
        """
        if not reviews or not analyses:
            return "No reviews available for summarization."
        
        # Try OpenAI summarization first
        if self.openai_client:
            try:
                return await self._generate_openai_summary(reviews, analyses)
            except Exception as e:
                logger.warning(f"OpenAI summarization failed: {e}")
        
        # Fallback to extractive summarization
        return await self._generate_extractive_summary(reviews, analyses)
    
    async def _generate_openai_summary(
        self,
        reviews: List[Review],
        analyses: List[ReviewAnalysis]
    ) -> str:
        """
        Generate summary using OpenAI GPT
        """
        # Prepare data for summarization
        summary_data = self._prepare_summary_data(reviews, analyses)
        
        # Create prompt
        prompt = self._create_summary_prompt(summary_data)
        
        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert at analyzing customer reviews and creating concise, helpful summaries for potential buyers."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.7
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"OpenAI summary generation failed: {e}")
            raise
    
    async def _generate_extractive_summary(
        self,
        reviews: List[Review],
        analyses: List[ReviewAnalysis]
    ) -> str:
        """
        Generate summary using extractive methods
        """
        # Collect key insights
        summary_data = self._prepare_summary_data(reviews, analyses)
        
        # Build summary sections
        sections = []
        
        # Overall sentiment
        sentiment_counts = summary_data["sentiment_distribution"]
        total = sum(sentiment_counts.values())
        
        if sentiment_counts["positive"] / total > 0.6:
            sections.append("Customers are generally satisfied with this product.")
        elif sentiment_counts["negative"] / total > 0.4:
            sections.append("Customer reviews show mixed to negative feedback.")
        else:
            sections.append("Customer reviews show mixed opinions about this product.")
        
        # Top pros
        if summary_data["top_pros"]:
            pros_text = ", ".join(summary_data["top_pros"][:3])
            sections.append(f"Commonly praised for: {pros_text}.")
        
        # Top cons
        if summary_data["top_cons"]:
            cons_text = ", ".join(summary_data["top_cons"][:3])
            sections.append(f"Common complaints include: {cons_text}.")
        
        # Rating summary
        avg_rating = summary_data["average_rating"]
        sections.append(f"Average rating: {avg_rating:.1f}/5.0 based on {summary_data['total_reviews']} reviews.")
        
        # Verification
        verified_ratio = summary_data["verified_ratio"]
        if verified_ratio > 0.7:
            sections.append("Most reviews are from verified purchases.")
        elif verified_ratio < 0.3:
            sections.append("Note: Many reviews are from unverified purchases.")
        
        return " ".join(sections)
    
    def _prepare_summary_data(
        self,
        reviews: List[Review],
        analyses: List[ReviewAnalysis]
    ) -> Dict[str, Any]:
        """
        Prepare structured data for summarization
        """
        # Basic statistics
        total_reviews = len(reviews)
        average_rating = sum(r.rating for r in reviews) / total_reviews
        verified_ratio = sum(1 for r in reviews if r.verified_purchase) / total_reviews
        
        # Sentiment distribution
        sentiment_counts = {"positive": 0, "neutral": 0, "negative": 0}
        for analysis in analyses:
            sentiment_counts[analysis.overall_sentiment] += 1
        
        # Collect themes, pros, and cons
        all_themes = []
        all_pros = []
        all_cons = []
        
        for analysis in analyses:
            all_themes.extend(analysis.themes or [])
            all_pros.extend(analysis.pros or [])
            all_cons.extend(analysis.cons or [])
        
        # Get most common items
        from collections import Counter
        top_themes = [item for item, count in Counter(all_themes).most_common(5)]
        top_pros = [item for item, count in Counter(all_pros).most_common(5)]
        top_cons = [item for item, count in Counter(all_cons).most_common(5)]
        
        # Aspect analysis
        aspect_sentiments = {}
        for analysis in analyses:
            for aspect, data in (analysis.aspects or {}).items():
                if aspect not in aspect_sentiments:
                    aspect_sentiments[aspect] = []
                aspect_sentiments[aspect].append(data.get("score", 0))
        
        # Average aspect sentiments
        avg_aspects = {}
        for aspect, scores in aspect_sentiments.items():
            avg_aspects[aspect] = sum(scores) / len(scores)
        
        return {
            "total_reviews": total_reviews,
            "average_rating": average_rating,
            "verified_ratio": verified_ratio,
            "sentiment_distribution": sentiment_counts,
            "top_themes": top_themes,
            "top_pros": top_pros,
            "top_cons": top_cons,
            "aspect_sentiments": avg_aspects
        }
    
    def _create_summary_prompt(self, data: Dict[str, Any]) -> str:
        """
        Create a prompt for GPT summarization
        """
        prompt = f"""
        Please create a concise product review summary based on the following data:

        **Product Review Statistics:**
        - Total Reviews: {data['total_reviews']}
        - Average Rating: {data['average_rating']:.1f}/5.0
        - Verified Purchases: {data['verified_ratio']:.1%}
        
        **Sentiment Distribution:**
        - Positive: {data['sentiment_distribution']['positive']} reviews
        - Neutral: {data['sentiment_distribution']['neutral']} reviews  
        - Negative: {data['sentiment_distribution']['negative']} reviews
        
        **Most Common Positive Points:**
        {', '.join(data['top_pros'][:5]) if data['top_pros'] else 'None identified'}
        
        **Most Common Negative Points:**
        {', '.join(data['top_cons'][:5]) if data['top_cons'] else 'None identified'}
        
        **Key Themes Discussed:**
        {', '.join(data['top_themes'][:5]) if data['top_themes'] else 'None identified'}
        """
        
        if data['aspect_sentiments']:
            prompt += "\n**Aspect Analysis:**\n"
            for aspect, score in data['aspect_sentiments'].items():
                sentiment = "positive" if score > 0.3 else "negative" if score < -0.3 else "neutral"
                prompt += f"- {aspect.title()}: {sentiment} sentiment\n"
        
        prompt += """
        
        Please write a helpful 2-3 sentence summary that would help potential buyers understand the overall customer sentiment and key points about this product. Focus on the most important insights and be objective.
        """
        
        return prompt
    
    async def generate_review_highlights(
        self,
        reviews: List[Review],
        analyses: List[ReviewAnalysis],
        max_highlights: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Generate key highlights from reviews
        """
        highlights = []
        
        # Find most helpful positive review
        positive_reviews = [
            (r, a) for r, a in zip(reviews, analyses) 
            if a.overall_sentiment == "positive" and a.helpfulness_predicted > 0.6
        ]
        
        if positive_reviews:
            best_positive = max(positive_reviews, key=lambda x: x[1].helpfulness_predicted)
            highlights.append({
                "type": "positive",
                "title": "Most Helpful Positive Review",
                "content": best_positive[0].content[:200] + "..." if len(best_positive[0].content) > 200 else best_positive[0].content,
                "rating": best_positive[0].rating,
                "helpfulness": best_positive[1].helpfulness_predicted
            })
        
        # Find most helpful critical review
        negative_reviews = [
            (r, a) for r, a in zip(reviews, analyses)
            if a.overall_sentiment == "negative" and a.helpfulness_predicted > 0.6
        ]
        
        if negative_reviews:
            best_negative = max(negative_reviews, key=lambda x: x[1].helpfulness_predicted)
            highlights.append({
                "type": "negative",
                "title": "Most Helpful Critical Review",
                "content": best_negative[0].content[:200] + "..." if len(best_negative[0].content) > 200 else best_negative[0].content,
                "rating": best_negative[0].rating,
                "helpfulness": best_negative[1].helpfulness_predicted
            })
        
        # Find most detailed review
        detailed_reviews = [(r, a) for r, a in zip(reviews, analyses) if r.word_count > 50]
        if detailed_reviews:
            most_detailed = max(detailed_reviews, key=lambda x: x[0].word_count)
            highlights.append({
                "type": "detailed",
                "title": "Most Detailed Review",
                "content": most_detailed[0].content[:200] + "..." if len(most_detailed[0].content) > 200 else most_detailed[0].content,
                "rating": most_detailed[0].rating,
                "word_count": most_detailed[0].word_count
            })
        
        # Find verified purchase reviews
        verified_reviews = [(r, a) for r, a in zip(reviews, analyses) if r.verified_purchase]
        if verified_reviews:
            best_verified = max(verified_reviews, key=lambda x: x[1].helpfulness_predicted)
            highlights.append({
                "type": "verified",
                "title": "Top Verified Purchase Review",
                "content": best_verified[0].content[:200] + "..." if len(best_verified[0].content) > 200 else best_verified[0].content,
                "rating": best_verified[0].rating,
                "verified": True
            })
        
        return highlights[:max_highlights]
    
    async def generate_comparison_summary(
        self,
        product_summaries: List[Dict[str, Any]]
    ) -> str:
        """
        Generate a comparative summary of multiple products
        """
        if len(product_summaries) < 2:
            return "Need at least 2 products for comparison."
        
        # Try OpenAI comparison
        if self.openai_client:
            try:
                return await self._generate_openai_comparison(product_summaries)
            except Exception as e:
                logger.warning(f"OpenAI comparison failed: {e}")
        
        # Fallback to rule-based comparison
        return self._generate_rule_based_comparison(product_summaries)
    
    async def _generate_openai_comparison(self, summaries: List[Dict[str, Any]]) -> str:
        """
        Generate comparison using OpenAI
        """
        prompt = "Compare these products based on customer reviews:\n\n"
        
        for i, summary in enumerate(summaries, 1):
            prompt += f"**Product {i}:**\n"
            prompt += f"- Average Rating: {summary.get('average_rating', 0):.1f}/5.0\n"
            prompt += f"- Total Reviews: {summary.get('total_reviews', 0)}\n"
            prompt += f"- Overall Sentiment: {summary.get('overall_sentiment', 'unknown')}\n"
            prompt += f"- Top Pros: {', '.join(summary.get('common_pros', [])[:3])}\n"
            prompt += f"- Top Cons: {', '.join(summary.get('common_cons', [])[:3])}\n\n"
        
        prompt += "Please provide a concise comparison highlighting the key differences and which product might be better for different use cases."
        
        response = await self.openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful product comparison expert."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=400,
            temperature=0.7
        )
        
        return response.choices[0].message.content.strip()
    
    def _generate_rule_based_comparison(self, summaries: List[Dict[str, Any]]) -> str:
        """
        Generate rule-based comparison
        """
        comparison = []
        
        # Compare ratings
        ratings = [(i, s.get('average_rating', 0)) for i, s in enumerate(summaries)]
        best_rated = max(ratings, key=lambda x: x[1])
        comparison.append(f"Product {best_rated[0] + 1} has the highest rating ({best_rated[1]:.1f}/5.0).")
        
        # Compare review counts
        review_counts = [(i, s.get('total_reviews', 0)) for i, s in enumerate(summaries)]
        most_reviewed = max(review_counts, key=lambda x: x[1])
        comparison.append(f"Product {most_reviewed[0] + 1} has the most reviews ({most_reviewed[1]}).")
        
        # Compare sentiment
        positive_products = []
        for i, summary in enumerate(summaries):
            if summary.get('overall_sentiment') == 'positive':
                positive_products.append(i + 1)
        
        if positive_products:
            comparison.append(f"Products {', '.join(map(str, positive_products))} have overall positive sentiment.")
        
        return " ".join(comparison)