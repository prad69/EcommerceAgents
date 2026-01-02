import asyncio
from typing import Dict, List, Any, Optional, Tuple
from loguru import logger
import json
import re
from datetime import datetime
from dataclasses import dataclass
import math

from src.core.config import settings
from src.core.database import get_db
from src.models.product_description import SEOKeyword, ProductDescription


@dataclass
class SEOAnalysis:
    """SEO analysis results"""
    seo_score: float
    keyword_density: Dict[str, float]
    title_optimization: Dict[str, Any]
    meta_description_optimization: Dict[str, Any]
    content_optimization: Dict[str, Any]
    recommendations: List[str]
    technical_issues: List[str]
    opportunities: List[str]


@dataclass
class KeywordAnalysis:
    """Keyword analysis results"""
    keyword: str
    search_volume: int
    competition_level: str
    difficulty_score: float
    current_rank: Optional[int]
    density: float
    prominence_score: float
    relevance_score: float


class SEOOptimizerService:
    """
    SEO optimization service for product descriptions
    """
    
    def __init__(self):
        self.openai_client = None
        self.keyword_research_api = None
        self.seo_rules = self._setup_seo_rules()
        self._setup_models()
    
    def _setup_models(self):
        """
        Initialize SEO analysis tools
        """
        # Setup OpenAI for content optimization
        if settings.openai_api_key:
            try:
                import openai
                self.openai_client = openai.AsyncOpenAI(
                    api_key=settings.openai_api_key
                )
                logger.info("OpenAI SEO optimizer initialized")
            except ImportError:
                logger.warning("OpenAI library not available")
    
    def _setup_seo_rules(self) -> Dict[str, Any]:
        """
        Setup SEO optimization rules and guidelines
        """
        return {
            "title_optimization": {
                "min_length": 30,
                "max_length": 60,
                "include_brand": True,
                "include_main_keyword": True,
                "avoid_keyword_stuffing": True,
                "patterns": {
                    "product_title": "{product_name} | {category} | {brand}",
                    "descriptive": "{main_keyword} - {product_name} by {brand}",
                    "benefit_focused": "{primary_benefit} - {product_name}"
                }
            },
            "meta_description": {
                "min_length": 120,
                "max_length": 160,
                "include_main_keyword": True,
                "include_call_to_action": True,
                "compelling_language": True
            },
            "content_optimization": {
                "keyword_density": {
                    "target": 0.02,  # 2%
                    "min": 0.005,    # 0.5%
                    "max": 0.05      # 5%
                },
                "keyword_prominence": {
                    "title_weight": 3.0,
                    "first_paragraph_weight": 2.0,
                    "headings_weight": 2.5,
                    "body_weight": 1.0
                },
                "content_structure": {
                    "min_words": 50,
                    "max_words": 500,
                    "use_headings": True,
                    "bullet_points": True,
                    "short_paragraphs": True
                }
            },
            "technical_seo": {
                "schema_markup": True,
                "url_optimization": True,
                "image_alt_text": True,
                "internal_linking": True
            }
        }
    
    async def analyze_seo_performance(
        self,
        description_id: str,
        target_keywords: Optional[List[str]] = None
    ) -> SEOAnalysis:
        """
        Analyze SEO performance of a product description
        """
        try:
            # Get description from database
            db = next(get_db())
            description = db.query(ProductDescription).filter(
                ProductDescription.id == description_id
            ).first()
            db.close()
            
            if not description:
                raise ValueError(f"Description {description_id} not found")
            
            logger.info(f"Analyzing SEO performance for description: {description.title}")
            
            # Extract content elements
            content_elements = {
                "title": description.title or "",
                "meta_title": description.meta_title or "",
                "meta_description": description.meta_description or "",
                "content": description.content or "",
                "keywords": description.keywords or []
            }
            
            # Use target keywords or extract from description
            if not target_keywords:
                target_keywords = content_elements["keywords"][:5]  # Top 5 stored keywords
            
            # Perform comprehensive SEO analysis
            analysis_tasks = [
                self._analyze_title_optimization(content_elements, target_keywords),
                self._analyze_meta_description(content_elements, target_keywords),
                self._analyze_content_optimization(content_elements, target_keywords),
                self._analyze_keyword_density(content_elements, target_keywords),
                self._check_technical_seo(content_elements)
            ]
            
            results = await asyncio.gather(*analysis_tasks, return_exceptions=True)
            
            # Compile results
            title_analysis = results[0] if not isinstance(results[0], Exception) else {}
            meta_analysis = results[1] if not isinstance(results[1], Exception) else {}
            content_analysis = results[2] if not isinstance(results[2], Exception) else {}
            keyword_analysis = results[3] if not isinstance(results[3], Exception) else {}
            technical_analysis = results[4] if not isinstance(results[4], Exception) else {}
            
            # Calculate overall SEO score
            seo_score = await self._calculate_overall_seo_score({
                "title": title_analysis,
                "meta": meta_analysis,
                "content": content_analysis,
                "keywords": keyword_analysis,
                "technical": technical_analysis
            })
            
            # Generate recommendations
            recommendations = await self._generate_seo_recommendations({
                "title": title_analysis,
                "meta": meta_analysis,
                "content": content_analysis,
                "keywords": keyword_analysis,
                "technical": technical_analysis
            }, target_keywords)
            
            # Identify opportunities
            opportunities = await self._identify_seo_opportunities(
                content_elements, target_keywords
            )
            
            return SEOAnalysis(
                seo_score=seo_score,
                keyword_density=keyword_analysis.get("density_scores", {}),
                title_optimization=title_analysis,
                meta_description_optimization=meta_analysis,
                content_optimization=content_analysis,
                recommendations=recommendations,
                technical_issues=technical_analysis.get("issues", []),
                opportunities=opportunities
            )
            
        except Exception as e:
            logger.error(f"SEO analysis failed: {e}")
            raise
    
    async def _analyze_title_optimization(
        self,
        content: Dict[str, Any],
        target_keywords: List[str]
    ) -> Dict[str, Any]:
        """
        Analyze title SEO optimization
        """
        title = content.get("meta_title") or content.get("title", "")
        
        analysis = {
            "title": title,
            "length": len(title),
            "word_count": len(title.split()),
            "issues": [],
            "score": 0.0
        }
        
        rules = self.seo_rules["title_optimization"]
        score_components = []
        
        # Length check
        if rules["min_length"] <= len(title) <= rules["max_length"]:
            score_components.append(1.0)
        elif len(title) < rules["min_length"]:
            analysis["issues"].append(f"Title too short ({len(title)} chars, min {rules['min_length']})")
            score_components.append(len(title) / rules["min_length"])
        else:
            analysis["issues"].append(f"Title too long ({len(title)} chars, max {rules['max_length']})")
            score_components.append(rules["max_length"] / len(title))
        
        # Keyword presence
        title_lower = title.lower()
        keyword_found = False
        for keyword in target_keywords:
            if keyword.lower() in title_lower:
                keyword_found = True
                analysis["primary_keyword"] = keyword
                break
        
        if keyword_found:
            score_components.append(1.0)
        else:
            analysis["issues"].append("No target keywords found in title")
            score_components.append(0.0)
        
        # Keyword stuffing check
        keyword_count = sum(title_lower.count(kw.lower()) for kw in target_keywords)
        word_count = len(title.split())
        keyword_density = keyword_count / word_count if word_count > 0 else 0
        
        if keyword_density <= 0.3:  # Max 30% keyword density
            score_components.append(1.0)
        else:
            analysis["issues"].append("Potential keyword stuffing detected")
            score_components.append(0.5)
        
        # Uniqueness and appeal
        common_words = ["the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"]
        unique_words = [w for w in title.lower().split() if w not in common_words]
        uniqueness_score = len(set(unique_words)) / max(len(title.split()), 1)
        score_components.append(min(uniqueness_score, 1.0))
        
        analysis["score"] = sum(score_components) / len(score_components)
        analysis["keyword_density"] = keyword_density
        analysis["uniqueness_score"] = uniqueness_score
        
        return analysis
    
    async def _analyze_meta_description(
        self,
        content: Dict[str, Any],
        target_keywords: List[str]
    ) -> Dict[str, Any]:
        """
        Analyze meta description optimization
        """
        meta_desc = content.get("meta_description", "")
        
        analysis = {
            "meta_description": meta_desc,
            "length": len(meta_desc),
            "issues": [],
            "score": 0.0
        }
        
        rules = self.seo_rules["meta_description"]
        score_components = []
        
        # Length check
        if rules["min_length"] <= len(meta_desc) <= rules["max_length"]:
            score_components.append(1.0)
        elif len(meta_desc) < rules["min_length"]:
            analysis["issues"].append(f"Meta description too short ({len(meta_desc)} chars)")
            score_components.append(len(meta_desc) / rules["min_length"] if meta_desc else 0)
        else:
            analysis["issues"].append(f"Meta description too long ({len(meta_desc)} chars)")
            score_components.append(rules["max_length"] / len(meta_desc))
        
        # Keyword presence
        if meta_desc:
            meta_lower = meta_desc.lower()
            keyword_found = any(kw.lower() in meta_lower for kw in target_keywords)
            if keyword_found:
                score_components.append(1.0)
            else:
                analysis["issues"].append("No target keywords in meta description")
                score_components.append(0.3)
        else:
            analysis["issues"].append("Missing meta description")
            score_components.append(0.0)
        
        # Call to action check
        cta_words = ["buy", "shop", "discover", "explore", "learn", "get", "find", "order", "purchase"]
        has_cta = any(word in meta_desc.lower() for word in cta_words)
        if has_cta:
            score_components.append(1.0)
            analysis["has_call_to_action"] = True
        else:
            analysis["issues"].append("Consider adding call-to-action")
            score_components.append(0.7)
            analysis["has_call_to_action"] = False
        
        # Compelling language check
        compelling_words = ["amazing", "best", "premium", "quality", "exclusive", "limited", "free", "save"]
        compelling_score = sum(1 for word in compelling_words if word in meta_desc.lower()) / len(compelling_words)
        score_components.append(min(compelling_score * 2, 1.0))
        
        analysis["score"] = sum(score_components) / len(score_components)
        
        return analysis
    
    async def _analyze_content_optimization(
        self,
        content: Dict[str, Any],
        target_keywords: List[str]
    ) -> Dict[str, Any]:
        """
        Analyze content optimization for SEO
        """
        text_content = content.get("content", "")
        
        analysis = {
            "word_count": len(text_content.split()),
            "paragraph_count": len([p for p in text_content.split('\n\n') if p.strip()]),
            "issues": [],
            "score": 0.0
        }
        
        rules = self.seo_rules["content_optimization"]["content_structure"]
        score_components = []
        
        word_count = analysis["word_count"]
        
        # Word count check
        if rules["min_words"] <= word_count <= rules["max_words"]:
            score_components.append(1.0)
        elif word_count < rules["min_words"]:
            analysis["issues"].append(f"Content too short ({word_count} words, min {rules['min_words']})")
            score_components.append(word_count / rules["min_words"])
        else:
            analysis["issues"].append(f"Content may be too long ({word_count} words)")
            score_components.append(0.9)  # Slight penalty for very long content
        
        # Content structure analysis
        has_headings = bool(re.search(r'#{1,6}\s+', text_content) or 
                           re.search(r'<h[1-6]>', text_content))
        if has_headings:
            score_components.append(1.0)
        else:
            analysis["issues"].append("Consider adding headings for better structure")
            score_components.append(0.7)
        
        # Bullet points check
        has_bullets = bool(re.search(r'[•\-\*]\s+', text_content) or 
                          re.search(r'<li>', text_content))
        if has_bullets:
            score_components.append(1.0)
            analysis["has_bullet_points"] = True
        else:
            analysis["issues"].append("Consider using bullet points for key features")
            score_components.append(0.8)
            analysis["has_bullet_points"] = False
        
        # Paragraph length analysis
        paragraphs = [p.strip() for p in text_content.split('\n\n') if p.strip()]
        if paragraphs:
            avg_paragraph_length = sum(len(p.split()) for p in paragraphs) / len(paragraphs)
            if avg_paragraph_length <= 50:  # Good paragraph length
                score_components.append(1.0)
            else:
                analysis["issues"].append("Paragraphs may be too long for readability")
                score_components.append(0.8)
        else:
            score_components.append(0.5)
        
        analysis["score"] = sum(score_components) / len(score_components)
        analysis["structure_score"] = {
            "has_headings": has_headings,
            "has_bullets": has_bullets,
            "average_paragraph_length": avg_paragraph_length if paragraphs else 0
        }
        
        return analysis
    
    async def _analyze_keyword_density(
        self,
        content: Dict[str, Any],
        target_keywords: List[str]
    ) -> Dict[str, Any]:
        """
        Analyze keyword density and distribution
        """
        text_content = content.get("content", "")
        title = content.get("title", "")
        
        analysis = {
            "density_scores": {},
            "prominence_scores": {},
            "issues": [],
            "score": 0.0
        }
        
        rules = self.seo_rules["content_optimization"]["keyword_density"]
        prominence_rules = self.seo_rules["content_optimization"]["keyword_prominence"]
        
        word_count = len(text_content.split())
        if word_count == 0:
            analysis["issues"].append("No content to analyze")
            return analysis
        
        keyword_scores = []
        
        for keyword in target_keywords:
            keyword_lower = keyword.lower()
            
            # Count occurrences in different sections
            title_count = title.lower().count(keyword_lower)
            content_count = text_content.lower().count(keyword_lower)
            
            # Calculate density
            density = content_count / word_count
            analysis["density_scores"][keyword] = density
            
            # Calculate prominence score
            prominence_score = 0.0
            
            # Title prominence
            if title_count > 0:
                prominence_score += prominence_rules["title_weight"]
            
            # First paragraph prominence
            first_paragraph = text_content.split('\n')[0] if text_content else ""
            if keyword_lower in first_paragraph.lower():
                prominence_score += prominence_rules["first_paragraph_weight"]
            
            # Heading prominence
            if re.search(rf'#{1,6}\s+[^#]*{re.escape(keyword_lower)}', text_content, re.IGNORECASE):
                prominence_score += prominence_rules["headings_weight"]
            
            # Body prominence (normalized by content length)
            body_prominence = (content_count / max(word_count, 1)) * prominence_rules["body_weight"]
            prominence_score += body_prominence
            
            analysis["prominence_scores"][keyword] = prominence_score
            
            # Score the keyword usage
            keyword_score = 0.0
            
            # Density scoring
            if rules["min"] <= density <= rules["max"]:
                if abs(density - rules["target"]) <= 0.005:  # Within 0.5% of target
                    keyword_score += 1.0
                else:
                    keyword_score += 0.8
            elif density < rules["min"]:
                keyword_score += density / rules["min"]
                analysis["issues"].append(f"Low keyword density for '{keyword}': {density:.3f}")
            else:
                keyword_score += rules["max"] / density
                analysis["issues"].append(f"High keyword density for '{keyword}': {density:.3f}")
            
            # Prominence scoring
            max_prominence = prominence_rules["title_weight"] + prominence_rules["first_paragraph_weight"] + prominence_rules["headings_weight"] + prominence_rules["body_weight"]
            prominence_normalized = min(prominence_score / max_prominence, 1.0)
            keyword_score = (keyword_score + prominence_normalized) / 2
            
            keyword_scores.append(keyword_score)
        
        analysis["score"] = sum(keyword_scores) / len(keyword_scores) if keyword_scores else 0.0
        
        return analysis
    
    async def _check_technical_seo(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check technical SEO aspects
        """
        analysis = {
            "issues": [],
            "recommendations": [],
            "score": 0.0
        }
        
        score_components = []
        
        # Meta title check
        if content.get("meta_title"):
            score_components.append(1.0)
        else:
            analysis["issues"].append("Missing meta title")
            score_components.append(0.0)
        
        # Meta description check
        if content.get("meta_description"):
            score_components.append(1.0)
        else:
            analysis["issues"].append("Missing meta description")
            score_components.append(0.0)
        
        # Keywords check
        if content.get("keywords"):
            score_components.append(1.0)
        else:
            analysis["issues"].append("No target keywords defined")
            score_components.append(0.5)
        
        # Content uniqueness (basic check)
        text_content = content.get("content", "")
        if len(text_content) > 50:
            # Check for duplicate phrases (simplified)
            sentences = text_content.split('.')
            unique_sentences = set(s.strip().lower() for s in sentences if len(s.strip()) > 10)
            uniqueness_ratio = len(unique_sentences) / len(sentences) if sentences else 0
            
            if uniqueness_ratio > 0.8:
                score_components.append(1.0)
            else:
                analysis["issues"].append("Content may contain duplicate text")
                score_components.append(uniqueness_ratio)
        else:
            score_components.append(0.5)
        
        # Technical recommendations
        analysis["recommendations"].extend([
            "Implement structured data markup",
            "Optimize URL structure",
            "Add alt text for images",
            "Consider internal linking opportunities"
        ])
        
        analysis["score"] = sum(score_components) / len(score_components)
        
        return analysis
    
    async def _calculate_overall_seo_score(self, analyses: Dict[str, Dict]) -> float:
        """
        Calculate overall SEO score from individual analyses
        """
        weights = {
            "title": 0.25,
            "meta": 0.20,
            "content": 0.25,
            "keywords": 0.20,
            "technical": 0.10
        }
        
        weighted_score = 0.0
        total_weight = 0.0
        
        for component, analysis in analyses.items():
            if component in weights and isinstance(analysis, dict):
                score = analysis.get("score", 0.0)
                weight = weights[component]
                weighted_score += score * weight
                total_weight += weight
        
        return weighted_score / total_weight if total_weight > 0 else 0.0
    
    async def _generate_seo_recommendations(
        self,
        analyses: Dict[str, Dict],
        target_keywords: List[str]
    ) -> List[str]:
        """
        Generate SEO improvement recommendations
        """
        recommendations = []
        
        # Title recommendations
        title_analysis = analyses.get("title", {})
        if title_analysis.get("score", 0) < 0.8:
            recommendations.append("Optimize title for length and keyword inclusion")
        
        # Meta description recommendations
        meta_analysis = analyses.get("meta", {})
        if meta_analysis.get("score", 0) < 0.8:
            recommendations.append("Improve meta description with keywords and call-to-action")
        
        # Content recommendations
        content_analysis = analyses.get("content", {})
        if content_analysis.get("score", 0) < 0.8:
            issues = content_analysis.get("issues", [])
            if "too short" in " ".join(issues).lower():
                recommendations.append("Expand content with more detailed information")
            if "headings" in " ".join(issues).lower():
                recommendations.append("Add headings to improve content structure")
            if "bullet" in " ".join(issues).lower():
                recommendations.append("Use bullet points for better readability")
        
        # Keyword recommendations
        keyword_analysis = analyses.get("keywords", {})
        if keyword_analysis.get("score", 0) < 0.7:
            recommendations.append("Optimize keyword density and distribution")
            recommendations.append("Include target keywords in title and headings")
        
        # Technical recommendations
        technical_analysis = analyses.get("technical", {})
        technical_issues = technical_analysis.get("issues", [])
        if technical_issues:
            recommendations.extend([
                "Fix technical SEO issues",
                "Add missing meta tags",
                "Implement structured data markup"
            ])
        
        # General recommendations based on score
        overall_scores = [a.get("score", 0) for a in analyses.values() if isinstance(a, dict)]
        avg_score = sum(overall_scores) / len(overall_scores) if overall_scores else 0
        
        if avg_score < 0.6:
            recommendations.append("Consider comprehensive SEO audit and optimization")
        elif avg_score < 0.8:
            recommendations.append("Focus on top SEO improvement opportunities")
        
        return list(set(recommendations))  # Remove duplicates
    
    async def _identify_seo_opportunities(
        self,
        content: Dict[str, Any],
        target_keywords: List[str]
    ) -> List[str]:
        """
        Identify SEO opportunities for improvement
        """
        opportunities = []
        
        text_content = content.get("content", "")
        
        # Long-tail keyword opportunities
        if target_keywords:
            opportunities.append("Develop long-tail keyword variations")
            opportunities.append("Create topic clusters around main keywords")
        
        # Content expansion opportunities
        word_count = len(text_content.split())
        if word_count < 200:
            opportunities.append("Expand content with detailed specifications")
            opportunities.append("Add customer use cases and benefits")
        
        # Semantic SEO opportunities
        opportunities.extend([
            "Include related keywords and synonyms",
            "Add FAQ section for voice search optimization",
            "Implement local SEO if applicable",
            "Optimize for featured snippets"
        ])
        
        # Technical opportunities
        opportunities.extend([
            "Implement breadcrumb navigation",
            "Optimize page loading speed",
            "Add social sharing meta tags",
            "Consider AMP implementation"
        ])
        
        return opportunities
    
    async def optimize_content_for_seo(
        self,
        content: str,
        target_keywords: List[str],
        optimization_level: str = "moderate"
    ) -> Dict[str, Any]:
        """
        Optimize content for SEO using AI
        """
        try:
            if not self.openai_client:
                raise ValueError("OpenAI client not available for content optimization")
            
            # Analyze current content
            current_analysis = await self._analyze_keyword_density(
                {"content": content, "title": "", "meta_description": ""},
                target_keywords
            )
            
            # Create optimization prompt
            optimization_prompt = await self._create_optimization_prompt(
                content, target_keywords, optimization_level, current_analysis
            )
            
            # Generate optimized content
            response = await self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": self._create_seo_system_prompt()},
                    {"role": "user", "content": optimization_prompt}
                ],
                max_tokens=800,
                temperature=0.3
            )
            
            optimized_content = response.choices[0].message.content.strip()
            
            # Analyze optimized content
            optimized_analysis = await self._analyze_keyword_density(
                {"content": optimized_content, "title": "", "meta_description": ""},
                target_keywords
            )
            
            return {
                "original_content": content,
                "optimized_content": optimized_content,
                "improvements": {
                    "before_score": current_analysis.get("score", 0.0),
                    "after_score": optimized_analysis.get("score", 0.0),
                    "keyword_density_before": current_analysis.get("density_scores", {}),
                    "keyword_density_after": optimized_analysis.get("density_scores", {}),
                    "optimization_level": optimization_level
                },
                "recommendations": [
                    "Review optimized content for accuracy",
                    "Test with target audience",
                    "Monitor search performance after implementation"
                ]
            }
            
        except Exception as e:
            logger.error(f"SEO content optimization failed: {e}")
            raise
    
    def _create_seo_system_prompt(self) -> str:
        """
        Create system prompt for SEO optimization
        """
        return """
        You are an expert SEO content optimizer specializing in e-commerce product descriptions.
        
        Your expertise:
        - Natural keyword integration without stuffing
        - Maintaining readability while optimizing for search
        - Creating compelling content that converts
        - Following SEO best practices
        - Preserving original meaning and accuracy
        
        Guidelines:
        - Keep keyword density between 1-3%
        - Use keywords naturally in context
        - Maintain original tone and style
        - Ensure content remains factual and helpful
        - Don't sacrifice readability for SEO
        - Include related keywords and synonyms
        """
    
    async def _create_optimization_prompt(
        self,
        content: str,
        target_keywords: List[str],
        optimization_level: str,
        current_analysis: Dict[str, Any]
    ) -> str:
        """
        Create optimization prompt for AI
        """
        return f"""
        Optimize this product description for SEO while maintaining its quality and readability.
        
        Current content:
        {content}
        
        Target keywords: {', '.join(target_keywords)}
        Optimization level: {optimization_level}
        
        Current keyword analysis:
        - Keyword density scores: {current_analysis.get('density_scores', {})}
        - Current issues: {current_analysis.get('issues', [])}
        
        Optimization instructions:
        1. Naturally integrate target keywords
        2. Improve keyword prominence (use in headings, first paragraph)
        3. Add related keywords and synonyms
        4. Maintain original meaning and accuracy
        5. Keep the content engaging and readable
        6. {"Minimal changes" if optimization_level == "light" else "Moderate optimization" if optimization_level == "moderate" else "Comprehensive optimization"}
        
        Return only the optimized content.
        """
    
    async def generate_meta_tags(
        self,
        product_name: str,
        content: str,
        target_keywords: List[str],
        brand: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Generate optimized meta tags
        """
        try:
            # Generate meta title
            meta_title = await self._generate_meta_title(
                product_name, target_keywords, brand
            )
            
            # Generate meta description
            meta_description = await self._generate_meta_description(
                content, target_keywords
            )
            
            # Generate keywords meta tag
            keywords_tag = ", ".join(target_keywords[:10])  # Limit to 10 keywords
            
            return {
                "title": meta_title,
                "description": meta_description,
                "keywords": keywords_tag,
                "og:title": meta_title,
                "og:description": meta_description,
                "twitter:title": meta_title,
                "twitter:description": meta_description
            }
            
        except Exception as e:
            logger.error(f"Meta tag generation failed: {e}")
            return {
                "title": product_name,
                "description": content[:160],
                "keywords": ", ".join(target_keywords)
            }
    
    async def _generate_meta_title(
        self,
        product_name: str,
        target_keywords: List[str],
        brand: Optional[str] = None
    ) -> str:
        """
        Generate optimized meta title
        """
        # Use primary keyword if available
        primary_keyword = target_keywords[0] if target_keywords else ""
        
        # Template-based generation
        if brand and brand != "Unknown":
            if primary_keyword and primary_keyword.lower() != product_name.lower():
                title = f"{product_name} - {primary_keyword} | {brand}"
            else:
                title = f"{product_name} | {brand}"
        else:
            if primary_keyword and primary_keyword.lower() != product_name.lower():
                title = f"{product_name} - {primary_keyword}"
            else:
                title = product_name
        
        # Ensure within length limits
        if len(title) > 60:
            if brand:
                title = f"{product_name} | {brand}"
            else:
                title = product_name
            
            if len(title) > 60:
                title = title[:57] + "..."
        
        return title
    
    async def _generate_meta_description(
        self,
        content: str,
        target_keywords: List[str]
    ) -> str:
        """
        Generate optimized meta description
        """
        # Extract first sentence or paragraph
        sentences = content.split('.')
        first_sentence = sentences[0].strip() if sentences else content[:100]
        
        # Ensure primary keyword is included
        primary_keyword = target_keywords[0] if target_keywords else ""
        
        if primary_keyword and primary_keyword.lower() not in first_sentence.lower():
            meta_desc = f"{first_sentence}. {primary_keyword} with premium quality."
        else:
            meta_desc = first_sentence
        
        # Add call to action
        if not any(word in meta_desc.lower() for word in ["buy", "shop", "get", "order"]):
            meta_desc += " Shop now for the best deals."
        
        # Ensure within length limits
        if len(meta_desc) > 160:
            meta_desc = meta_desc[:157] + "..."
        
        return meta_desc
    
    async def track_keyword_rankings(
        self,
        keywords: List[str],
        domain: str = "example.com"
    ) -> Dict[str, KeywordAnalysis]:
        """
        Track keyword rankings (placeholder for real SEO API integration)
        """
        # This would integrate with real SEO APIs like SEMrush, Ahrefs, or custom ranking checker
        rankings = {}
        
        for keyword in keywords:
            # Simulated ranking data - replace with real API calls
            rankings[keyword] = KeywordAnalysis(
                keyword=keyword,
                search_volume=random.randint(100, 10000),
                competition_level=random.choice(["low", "medium", "high"]),
                difficulty_score=random.uniform(0.1, 1.0),
                current_rank=random.randint(1, 100) if random.random() > 0.3 else None,
                density=0.0,  # Would be calculated from actual content
                prominence_score=random.uniform(0.0, 1.0),
                relevance_score=random.uniform(0.7, 1.0)
            )
        
        return rankings
    
    async def generate_schema_markup(
        self,
        product_name: str,
        product_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate structured data markup for products
        """
        schema = {
            "@context": "https://schema.org",
            "@type": "Product",
            "name": product_name,
            "description": product_data.get("description", ""),
        }
        
        # Add optional fields if available
        if product_data.get("brand"):
            schema["brand"] = {
                "@type": "Brand",
                "name": product_data["brand"]
            }
        
        if product_data.get("price"):
            schema["offers"] = {
                "@type": "Offer",
                "price": str(product_data["price"]),
                "priceCurrency": "USD",
                "availability": "https://schema.org/InStock"
            }
        
        if product_data.get("rating"):
            schema["aggregateRating"] = {
                "@type": "AggregateRating",
                "ratingValue": str(product_data["rating"]),
                "reviewCount": str(product_data.get("review_count", 1))
            }
        
        if product_data.get("image_url"):
            schema["image"] = product_data["image_url"]
        
        if product_data.get("category"):
            schema["category"] = product_data["category"]
        
        return schema