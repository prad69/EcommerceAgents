import asyncio
from typing import Dict, List, Any, Optional, Tuple
from loguru import logger
import json
import re
from datetime import datetime
from dataclasses import dataclass
import random

from src.core.config import settings
from src.core.database import get_db
from src.models.product_description import (
    DescriptionTemplate, ProductDescription, DescriptionType, 
    DescriptionStatus, GenerationMethod, BrandGuideline
)
from src.services.product_analyzer import ProductAnalyzerService, ProductSpecification


@dataclass
class GenerationRequest:
    """Description generation request parameters"""
    product_id: str
    description_types: List[DescriptionType]
    target_keywords: List[str] = None
    target_word_count: Optional[int] = None
    tone: str = "professional"
    brand_guidelines_id: Optional[str] = None
    template_id: Optional[str] = None
    custom_prompt: Optional[str] = None
    include_seo: bool = True
    include_specifications: bool = True
    target_audience: Optional[str] = None


class DescriptionGeneratorService:
    """
    Automated product description generation service
    """
    
    def __init__(self):
        self.product_analyzer = ProductAnalyzerService()
        self.openai_client = None
        self.default_templates = {}
        self._setup_models()
        self._setup_default_templates()
    
    def _setup_models(self):
        """
        Initialize AI models and services
        """
        # Setup OpenAI
        if settings.openai_api_key:
            try:
                import openai
                self.openai_client = openai.AsyncOpenAI(
                    api_key=settings.openai_api_key
                )
                logger.info("OpenAI description generation client initialized")
            except ImportError:
                logger.warning("OpenAI library not available")
    
    def _setup_default_templates(self):
        """
        Setup default description templates
        """
        self.default_templates = {
            DescriptionType.SHORT: {
                "professional": "{name} - {primary_benefit}. {key_feature}. {price_positioning}",
                "casual": "Meet the {name}! {primary_benefit} with {key_feature}. {call_to_action}",
                "luxury": "Introducing {name}: {premium_positioning}. {exclusivity}. {luxury_benefit}.",
                "technical": "{name} features {technical_spec}. {performance_benefit}. {compatibility}."
            },
            
            DescriptionType.MEDIUM: {
                "professional": """
                {name} delivers {primary_benefit} for {target_audience}. 
                
                Key features include {feature_list}. {material_quality} construction ensures {durability_benefit}.
                
                {use_case_description} Perfect for {primary_use_cases}.
                """,
                "casual": """
                Looking for {solution_statement}? {name} has got you covered!
                
                What makes it awesome:
                • {feature_1}
                • {feature_2} 
                • {feature_3}
                
                {benefit_statement} {social_proof}
                """,
                "luxury": """
                Experience exceptional {category} with {name}. Crafted for discerning individuals who appreciate {quality_aspect}.
                
                Distinguished features:
                - {premium_feature_1}
                - {premium_feature_2}
                - {premium_feature_3}
                
                {exclusivity_statement} {heritage_statement}
                """,
                "technical": """
                {name} - {technical_summary}
                
                Specifications:
                • {spec_1}
                • {spec_2}
                • {spec_3}
                
                {performance_data} {compatibility_info} {warranty_info}
                """
            },
            
            DescriptionType.LONG: {
                "professional": """
                {introduction_paragraph}
                
                Key Features:
                {detailed_feature_list}
                
                Benefits:
                {benefit_paragraphs}
                
                Specifications:
                {specification_details}
                
                Applications:
                {use_case_details}
                
                {conclusion_paragraph}
                """,
                "storytelling": """
                {story_opening}
                
                {problem_statement}
                
                {solution_introduction}
                
                {feature_story}
                
                {benefit_narrative}
                
                {user_journey}
                
                {conclusion_cta}
                """
            },
            
            DescriptionType.BULLETS: {
                "features": "• {feature_1}\n• {feature_2}\n• {feature_3}\n• {feature_4}\n• {feature_5}",
                "benefits": "✓ {benefit_1}\n✓ {benefit_2}\n✓ {benefit_3}\n✓ {benefit_4}",
                "specifications": "▪ {spec_1}\n▪ {spec_2}\n▪ {spec_3}\n▪ {spec_4}"
            },
            
            DescriptionType.SEO: {
                "optimized": """
                {seo_title} | {brand_name}
                
                {keyword_rich_introduction}
                
                {seo_feature_section}
                
                {seo_benefit_section}
                
                {seo_conclusion}
                
                {structured_data}
                """
            }
        }
    
    async def generate_descriptions(
        self,
        request: GenerationRequest
    ) -> Dict[str, Any]:
        """
        Generate product descriptions based on request parameters
        """
        try:
            logger.info(f"Generating descriptions for product {request.product_id}")
            
            # Analyze product specifications
            product_spec = await self.product_analyzer.analyze_product_specifications(
                request.product_id
            )
            
            # Get brand guidelines if specified
            brand_guidelines = None
            if request.brand_guidelines_id:
                brand_guidelines = await self._get_brand_guidelines(request.brand_guidelines_id)
            
            # Generate descriptions for each requested type
            generated_descriptions = {}
            
            for desc_type in request.description_types:
                logger.debug(f"Generating {desc_type.value} description")
                
                description_result = await self._generate_single_description(
                    desc_type=desc_type,
                    product_spec=product_spec,
                    request=request,
                    brand_guidelines=brand_guidelines
                )
                
                generated_descriptions[desc_type.value] = description_result
            
            # Calculate overall generation metrics
            total_time = sum(
                desc.get("generation_time_ms", 0) 
                for desc in generated_descriptions.values()
            )
            
            avg_quality = sum(
                desc.get("quality_score", 0.0) 
                for desc in generated_descriptions.values()
            ) / len(generated_descriptions) if generated_descriptions else 0.0
            
            result = {
                "product_id": request.product_id,
                "generation_id": f"gen_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                "descriptions": generated_descriptions,
                "metadata": {
                    "product_analysis": {
                        "category": product_spec.category,
                        "features_count": len(product_spec.features),
                        "benefits_count": len(product_spec.benefits),
                        "target_audience": product_spec.target_audience
                    },
                    "generation_stats": {
                        "total_descriptions": len(generated_descriptions),
                        "total_time_ms": total_time,
                        "average_quality_score": avg_quality,
                        "generation_method": "hybrid",
                        "brand_guidelines_applied": brand_guidelines is not None
                    }
                },
                "recommendations": await self._generate_content_recommendations(
                    product_spec, generated_descriptions
                ),
                "generated_at": datetime.utcnow().isoformat()
            }
            
            logger.info(f"Description generation completed for product {request.product_id}")
            return result
            
        except Exception as e:
            logger.error(f"Description generation failed: {e}")
            raise
    
    async def _generate_single_description(
        self,
        desc_type: DescriptionType,
        product_spec: ProductSpecification,
        request: GenerationRequest,
        brand_guidelines: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate a single description of specified type
        """
        start_time = datetime.utcnow()
        
        try:
            # Choose generation method
            generation_method = await self._select_generation_method(
                desc_type, product_spec, request
            )
            
            description_content = ""
            method_used = GenerationMethod.TEMPLATE
            
            # Generate using selected method
            if generation_method == "ai" and self.openai_client:
                description_content = await self._generate_with_ai(
                    desc_type, product_spec, request, brand_guidelines
                )
                method_used = GenerationMethod.AI_GENERATED
                
            elif generation_method == "template" or not description_content:
                description_content = await self._generate_with_template(
                    desc_type, product_spec, request, brand_guidelines
                )
                method_used = GenerationMethod.TEMPLATE
                if generation_method == "ai":
                    method_used = GenerationMethod.HYBRID
            
            # Post-process the description
            processed_content = await self._post_process_description(
                description_content, desc_type, product_spec, request
            )
            
            # Calculate quality scores
            quality_metrics = await self._calculate_quality_scores(
                processed_content, desc_type, product_spec, request
            )
            
            # Generate SEO metadata if needed
            seo_metadata = {}
            if request.include_seo:
                seo_metadata = await self._generate_seo_metadata(
                    processed_content, product_spec, request
                )
            
            # Calculate generation time
            generation_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return {
                "content": processed_content,
                "description_type": desc_type.value,
                "generation_method": method_used.value,
                "quality_metrics": quality_metrics,
                "seo_metadata": seo_metadata,
                "word_count": len(processed_content.split()),
                "character_count": len(processed_content),
                "generation_time_ms": int(generation_time),
                "template_used": request.template_id,
                "brand_guidelines_applied": brand_guidelines is not None,
                "quality_score": quality_metrics.get("overall_score", 0.0)
            }
            
        except Exception as e:
            logger.error(f"Single description generation failed: {e}")
            return {
                "content": f"Error generating {desc_type.value} description",
                "error": str(e),
                "description_type": desc_type.value,
                "quality_score": 0.0
            }
    
    async def _select_generation_method(
        self,
        desc_type: DescriptionType,
        product_spec: ProductSpecification,
        request: GenerationRequest
    ) -> str:
        """
        Select the best generation method based on context
        """
        # AI generation for complex requirements
        if (request.custom_prompt or 
            desc_type in [DescriptionType.LONG, DescriptionType.SEO] or
            len(product_spec.features) > 10 or
            request.tone not in ["professional", "casual"]):
            return "ai"
        
        # Template generation for simple, standard descriptions
        return "template"
    
    async def _generate_with_ai(
        self,
        desc_type: DescriptionType,
        product_spec: ProductSpecification,
        request: GenerationRequest,
        brand_guidelines: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate description using AI (OpenAI)
        """
        try:
            # Build context for AI generation
            context = await self._build_ai_context(
                product_spec, request, brand_guidelines
            )
            
            # Create generation prompt
            prompt = await self._create_ai_prompt(
                desc_type, product_spec, request, context
            )
            
            # Generate with OpenAI
            response = await self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": await self._create_system_prompt(desc_type, request)},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self._get_max_tokens_for_type(desc_type),
                temperature=0.7
            )
            
            generated_content = response.choices[0].message.content.strip()
            return generated_content
            
        except Exception as e:
            logger.error(f"AI generation failed: {e}")
            # Fallback to template generation
            return await self._generate_with_template(
                desc_type, product_spec, request, brand_guidelines
            )
    
    async def _generate_with_template(
        self,
        desc_type: DescriptionType,
        product_spec: ProductSpecification,
        request: GenerationRequest,
        brand_guidelines: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate description using templates
        """
        try:
            # Get appropriate template
            template = await self._get_template(desc_type, request, brand_guidelines)
            
            # Build template variables
            template_vars = await self._build_template_variables(
                product_spec, request, brand_guidelines
            )
            
            # Apply template
            try:
                description = template.format(**template_vars)
            except KeyError as e:
                # Handle missing template variables
                logger.warning(f"Missing template variable {e}, using fallback")
                description = await self._generate_fallback_description(
                    desc_type, product_spec
                )
            
            # Clean up the description
            description = re.sub(r'\n\s*\n', '\n\n', description)  # Normalize newlines
            description = re.sub(r'\s+', ' ', description)  # Normalize spaces
            description = description.strip()
            
            return description
            
        except Exception as e:
            logger.error(f"Template generation failed: {e}")
            return await self._generate_fallback_description(desc_type, product_spec)
    
    async def _build_ai_context(
        self,
        product_spec: ProductSpecification,
        request: GenerationRequest,
        brand_guidelines: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Build context for AI generation
        """
        context = {
            "product": {
                "name": product_spec.name,
                "category": product_spec.category,
                "brand": product_spec.brand,
                "price": product_spec.price,
                "features": product_spec.features[:8],  # Limit for token count
                "benefits": product_spec.benefits[:5],
                "target_audience": product_spec.target_audience,
                "use_cases": product_spec.use_cases[:3]
            },
            "requirements": {
                "tone": request.tone,
                "target_keywords": request.target_keywords or [],
                "word_count": request.target_word_count,
                "include_seo": request.include_seo,
                "target_audience": request.target_audience
            }
        }
        
        if brand_guidelines:
            context["brand"] = {
                "tone_of_voice": brand_guidelines.get("tone_of_voice", {}),
                "preferred_words": brand_guidelines.get("preferred_words", []),
                "prohibited_words": brand_guidelines.get("prohibited_words", [])
            }
        
        return context
    
    async def _create_ai_prompt(
        self,
        desc_type: DescriptionType,
        product_spec: ProductSpecification,
        request: GenerationRequest,
        context: Dict[str, Any]
    ) -> str:
        """
        Create AI generation prompt
        """
        if request.custom_prompt:
            return f"""
            {request.custom_prompt}
            
            Product Information:
            {json.dumps(context['product'], indent=2)}
            
            Requirements:
            {json.dumps(context['requirements'], indent=2)}
            """
        
        # Type-specific prompts
        type_prompts = {
            DescriptionType.SHORT: f"""
            Create a concise, compelling product description for {product_spec.name}.
            
            Requirements:
            - 1-2 sentences maximum
            - Highlight the main benefit
            - Include primary feature
            - Use {request.tone} tone
            - Target: {request.target_audience or 'general audience'}
            
            Product details: {json.dumps(context['product'], indent=2)}
            """,
            
            DescriptionType.MEDIUM: f"""
            Write a comprehensive product description for {product_spec.name}.
            
            Structure:
            1. Opening hook (main benefit)
            2. Key features (3-4 bullet points)
            3. Benefits explanation
            4. Call to action
            
            Tone: {request.tone}
            Target audience: {request.target_audience or 'general consumers'}
            Word count: {request.target_word_count or '100-150'} words
            
            Product information: {json.dumps(context['product'], indent=2)}
            """,
            
            DescriptionType.LONG: f"""
            Create a detailed, comprehensive product description for {product_spec.name}.
            
            Include sections for:
            1. Product introduction and positioning
            2. Detailed feature breakdown
            3. Benefits and value proposition
            4. Technical specifications
            5. Use cases and applications
            6. Compelling conclusion
            
            Requirements:
            - Professional, informative tone
            - 300-500 words
            - Include technical details
            - Appeal to {request.target_audience or 'informed buyers'}
            
            Product data: {json.dumps(context['product'], indent=2)}
            """,
            
            DescriptionType.SEO: f"""
            Write an SEO-optimized product description for {product_spec.name}.
            
            SEO Requirements:
            - Include target keywords naturally: {', '.join(request.target_keywords or [])}
            - Write compelling meta title (max 60 chars)
            - Create meta description (max 160 chars)
            - Use header structure (H1, H2, H3)
            - Include schema markup suggestions
            
            Content should be informative and conversion-focused.
            Target keywords: {request.target_keywords}
            
            Product details: {json.dumps(context['product'], indent=2)}
            """
        }
        
        return type_prompts.get(desc_type, type_prompts[DescriptionType.MEDIUM])
    
    async def _create_system_prompt(
        self,
        desc_type: DescriptionType,
        request: GenerationRequest
    ) -> str:
        """
        Create system prompt for AI generation
        """
        return f"""
        You are an expert product copywriter specializing in e-commerce descriptions.
        
        Your expertise:
        - Converting features into benefits
        - Writing compelling, conversion-focused copy
        - Optimizing for search engines when required
        - Adapting tone and style to target audience
        - Following brand guidelines and best practices
        
        Current task: Generate a {desc_type.value} product description
        Tone requirement: {request.tone}
        
        Guidelines:
        - Focus on customer benefits, not just features
        - Use action-oriented language
        - Be specific and credible
        - Avoid marketing fluff
        - Write for the target audience
        - Ensure accuracy and relevance
        """
    
    def _get_max_tokens_for_type(self, desc_type: DescriptionType) -> int:
        """
        Get appropriate token limit for description type
        """
        token_limits = {
            DescriptionType.SHORT: 100,
            DescriptionType.MEDIUM: 250,
            DescriptionType.LONG: 500,
            DescriptionType.BULLETS: 150,
            DescriptionType.SEO: 400,
            DescriptionType.SOCIAL: 80,
            DescriptionType.TECHNICAL: 300
        }
        
        return token_limits.get(desc_type, 250)
    
    async def _get_template(
        self,
        desc_type: DescriptionType,
        request: GenerationRequest,
        brand_guidelines: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Get appropriate template for description generation
        """
        # Check for custom template
        if request.template_id:
            db = next(get_db())
            custom_template = db.query(DescriptionTemplate).filter(
                DescriptionTemplate.id == request.template_id
            ).first()
            db.close()
            
            if custom_template:
                return custom_template.template_text
        
        # Use default templates
        type_templates = self.default_templates.get(desc_type, {})
        
        # Select template based on tone
        tone = request.tone.lower()
        if tone in type_templates:
            return type_templates[tone]
        
        # Fallback to professional tone
        return type_templates.get("professional", type_templates.get(list(type_templates.keys())[0], ""))
    
    async def _build_template_variables(
        self,
        product_spec: ProductSpecification,
        request: GenerationRequest,
        brand_guidelines: Optional[Dict[str, Any]] = None
    ) -> Dict[str, str]:
        """
        Build variables for template replacement
        """
        # Basic variables
        variables = {
            "name": product_spec.name,
            "category": product_spec.category,
            "brand": product_spec.brand,
            "price": f"${product_spec.price:.2f}",
            "primary_benefit": product_spec.benefits[0] if product_spec.benefits else "quality and value",
            "key_feature": product_spec.features[0] if product_spec.features else "premium construction",
            "target_audience": request.target_audience or product_spec.target_audience[0] if product_spec.target_audience else "customers",
            "primary_use_cases": ", ".join(product_spec.use_cases[:2]) if product_spec.use_cases else "daily use"
        }
        
        # Feature variables
        for i, feature in enumerate(product_spec.features[:5], 1):
            variables[f"feature_{i}"] = feature
        
        # Benefit variables
        for i, benefit in enumerate(product_spec.benefits[:4], 1):
            variables[f"benefit_{i}"] = benefit
        
        # Specification variables
        for i, (spec_key, spec_value) in enumerate(product_spec.specifications.items(), 1):
            variables[f"spec_{i}"] = f"{spec_key}: {spec_value}"
            if i >= 4:  # Limit specs
                break
        
        # Dynamic content generation
        variables.update({
            "feature_list": self._format_feature_list(product_spec.features[:5]),
            "benefit_statement": self._create_benefit_statement(product_spec),
            "price_positioning": self._create_price_positioning(product_spec.price),
            "call_to_action": self._create_call_to_action(request.tone),
            "solution_statement": self._create_solution_statement(product_spec),
            "social_proof": self._create_social_proof(product_spec)
        })
        
        # Fill missing variables with defaults
        return self._fill_missing_variables(variables)
    
    def _format_feature_list(self, features: List[str]) -> str:
        """
        Format features as a readable list
        """
        if not features:
            return "premium quality features"
        
        if len(features) == 1:
            return features[0]
        elif len(features) == 2:
            return f"{features[0]} and {features[1]}"
        else:
            return f"{', '.join(features[:-1])}, and {features[-1]}"
    
    def _create_benefit_statement(self, product_spec: ProductSpecification) -> str:
        """
        Create a compelling benefit statement
        """
        if not product_spec.benefits:
            return "delivers exceptional value and performance"
        
        primary_benefit = product_spec.benefits[0]
        return f"providing {primary_benefit}"
    
    def _create_price_positioning(self, price: float) -> str:
        """
        Create price positioning statement
        """
        if price < 25:
            return "At an affordable price point"
        elif price < 100:
            return "Offering excellent value"
        elif price < 500:
            return "A premium investment"
        else:
            return "Luxury quality at its finest"
    
    def _create_call_to_action(self, tone: str) -> str:
        """
        Create appropriate call to action
        """
        ctas = {
            "professional": "Contact us for more information.",
            "casual": "Get yours today!",
            "luxury": "Experience excellence.",
            "technical": "Specifications available upon request."
        }
        
        return ctas.get(tone.lower(), "Learn more today.")
    
    def _create_solution_statement(self, product_spec: ProductSpecification) -> str:
        """
        Create a solution-focused statement
        """
        if product_spec.use_cases:
            return f"a reliable solution for {product_spec.use_cases[0]}"
        else:
            return f"the perfect {product_spec.category.lower()} solution"
    
    def _create_social_proof(self, product_spec: ProductSpecification) -> str:
        """
        Create social proof statement
        """
        if product_spec.brand and product_spec.brand != "Unknown":
            return f"Trusted by {product_spec.brand} customers worldwide."
        else:
            return "Join thousands of satisfied customers."
    
    def _fill_missing_variables(self, variables: Dict[str, str]) -> Dict[str, str]:
        """
        Fill in any missing template variables with sensible defaults
        """
        defaults = {
            "material_quality": "high-quality",
            "durability_benefit": "long-lasting performance",
            "use_case_description": "Designed for versatile use.",
            "performance_data": "Delivers reliable performance.",
            "compatibility_info": "Compatible with standard systems.",
            "warranty_info": "Backed by manufacturer warranty."
        }
        
        # Add defaults for missing variables
        for key, default_value in defaults.items():
            if key not in variables:
                variables[key] = default_value
        
        # Ensure all numbered variables exist
        for i in range(1, 6):
            if f"feature_{i}" not in variables:
                variables[f"feature_{i}"] = "quality construction"
            if f"benefit_{i}" not in variables:
                variables[f"benefit_{i}"] = "enhanced performance"
            if f"spec_{i}" not in variables:
                variables[f"spec_{i}"] = "premium specifications"
        
        return variables
    
    async def _generate_fallback_description(
        self,
        desc_type: DescriptionType,
        product_spec: ProductSpecification
    ) -> str:
        """
        Generate basic fallback description when other methods fail
        """
        fallbacks = {
            DescriptionType.SHORT: f"{product_spec.name} - Quality {product_spec.category.lower()} for everyday use. Features premium construction and reliable performance.",
            DescriptionType.MEDIUM: f"""
            {product_spec.name} is a high-quality {product_spec.category.lower()} designed for performance and reliability.
            
            Key features include premium construction, durable materials, and user-friendly design.
            Perfect for both personal and professional use.
            """,
            DescriptionType.LONG: f"""
            Discover the exceptional quality of {product_spec.name}, a premium {product_spec.category.lower()} 
            crafted for those who demand the best.
            
            This product combines innovative design with practical functionality, delivering outstanding 
            performance in every application. Built with attention to detail and quality materials.
            
            Whether for personal or professional use, {product_spec.name} provides the reliability 
            and performance you need.
            """,
            DescriptionType.BULLETS: f"""
            • Premium {product_spec.category.lower()} design
            • High-quality construction
            • Reliable performance
            • User-friendly operation
            • Excellent value
            """
        }
        
        return fallbacks.get(desc_type, f"{product_spec.name} - Quality {product_spec.category.lower()}.")
    
    async def _post_process_description(
        self,
        content: str,
        desc_type: DescriptionType,
        product_spec: ProductSpecification,
        request: GenerationRequest
    ) -> str:
        """
        Post-process generated description
        """
        # Clean up formatting
        content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)  # Remove excess newlines
        content = re.sub(r'^\s+|\s+$', '', content, flags=re.MULTILINE)  # Trim lines
        content = content.strip()
        
        # Apply word count limits if specified
        if request.target_word_count:
            words = content.split()
            if len(words) > request.target_word_count * 1.2:  # 20% tolerance
                content = ' '.join(words[:int(request.target_word_count * 1.1)])
                content += "..."
        
        # Ensure minimum length for certain types
        min_lengths = {
            DescriptionType.SHORT: 20,
            DescriptionType.MEDIUM: 50,
            DescriptionType.LONG: 100
        }
        
        min_length = min_lengths.get(desc_type, 0)
        if len(content) < min_length:
            # Pad with additional content
            content += f" Featuring {product_spec.category.lower()} excellence."
        
        return content
    
    async def _calculate_quality_scores(
        self,
        content: str,
        desc_type: DescriptionType,
        product_spec: ProductSpecification,
        request: GenerationRequest
    ) -> Dict[str, float]:
        """
        Calculate quality scores for generated description
        """
        scores = {}
        
        # Length appropriateness
        word_count = len(content.split())
        expected_lengths = {
            DescriptionType.SHORT: (15, 40),
            DescriptionType.MEDIUM: (50, 150),
            DescriptionType.LONG: (200, 500),
            DescriptionType.BULLETS: (20, 80)
        }
        
        expected_min, expected_max = expected_lengths.get(desc_type, (50, 150))
        if expected_min <= word_count <= expected_max:
            scores["length_score"] = 1.0
        elif word_count < expected_min:
            scores["length_score"] = word_count / expected_min
        else:
            scores["length_score"] = expected_max / word_count
        
        # Keyword inclusion
        if request.target_keywords:
            content_lower = content.lower()
            keyword_hits = sum(1 for kw in request.target_keywords if kw.lower() in content_lower)
            scores["keyword_score"] = keyword_hits / len(request.target_keywords)
        else:
            scores["keyword_score"] = 0.8  # Default when no keywords specified
        
        # Feature coverage
        content_lower = content.lower()
        feature_hits = sum(1 for feature in product_spec.features[:5] 
                          if any(word.lower() in content_lower for word in feature.split()))
        scores["feature_coverage"] = feature_hits / min(len(product_spec.features), 5) if product_spec.features else 0.5
        
        # Readability (simplified)
        sentences = content.count('.') + content.count('!') + content.count('?')
        avg_words_per_sentence = word_count / max(sentences, 1)
        if 10 <= avg_words_per_sentence <= 20:
            scores["readability_score"] = 1.0
        else:
            scores["readability_score"] = 0.8
        
        # Overall score
        scores["overall_score"] = sum(scores.values()) / len(scores)
        
        return scores
    
    async def _generate_seo_metadata(
        self,
        content: str,
        product_spec: ProductSpecification,
        request: GenerationRequest
    ) -> Dict[str, Any]:
        """
        Generate SEO metadata for the description
        """
        # Generate meta title
        meta_title = f"{product_spec.name} | {product_spec.category}"
        if product_spec.brand and product_spec.brand != "Unknown":
            meta_title += f" | {product_spec.brand}"
        
        # Truncate to 60 characters
        if len(meta_title) > 60:
            meta_title = meta_title[:57] + "..."
        
        # Generate meta description
        first_sentence = content.split('.')[0] if '.' in content else content[:100]
        meta_description = first_sentence.strip()
        if len(meta_description) > 160:
            meta_description = meta_description[:157] + "..."
        
        # Extract keywords from content
        content_words = re.findall(r'\b\w{4,}\b', content.lower())
        content_keywords = list(set(content_words))[:10]
        
        return {
            "meta_title": meta_title,
            "meta_description": meta_description,
            "keywords": content_keywords,
            "target_keywords": request.target_keywords or [],
            "structured_data_suggestions": {
                "type": "Product",
                "name": product_spec.name,
                "category": product_spec.category,
                "price": product_spec.price,
                "description": meta_description
            }
        }
    
    async def _generate_content_recommendations(
        self,
        product_spec: ProductSpecification,
        generated_descriptions: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate content improvement recommendations
        """
        recommendations = {
            "content_improvements": [],
            "seo_opportunities": [],
            "conversion_optimizations": []
        }
        
        # Content improvements
        if len(product_spec.features) < 3:
            recommendations["content_improvements"].append(
                "Consider adding more detailed product features"
            )
        
        if not product_spec.benefits:
            recommendations["content_improvements"].append(
                "Add customer benefits to improve conversion"
            )
        
        # SEO opportunities
        if not any("seo" in desc.get("description_type", "") for desc in generated_descriptions.values()):
            recommendations["seo_opportunities"].append(
                "Consider generating SEO-optimized description variant"
            )
        
        # Conversion optimizations
        content_text = " ".join(desc.get("content", "") for desc in generated_descriptions.values()).lower()
        
        if "guarantee" not in content_text and "warranty" not in content_text:
            recommendations["conversion_optimizations"].append(
                "Consider mentioning warranty or guarantee to build trust"
            )
        
        if not any(word in content_text for word in ["free shipping", "fast delivery", "quick"]):
            recommendations["conversion_optimizations"].append(
                "Consider highlighting shipping benefits"
            )
        
        return recommendations
    
    async def _get_brand_guidelines(self, guidelines_id: str) -> Optional[Dict[str, Any]]:
        """
        Get brand guidelines from database
        """
        try:
            db = next(get_db())
            guidelines = db.query(BrandGuideline).filter(
                BrandGuideline.id == guidelines_id,
                BrandGuideline.is_active == True
            ).first()
            db.close()
            
            if guidelines:
                return {
                    "brand_name": guidelines.brand_name,
                    "tone_of_voice": guidelines.tone_of_voice or {},
                    "writing_style": guidelines.writing_style or {},
                    "prohibited_words": guidelines.prohibited_words or [],
                    "preferred_words": guidelines.preferred_words or [],
                    "min_length": guidelines.min_description_length,
                    "max_length": guidelines.max_description_length,
                    "required_sections": guidelines.required_sections or [],
                    "disclaimers": guidelines.mandatory_disclaimers or []
                }
            
            return None
            
        except Exception as e:
            logger.warning(f"Failed to get brand guidelines: {e}")
            return None