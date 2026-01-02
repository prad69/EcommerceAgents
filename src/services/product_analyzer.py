import asyncio
from typing import Dict, List, Any, Optional, Tuple
from loguru import logger
import json
import re
from datetime import datetime
from dataclasses import dataclass, field

from src.core.database import get_db
from src.models.product import Product
from src.models.product_description import SEOKeyword, CompetitorAnalysis


@dataclass
class ProductSpecification:
    """Structured product specification data"""
    name: str
    category: str
    brand: str
    price: float
    features: List[str] = field(default_factory=list)
    specifications: Dict[str, Any] = field(default_factory=dict)
    dimensions: Dict[str, Any] = field(default_factory=dict)
    materials: List[str] = field(default_factory=list)
    colors: List[str] = field(default_factory=list)
    sizes: List[str] = field(default_factory=list)
    weight: Optional[str] = None
    warranty: Optional[str] = None
    certifications: List[str] = field(default_factory=list)
    target_audience: List[str] = field(default_factory=list)
    use_cases: List[str] = field(default_factory=list)
    benefits: List[str] = field(default_factory=list)
    unique_selling_points: List[str] = field(default_factory=list)
    technical_specs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CompetitiveInsight:
    """Competitive analysis insights"""
    competitor: str
    product_name: str
    price_comparison: str  # higher, lower, similar
    feature_gaps: List[str]
    positioning_advantages: List[str]
    content_quality_score: float
    market_position: str  # premium, mid-range, budget


class ProductAnalyzerService:
    """
    Analyzes product specifications and market positioning for description generation
    """
    
    def __init__(self):
        self.feature_keywords = self._setup_feature_keywords()
        self.category_analyzers = self._setup_category_analyzers()
        self.specification_patterns = self._setup_specification_patterns()
    
    def _setup_feature_keywords(self) -> Dict[str, List[str]]:
        """
        Setup keywords that indicate specific product features
        """
        return {
            "quality_indicators": [
                "premium", "high-quality", "professional", "durable", "robust",
                "heavy-duty", "commercial-grade", "industrial", "precision"
            ],
            "convenience_features": [
                "easy-to-use", "user-friendly", "convenient", "quick", "fast",
                "instant", "automatic", "wireless", "portable", "compact"
            ],
            "performance_features": [
                "efficient", "powerful", "high-performance", "advanced", "innovative",
                "cutting-edge", "state-of-the-art", "optimized", "enhanced"
            ],
            "safety_features": [
                "safe", "secure", "protection", "certified", "tested",
                "approved", "compliant", "non-toxic", "child-safe"
            ],
            "aesthetic_features": [
                "beautiful", "elegant", "stylish", "modern", "classic",
                "sleek", "attractive", "designer", "fashionable"
            ],
            "environmental_features": [
                "eco-friendly", "sustainable", "recyclable", "biodegradable",
                "energy-efficient", "green", "organic", "natural"
            ]
        }
    
    def _setup_category_analyzers(self) -> Dict[str, Dict[str, Any]]:
        """
        Setup category-specific analysis patterns
        """
        return {
            "Electronics": {
                "key_specs": ["processor", "memory", "storage", "display", "battery", "connectivity"],
                "performance_metrics": ["speed", "resolution", "capacity", "efficiency"],
                "important_features": ["compatibility", "durability", "warranty", "support"],
                "target_keywords": ["tech", "digital", "smart", "wireless", "HD", "bluetooth"]
            },
            "Clothing": {
                "key_specs": ["material", "size", "color", "fit", "care"],
                "performance_metrics": ["comfort", "durability", "style", "breathability"],
                "important_features": ["fabric", "design", "occasion", "season"],
                "target_keywords": ["fashion", "style", "comfortable", "trendy", "quality"]
            },
            "Home & Garden": {
                "key_specs": ["dimensions", "material", "capacity", "installation"],
                "performance_metrics": ["durability", "functionality", "aesthetics"],
                "important_features": ["weather-resistance", "maintenance", "assembly"],
                "target_keywords": ["home", "outdoor", "indoor", "decorative", "functional"]
            },
            "Beauty": {
                "key_specs": ["ingredients", "skin_type", "application", "volume"],
                "performance_metrics": ["effectiveness", "safety", "longevity"],
                "important_features": ["hypoallergenic", "natural", "tested", "certified"],
                "target_keywords": ["beauty", "skincare", "makeup", "natural", "gentle"]
            },
            "Sports": {
                "key_specs": ["size", "weight", "material", "performance"],
                "performance_metrics": ["durability", "performance", "comfort", "safety"],
                "important_features": ["professional", "training", "competition", "fitness"],
                "target_keywords": ["sports", "fitness", "training", "performance", "athletic"]
            }
        }
    
    def _setup_specification_patterns(self) -> Dict[str, str]:
        """
        Setup regex patterns for extracting specifications
        """
        return {
            "dimensions": r'(\d+(?:\.\d+)?)\s*[x×]\s*(\d+(?:\.\d+)?)\s*[x×]?\s*(\d+(?:\.\d+)?)?.*?(?:inches?|in|cm|mm|feet?|ft)',
            "weight": r'(\d+(?:\.\d+)?)\s*(lbs?|pounds?|kg|kilograms?|oz|ounces?|grams?|g)',
            "capacity": r'(\d+(?:\.\d+)?)\s*(liters?|l|gallons?|gal|ml|milliliters?|oz|ounces?)',
            "power": r'(\d+(?:\.\d+)?)\s*(watts?|w|volts?|v|amps?|a|mah|milliamps?)',
            "speed": r'(\d+(?:\.\d+)?)\s*(mph|kmh|km/h|rpm|ghz|mhz|fps)',
            "resolution": r'(\d+)\s*[x×]\s*(\d+)(?:\s*(?:pixels?|px|p))?',
            "memory": r'(\d+(?:\.\d+)?)\s*(gb|mb|tb|gigabytes?|megabytes?|terabytes?)',
            "warranty": r'(\d+)\s*(?:year|yr|month|mo)\s*warranty'
        }
    
    async def analyze_product_specifications(
        self,
        product_id: str
    ) -> ProductSpecification:
        """
        Analyze product and extract structured specifications
        """
        try:
            # Get product from database
            db = next(get_db())
            product = db.query(Product).filter(Product.id == product_id).first()
            db.close()
            
            if not product:
                raise ValueError(f"Product {product_id} not found")
            
            logger.info(f"Analyzing product specifications for: {product.name}")
            
            # Extract base information
            spec = ProductSpecification(
                name=product.name,
                category=product.category or "General",
                brand=product.brand or "Unknown",
                price=float(product.price)
            )
            
            # Parse features from description and features field
            spec.features = await self._extract_features(product)
            
            # Extract technical specifications
            spec.specifications = await self._extract_specifications(product)
            
            # Analyze dimensions and physical properties
            spec.dimensions = await self._extract_dimensions(product)
            spec.weight = await self._extract_weight(product)
            
            # Extract material and construction details
            spec.materials = await self._extract_materials(product)
            
            # Extract variant information
            spec.colors = await self._extract_colors(product)
            spec.sizes = await self._extract_sizes(product)
            
            # Extract warranty and certifications
            spec.warranty = await self._extract_warranty(product)
            spec.certifications = await self._extract_certifications(product)
            
            # Analyze target audience and use cases
            spec.target_audience = await self._identify_target_audience(product, spec)
            spec.use_cases = await self._identify_use_cases(product, spec)
            
            # Extract benefits and unique selling points
            spec.benefits = await self._extract_benefits(product, spec)
            spec.unique_selling_points = await self._identify_unique_selling_points(product, spec)
            
            # Category-specific analysis
            spec.technical_specs = await self._analyze_category_specifics(product, spec)
            
            logger.info(f"Product analysis completed: {len(spec.features)} features, {len(spec.benefits)} benefits")
            
            return spec
            
        except Exception as e:
            logger.error(f"Product specification analysis failed: {e}")
            raise
    
    async def _extract_features(self, product: Product) -> List[str]:
        """
        Extract product features from various fields
        """
        features = []
        
        # From features field (if it exists and is JSON)
        if hasattr(product, 'features') and product.features:
            try:
                if isinstance(product.features, str):
                    features_data = json.loads(product.features)
                else:
                    features_data = product.features
                
                if isinstance(features_data, list):
                    features.extend(features_data)
                elif isinstance(features_data, dict):
                    for key, value in features_data.items():
                        if isinstance(value, list):
                            features.extend(value)
                        else:
                            features.append(f"{key}: {value}")
            except json.JSONDecodeError:
                # If features is a plain text string
                features.extend([f.strip() for f in product.features.split('\n') if f.strip()])
        
        # Extract from description using feature keywords
        if product.description:
            description_features = await self._extract_features_from_text(product.description)
            features.extend(description_features)
        
        # Remove duplicates while preserving order
        unique_features = []
        seen = set()
        for feature in features:
            feature_clean = feature.strip().lower()
            if feature_clean not in seen and len(feature_clean) > 2:
                unique_features.append(feature.strip())
                seen.add(feature_clean)
        
        return unique_features
    
    async def _extract_features_from_text(self, text: str) -> List[str]:
        """
        Extract features from text using NLP and pattern matching
        """
        features = []
        text_lower = text.lower()
        
        # Look for feature indicators
        feature_patterns = [
            r'features?:?\s*([^\.]+)',
            r'includes?:?\s*([^\.]+)',
            r'specifications?:?\s*([^\.]+)',
            r'benefits?:?\s*([^\.]+)',
            r'with\s+([^\.]+)',
            r'equipped with\s+([^\.]+)'
        ]
        
        for pattern in feature_patterns:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                feature_text = match.group(1)
                # Split by common delimiters
                items = re.split(r'[,;•\n]', feature_text)
                for item in items:
                    item = item.strip()
                    if len(item) > 5 and len(item) < 100:
                        features.append(item)
        
        # Look for bullet points and lists
        bullet_patterns = [
            r'[•\-\*]\s*([^•\-\*\n]+)',
            r'\n\s*[\-\*]\s*([^\n]+)',
            r'\d+\.\s*([^\n]+)'
        ]
        
        for pattern in bullet_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                feature = match.group(1).strip()
                if len(feature) > 5 and len(feature) < 100:
                    features.append(feature)
        
        return features
    
    async def _extract_specifications(self, product: Product) -> Dict[str, Any]:
        """
        Extract technical specifications using patterns
        """
        specifications = {}
        
        # Combine description and features for analysis
        text = ""
        if product.description:
            text += product.description + " "
        if hasattr(product, 'features') and product.features:
            if isinstance(product.features, str):
                text += product.features
        
        # Apply specification patterns
        for spec_type, pattern in self.specification_patterns.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                if spec_type == "dimensions":
                    dimensions = [match.group(1), match.group(2)]
                    if match.group(3):
                        dimensions.append(match.group(3))
                    specifications[spec_type] = dimensions
                elif spec_type == "resolution":
                    specifications[spec_type] = f"{match.group(1)}x{match.group(2)}"
                else:
                    specifications[spec_type] = f"{match.group(1)} {match.group(2)}"
        
        return specifications
    
    async def _extract_dimensions(self, product: Product) -> Dict[str, Any]:
        """
        Extract dimensional information
        """
        dimensions = {}
        
        # Check if already extracted in specifications
        if hasattr(product, 'dimensions') and product.dimensions:
            try:
                if isinstance(product.dimensions, str):
                    dimensions = json.loads(product.dimensions)
                else:
                    dimensions = product.dimensions
            except json.JSONDecodeError:
                pass
        
        # Extract from text if not found
        if not dimensions and product.description:
            dim_patterns = [
                r'(\d+(?:\.\d+)?)\s*["\']?\s*[wW]\s*[x×]\s*(\d+(?:\.\d+)?)\s*["\']?\s*[hH]\s*[x×]?\s*(\d+(?:\.\d+)?)?\s*["\']?\s*[dDlL]?',
                r'dimensions?:?\s*(\d+(?:\.\d+)?)\s*[x×]\s*(\d+(?:\.\d+)?)\s*[x×]?\s*(\d+(?:\.\d+)?)?'
            ]
            
            for pattern in dim_patterns:
                match = re.search(pattern, product.description, re.IGNORECASE)
                if match:
                    dimensions = {
                        "width": float(match.group(1)),
                        "height": float(match.group(2))
                    }
                    if match.group(3):
                        dimensions["depth"] = float(match.group(3))
                    break
        
        return dimensions
    
    async def _extract_weight(self, product: Product) -> Optional[str]:
        """
        Extract weight information
        """
        if hasattr(product, 'weight') and product.weight:
            return product.weight
        
        # Extract from description
        if product.description:
            weight_match = re.search(self.specification_patterns["weight"], product.description, re.IGNORECASE)
            if weight_match:
                return f"{weight_match.group(1)} {weight_match.group(2)}"
        
        return None
    
    async def _extract_materials(self, product: Product) -> List[str]:
        """
        Extract material information
        """
        materials = []
        
        # Common materials by category
        material_keywords = {
            "metals": ["aluminum", "steel", "iron", "brass", "copper", "titanium", "zinc"],
            "plastics": ["plastic", "abs", "polycarbonate", "acrylic", "vinyl", "pvc"],
            "textiles": ["cotton", "polyester", "nylon", "wool", "silk", "leather", "canvas"],
            "wood": ["wood", "bamboo", "oak", "pine", "mahogany", "cedar", "teak"],
            "glass": ["glass", "tempered", "crystal", "ceramic"],
            "composites": ["carbon fiber", "fiberglass", "composite"]
        }
        
        text = (product.description or "").lower()
        if hasattr(product, 'features') and product.features:
            text += " " + str(product.features).lower()
        
        for category, keywords in material_keywords.items():
            for keyword in keywords:
                if keyword in text:
                    materials.append(keyword.title())
        
        return list(set(materials))
    
    async def _extract_colors(self, product: Product) -> List[str]:
        """
        Extract available colors
        """
        colors = []
        
        # Common color names
        color_keywords = [
            "black", "white", "red", "blue", "green", "yellow", "orange", "purple",
            "pink", "brown", "gray", "grey", "silver", "gold", "beige", "navy",
            "maroon", "teal", "cyan", "magenta", "violet", "indigo", "turquoise"
        ]
        
        text = (product.description or "").lower()
        if product.name:
            text += " " + product.name.lower()
        
        for color in color_keywords:
            if color in text:
                colors.append(color.title())
        
        return colors
    
    async def _extract_sizes(self, product: Product) -> List[str]:
        """
        Extract available sizes
        """
        sizes = []
        
        # Common size patterns
        size_patterns = [
            r'\b(XS|S|M|L|XL|XXL|XXXL)\b',
            r'\b(\d+(?:\.\d+)?)\s*(?:inches?|in|cm|mm)\b',
            r'\bsize\s+(\w+)\b'
        ]
        
        text = product.description or ""
        if product.name:
            text += " " + product.name
        
        for pattern in size_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                size = match.group(1)
                if size not in sizes:
                    sizes.append(size)
        
        return sizes
    
    async def _extract_warranty(self, product: Product) -> Optional[str]:
        """
        Extract warranty information
        """
        if product.description:
            warranty_match = re.search(self.specification_patterns["warranty"], product.description, re.IGNORECASE)
            if warranty_match:
                return f"{warranty_match.group(1)} year warranty"
        
        return None
    
    async def _extract_certifications(self, product: Product) -> List[str]:
        """
        Extract certifications and standards
        """
        certifications = []
        
        cert_keywords = [
            "FDA approved", "CE certified", "ISO certified", "UL listed",
            "Energy Star", "OSHA compliant", "FCC approved", "RoHS compliant",
            "GREENGUARD certified", "CARB compliant"
        ]
        
        text = (product.description or "").lower()
        
        for cert in cert_keywords:
            if cert.lower() in text:
                certifications.append(cert)
        
        return certifications
    
    async def _identify_target_audience(
        self,
        product: Product,
        spec: ProductSpecification
    ) -> List[str]:
        """
        Identify target audience based on product characteristics
        """
        audiences = []
        
        # Category-based audience mapping
        category_audiences = {
            "Electronics": ["tech enthusiasts", "professionals", "gamers", "students"],
            "Clothing": ["fashion-conscious", "professionals", "casual wear", "athletes"],
            "Home & Garden": ["homeowners", "gardeners", "decorators", "DIY enthusiasts"],
            "Beauty": ["beauty enthusiasts", "skincare focused", "makeup artists", "professionals"],
            "Sports": ["athletes", "fitness enthusiasts", "professionals", "hobbyists"]
        }
        
        category = spec.category
        if category in category_audiences:
            audiences.extend(category_audiences[category])
        
        # Price-based audience
        if spec.price < 25:
            audiences.append("budget-conscious")
        elif spec.price > 200:
            audiences.append("premium buyers")
        else:
            audiences.append("value seekers")
        
        # Feature-based audience
        feature_text = " ".join(spec.features).lower()
        if "professional" in feature_text or "commercial" in feature_text:
            audiences.append("professionals")
        if "beginner" in feature_text or "easy" in feature_text:
            audiences.append("beginners")
        
        return list(set(audiences))
    
    async def _identify_use_cases(
        self,
        product: Product,
        spec: ProductSpecification
    ) -> List[str]:
        """
        Identify potential use cases and applications
        """
        use_cases = []
        
        # Extract from description
        if product.description:
            use_case_patterns = [
                r'(?:perfect|ideal|great)\s+for\s+([^\.]+)',
                r'use\s+(?:for|in|with)\s+([^\.]+)',
                r'suitable\s+for\s+([^\.]+)',
                r'designed\s+for\s+([^\.]+)'
            ]
            
            for pattern in use_case_patterns:
                matches = re.finditer(pattern, product.description, re.IGNORECASE)
                for match in matches:
                    use_case = match.group(1).strip()
                    if len(use_case) > 3 and len(use_case) < 50:
                        use_cases.append(use_case)
        
        # Category-specific use cases
        category_use_cases = {
            "Electronics": ["work", "entertainment", "communication", "productivity"],
            "Clothing": ["casual wear", "formal occasions", "exercise", "outdoor activities"],
            "Home & Garden": ["decoration", "organization", "maintenance", "improvement"],
            "Beauty": ["daily routine", "special occasions", "skincare", "makeup"],
            "Sports": ["training", "competition", "recreation", "fitness"]
        }
        
        if spec.category in category_use_cases:
            use_cases.extend(category_use_cases[spec.category])
        
        return list(set(use_cases))
    
    async def _extract_benefits(
        self,
        product: Product,
        spec: ProductSpecification
    ) -> List[str]:
        """
        Extract and infer product benefits
        """
        benefits = []
        
        # Extract explicit benefits from description
        if product.description:
            benefit_patterns = [
                r'benefits?:?\s*([^\.]+)',
                r'advantages?:?\s*([^\.]+)',
                r'helps?\s+(?:you\s+)?([^\.]+)',
                r'provides?\s+([^\.]+)',
                r'delivers?\s+([^\.]+)'
            ]
            
            for pattern in benefit_patterns:
                matches = re.finditer(pattern, product.description, re.IGNORECASE)
                for match in matches:
                    benefit = match.group(1).strip()
                    if len(benefit) > 5 and len(benefit) < 100:
                        benefits.append(benefit)
        
        # Infer benefits from features
        feature_text = " ".join(spec.features).lower()
        
        # Map feature keywords to benefits
        feature_benefit_map = {
            "durable": "long-lasting performance",
            "portable": "convenient mobility",
            "wireless": "freedom of movement",
            "automatic": "effortless operation",
            "energy-efficient": "cost savings",
            "eco-friendly": "environmental responsibility",
            "high-quality": "superior performance",
            "easy-to-use": "user convenience",
            "fast": "time savings",
            "secure": "peace of mind"
        }
        
        for keyword, benefit in feature_benefit_map.items():
            if keyword in feature_text:
                benefits.append(benefit)
        
        return list(set(benefits))
    
    async def _identify_unique_selling_points(
        self,
        product: Product,
        spec: ProductSpecification
    ) -> List[str]:
        """
        Identify unique selling propositions
        """
        usps = []
        
        # Price positioning
        if spec.price < 20:
            usps.append("affordable pricing")
        elif spec.price > 500:
            usps.append("premium quality")
        
        # Feature uniqueness
        unique_features = [
            "patented", "exclusive", "proprietary", "award-winning",
            "first-of-its-kind", "revolutionary", "breakthrough", "innovative"
        ]
        
        description = (product.description or "").lower()
        for feature in unique_features:
            if feature in description:
                usps.append(f"{feature} technology")
        
        # Brand reputation
        if spec.brand.lower() in ["apple", "samsung", "nike", "sony", "microsoft"]:
            usps.append("trusted brand reputation")
        
        # Warranty and support
        if spec.warranty:
            usps.append("comprehensive warranty coverage")
        
        # Material quality
        premium_materials = ["titanium", "carbon fiber", "premium leather", "solid wood"]
        for material in premium_materials:
            if material.lower() in description:
                usps.append("premium materials")
                break
        
        return usps
    
    async def _analyze_category_specifics(
        self,
        product: Product,
        spec: ProductSpecification
    ) -> Dict[str, Any]:
        """
        Perform category-specific analysis
        """
        category = spec.category
        technical_specs = {}
        
        if category not in self.category_analyzers:
            return technical_specs
        
        analyzer = self.category_analyzers[category]
        description = (product.description or "").lower()
        
        # Extract category-specific specifications
        for spec_key in analyzer["key_specs"]:
            # Look for this specification in the text
            patterns = [
                rf'{spec_key}:?\s*([^,\n\.]+)',
                rf'{spec_key}\s+is\s+([^,\n\.]+)',
                rf'with\s+([^,\n\.]*{spec_key}[^,\n\.]*)'
            ]
            
            for pattern in patterns:
                match = re.search(pattern, description, re.IGNORECASE)
                if match:
                    technical_specs[spec_key] = match.group(1).strip()
                    break
        
        # Performance analysis
        performance_scores = {}
        for metric in analyzer["performance_metrics"]:
            score = 0.0
            metric_keywords = [metric, f"high {metric}", f"superior {metric}", f"excellent {metric}"]
            
            for keyword in metric_keywords:
                if keyword in description:
                    score += 0.25
            
            if score > 0:
                performance_scores[metric] = min(score, 1.0)
        
        if performance_scores:
            technical_specs["performance_analysis"] = performance_scores
        
        # Feature importance scoring
        important_features = {}
        for feature in analyzer["important_features"]:
            if feature.lower() in description:
                important_features[feature] = True
        
        if important_features:
            technical_specs["important_features"] = important_features
        
        return technical_specs
    
    async def analyze_competitive_landscape(
        self,
        product_id: str,
        category: str,
        competitor_urls: Optional[List[str]] = None
    ) -> List[CompetitiveInsight]:
        """
        Analyze competitive landscape for the product
        """
        try:
            # Get existing competitive analysis from database
            db = next(get_db())
            
            existing_analysis = db.query(CompetitorAnalysis).filter(
                CompetitorAnalysis.product_category == category
            ).all()
            
            insights = []
            
            for analysis in existing_analysis:
                # Convert stored analysis to insight object
                insight = CompetitiveInsight(
                    competitor=analysis.competitor_name,
                    product_name="Competitor Product",  # Would be extracted from analysis
                    price_comparison="similar",  # Would be calculated
                    feature_gaps=[],  # Would be analyzed
                    positioning_advantages=[],
                    content_quality_score=0.7,  # Would be calculated
                    market_position="mid-range"  # Would be determined
                )
                insights.append(insight)
            
            db.close()
            
            # If no existing analysis and URLs provided, would scrape competitor data
            # This would be implemented with web scraping tools
            
            logger.info(f"Competitive analysis completed for category: {category}")
            return insights
            
        except Exception as e:
            logger.error(f"Competitive analysis failed: {e}")
            return []
    
    async def extract_seo_keywords(
        self,
        product_id: str,
        spec: ProductSpecification
    ) -> List[Dict[str, Any]]:
        """
        Extract and suggest SEO keywords for the product
        """
        try:
            keywords = []
            
            # Primary keywords from product attributes
            primary_keywords = [
                spec.name.lower(),
                spec.category.lower(),
                spec.brand.lower() if spec.brand != "Unknown" else None
            ]
            
            # Remove None values
            primary_keywords = [k for k in primary_keywords if k]
            
            # Feature-based keywords
            feature_keywords = []
            for feature in spec.features:
                # Extract meaningful keywords from features
                feature_words = re.findall(r'\b\w{3,}\b', feature.lower())
                feature_keywords.extend(feature_words)
            
            # Long-tail keywords
            long_tail = []
            if spec.use_cases:
                for use_case in spec.use_cases:
                    long_tail.append(f"{spec.name.lower()} for {use_case.lower()}")
            
            # Build keyword list with metadata
            for keyword in primary_keywords:
                keywords.append({
                    "keyword": keyword,
                    "type": "primary",
                    "search_volume": 1000,  # Would be fetched from SEO API
                    "competition": "medium",
                    "relevance_score": 0.9
                })
            
            for keyword in feature_keywords[:10]:  # Limit feature keywords
                keywords.append({
                    "keyword": keyword,
                    "type": "feature",
                    "search_volume": 500,
                    "competition": "low",
                    "relevance_score": 0.7
                })
            
            for keyword in long_tail[:5]:  # Limit long-tail keywords
                keywords.append({
                    "keyword": keyword,
                    "type": "long_tail",
                    "search_volume": 100,
                    "competition": "low",
                    "relevance_score": 0.8
                })
            
            logger.info(f"Extracted {len(keywords)} SEO keywords for product")
            return keywords
            
        except Exception as e:
            logger.error(f"SEO keyword extraction failed: {e}")
            return []
    
    async def generate_product_analysis_report(
        self,
        product_id: str
    ) -> Dict[str, Any]:
        """
        Generate comprehensive product analysis report
        """
        try:
            # Perform complete analysis
            spec = await self.analyze_product_specifications(product_id)
            keywords = await self.extract_seo_keywords(product_id, spec)
            competitive_insights = await self.analyze_competitive_landscape(
                product_id, spec.category
            )
            
            # Compile comprehensive report
            report = {
                "product_id": product_id,
                "analysis_date": datetime.utcnow().isoformat(),
                "specifications": {
                    "basic_info": {
                        "name": spec.name,
                        "category": spec.category,
                        "brand": spec.brand,
                        "price": spec.price
                    },
                    "features": spec.features,
                    "technical_specs": spec.specifications,
                    "physical_attributes": {
                        "dimensions": spec.dimensions,
                        "weight": spec.weight,
                        "materials": spec.materials
                    },
                    "variants": {
                        "colors": spec.colors,
                        "sizes": spec.sizes
                    }
                },
                "market_analysis": {
                    "target_audience": spec.target_audience,
                    "use_cases": spec.use_cases,
                    "benefits": spec.benefits,
                    "unique_selling_points": spec.unique_selling_points,
                    "competitive_insights": [
                        {
                            "competitor": insight.competitor,
                            "position": insight.market_position,
                            "advantages": insight.positioning_advantages
                        }
                        for insight in competitive_insights
                    ]
                },
                "seo_analysis": {
                    "primary_keywords": [k for k in keywords if k["type"] == "primary"],
                    "feature_keywords": [k for k in keywords if k["type"] == "feature"],
                    "long_tail_keywords": [k for k in keywords if k["type"] == "long_tail"],
                    "keyword_count": len(keywords)
                },
                "content_recommendations": {
                    "focus_features": spec.features[:5],
                    "highlight_benefits": spec.benefits[:3],
                    "target_keywords": [k["keyword"] for k in keywords[:10]],
                    "tone_suggestions": await self._suggest_content_tone(spec)
                },
                "quality_scores": {
                    "feature_completeness": len(spec.features) / 10.0 if len(spec.features) <= 10 else 1.0,
                    "specification_detail": len(spec.specifications) / 5.0 if len(spec.specifications) <= 5 else 1.0,
                    "market_positioning": len(spec.unique_selling_points) / 3.0 if len(spec.unique_selling_points) <= 3 else 1.0
                }
            }
            
            logger.info(f"Generated comprehensive analysis report for product {product_id}")
            return report
            
        except Exception as e:
            logger.error(f"Analysis report generation failed: {e}")
            return {"error": str(e), "product_id": product_id}
    
    async def _suggest_content_tone(self, spec: ProductSpecification) -> List[str]:
        """
        Suggest appropriate content tone based on product characteristics
        """
        tone_suggestions = []
        
        # Price-based tone
        if spec.price < 50:
            tone_suggestions.append("friendly and accessible")
        elif spec.price > 500:
            tone_suggestions.append("premium and sophisticated")
        else:
            tone_suggestions.append("professional and trustworthy")
        
        # Category-based tone
        category_tones = {
            "Electronics": "technical and informative",
            "Clothing": "stylish and aspirational", 
            "Beauty": "caring and personal",
            "Sports": "energetic and motivational",
            "Home & Garden": "practical and helpful"
        }
        
        if spec.category in category_tones:
            tone_suggestions.append(category_tones[spec.category])
        
        # Feature-based tone
        feature_text = " ".join(spec.features).lower()
        if "professional" in feature_text:
            tone_suggestions.append("authoritative and expert")
        if "easy" in feature_text or "simple" in feature_text:
            tone_suggestions.append("approachable and clear")
        
        return list(set(tone_suggestions))