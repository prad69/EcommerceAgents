import asyncio
from typing import Dict, List, Any, Optional, Tuple
from loguru import logger
import json
import random
from datetime import datetime, timedelta
from dataclasses import dataclass
import statistics
import math

from src.core.database import get_db
from src.models.product_description import DescriptionABTest, ProductDescription, DescriptionType
from src.models.product import Product


@dataclass
class ABTestConfig:
    """A/B test configuration"""
    test_name: str
    product_id: str
    variants: List[Dict[str, Any]]
    traffic_allocation: Dict[str, float]  # variant_name -> percentage
    success_metrics: List[str]
    duration_days: int
    confidence_threshold: float = 0.95
    minimum_sample_size: int = 100


@dataclass
class TestResults:
    """A/B test results"""
    test_id: str
    status: str
    duration_days: int
    total_participants: int
    confidence_level: float
    winner: Optional[str]
    statistical_significance: float
    effect_size: float
    conversion_rates: Dict[str, float]
    sample_sizes: Dict[str, int]
    recommendations: List[str]


@dataclass
class VariantMetrics:
    """Metrics for a single test variant"""
    variant_name: str
    participants: int
    views: int
    clicks: int
    conversions: int
    revenue: float
    conversion_rate: float
    click_through_rate: float
    revenue_per_visitor: float
    confidence_interval: Tuple[float, float]


class ABTestingService:
    """
    A/B testing service for product descriptions
    """
    
    def __init__(self):
        self.active_tests = {}
        self.test_assignments = {}  # session_id -> variant assignments
        self.statistical_methods = self._setup_statistical_methods()
    
    def _setup_statistical_methods(self) -> Dict[str, Any]:
        """
        Setup statistical analysis methods
        """
        return {
            "confidence_levels": {
                0.90: 1.645,  # Z-score for 90% confidence
                0.95: 1.96,   # Z-score for 95% confidence
                0.99: 2.576   # Z-score for 99% confidence
            },
            "minimum_effect_size": 0.05,  # 5% minimum detectable effect
            "alpha": 0.05,  # Type I error rate
            "power": 0.8    # Statistical power
        }
    
    async def create_ab_test(
        self,
        config: ABTestConfig,
        auto_start: bool = False
    ) -> str:
        """
        Create a new A/B test for product descriptions
        """
        try:
            logger.info(f"Creating A/B test: {config.test_name}")
            
            # Validate test configuration
            await self._validate_test_config(config)
            
            # Calculate statistical requirements
            sample_size_per_variant = await self._calculate_sample_size(
                config.confidence_threshold,
                self.statistical_methods["minimum_effect_size"],
                self.statistical_methods["power"]
            )
            
            db = next(get_db())
            
            # Create test records for each variant
            test_records = []
            
            for i, variant in enumerate(config.variants):
                variant_name = variant.get("name", f"variant_{chr(65 + i)}")  # A, B, C, etc.
                traffic_allocation = config.traffic_allocation.get(variant_name, 1.0 / len(config.variants))
                
                test_record = DescriptionABTest(
                    test_name=config.test_name,
                    product_id=config.product_id,
                    description_id=variant["description_id"],
                    variant_name=variant_name,
                    traffic_allocation=traffic_allocation,
                    status="draft" if not auto_start else "running",
                    planned_duration_days=config.duration_days,
                    start_date=datetime.utcnow() if auto_start else None
                )
                
                db.add(test_record)
                test_records.append(test_record)
            
            db.commit()
            
            # Get test IDs after commit
            for record in test_records:
                db.refresh(record)
            
            test_id = f"test_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            
            # Store test configuration
            test_config = {
                "test_id": test_id,
                "test_name": config.test_name,
                "product_id": config.product_id,
                "variants": {
                    record.variant_name: {
                        "id": str(record.id),
                        "description_id": record.description_id,
                        "traffic_allocation": record.traffic_allocation
                    }
                    for record in test_records
                },
                "success_metrics": config.success_metrics,
                "sample_size_per_variant": sample_size_per_variant,
                "confidence_threshold": config.confidence_threshold,
                "created_at": datetime.utcnow().isoformat(),
                "status": "draft" if not auto_start else "running"
            }
            
            # Cache active test
            if auto_start:
                self.active_tests[config.product_id] = test_config
            
            db.close()
            
            logger.info(f"A/B test created with ID: {test_id}")
            return test_id
            
        except Exception as e:
            logger.error(f"A/B test creation failed: {e}")
            raise
    
    async def _validate_test_config(self, config: ABTestConfig):
        """
        Validate A/B test configuration
        """
        # Check if product exists
        db = next(get_db())
        product = db.query(Product).filter(Product.id == config.product_id).first()
        if not product:
            raise ValueError(f"Product {config.product_id} not found")
        
        # Validate variants
        if len(config.variants) < 2:
            raise ValueError("At least 2 variants required for A/B test")
        
        if len(config.variants) > 5:
            raise ValueError("Maximum 5 variants supported")
        
        # Check if descriptions exist
        for variant in config.variants:
            description = db.query(ProductDescription).filter(
                ProductDescription.id == variant["description_id"]
            ).first()
            if not description:
                raise ValueError(f"Description {variant['description_id']} not found")
        
        # Validate traffic allocation
        total_allocation = sum(config.traffic_allocation.values())
        if abs(total_allocation - 1.0) > 0.01:  # Allow small floating point errors
            raise ValueError(f"Traffic allocation must sum to 1.0, got {total_allocation}")
        
        # Check for existing tests on this product
        existing_test = db.query(DescriptionABTest).filter(
            DescriptionABTest.product_id == config.product_id,
            DescriptionABTest.status.in_(["running", "paused"])
        ).first()
        
        if existing_test:
            raise ValueError(f"Product {config.product_id} already has an active test")
        
        db.close()
    
    async def _calculate_sample_size(
        self,
        confidence_level: float,
        effect_size: float,
        power: float
    ) -> int:
        """
        Calculate required sample size for statistical significance
        """
        try:
            # Get Z-scores
            alpha = 1 - confidence_level
            z_alpha = self.statistical_methods["confidence_levels"].get(confidence_level, 1.96)
            z_beta = 0.84  # Z-score for 80% power
            
            # Assumed baseline conversion rate (can be improved with historical data)
            baseline_conversion = 0.05  # 5%
            
            # Calculate sample size using formula for comparing two proportions
            p1 = baseline_conversion
            p2 = baseline_conversion * (1 + effect_size)
            p_pooled = (p1 + p2) / 2
            
            numerator = (z_alpha + z_beta) ** 2 * 2 * p_pooled * (1 - p_pooled)
            denominator = (p2 - p1) ** 2
            
            sample_size = max(int(math.ceil(numerator / denominator)), 100)
            
            logger.debug(f"Calculated sample size: {sample_size} per variant")
            return sample_size
            
        except Exception as e:
            logger.warning(f"Sample size calculation failed, using default: {e}")
            return 500  # Default sample size
    
    async def assign_variant(
        self,
        product_id: str,
        session_id: str,
        user_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Assign a variant to a user session
        """
        try:
            # Check if test is active for this product
            if product_id not in self.active_tests:
                return None
            
            # Check if user already assigned
            cache_key = f"{product_id}:{session_id}"
            if cache_key in self.test_assignments:
                return self.test_assignments[cache_key]
            
            test_config = self.active_tests[product_id]
            variants = test_config["variants"]
            
            # Assign variant based on traffic allocation
            variant = await self._select_variant_by_allocation(variants, session_id)
            
            # Get description content
            db = next(get_db())
            description = db.query(ProductDescription).filter(
                ProductDescription.id == variant["description_id"]
            ).first()
            db.close()
            
            if not description:
                logger.error(f"Description {variant['description_id']} not found")
                return None
            
            assignment = {
                "test_id": test_config["test_id"],
                "variant_name": variant["variant_name"],
                "description_id": variant["description_id"],
                "description_content": description.content,
                "description_title": description.title,
                "assigned_at": datetime.utcnow().isoformat()
            }
            
            # Cache assignment
            self.test_assignments[cache_key] = assignment
            
            # Record participant
            await self._record_test_participant(
                variant["id"], session_id, user_id
            )
            
            return assignment
            
        except Exception as e:
            logger.error(f"Variant assignment failed: {e}")
            return None
    
    async def _select_variant_by_allocation(
        self,
        variants: Dict[str, Dict[str, Any]],
        session_id: str
    ) -> Dict[str, Any]:
        """
        Select variant based on traffic allocation using deterministic method
        """
        # Use session ID hash for deterministic assignment
        hash_value = hash(session_id) % 100 / 100.0  # Convert to 0-1 range
        
        cumulative_allocation = 0.0
        for variant_name, variant_data in variants.items():
            cumulative_allocation += variant_data["traffic_allocation"]
            if hash_value <= cumulative_allocation:
                return {
                    "variant_name": variant_name,
                    "id": variant_data["id"],
                    "description_id": variant_data["description_id"]
                }
        
        # Fallback to first variant
        first_variant = next(iter(variants.values()))
        return {
            "variant_name": next(iter(variants.keys())),
            "id": first_variant["id"],
            "description_id": first_variant["description_id"]
        }
    
    async def _record_test_participant(
        self,
        variant_id: str,
        session_id: str,
        user_id: Optional[str] = None
    ):
        """
        Record test participant in database
        """
        try:
            db = next(get_db())
            
            # Update participant count
            test_record = db.query(DescriptionABTest).filter(
                DescriptionABTest.id == variant_id
            ).first()
            
            if test_record:
                test_record.participant_count += 1
                db.commit()
            
            db.close()
            
        except Exception as e:
            logger.warning(f"Failed to record test participant: {e}")
    
    async def record_test_event(
        self,
        product_id: str,
        session_id: str,
        event_type: str,
        event_data: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Record test event (view, click, conversion)
        """
        try:
            # Get user assignment
            cache_key = f"{product_id}:{session_id}"
            assignment = self.test_assignments.get(cache_key)
            
            if not assignment:
                return False
            
            # Update test metrics in database
            db = next(get_db())
            
            variant_id = None
            for test_config in self.active_tests.values():
                for variant_name, variant_data in test_config["variants"].items():
                    if variant_name == assignment["variant_name"]:
                        variant_id = variant_data["id"]
                        break
            
            if not variant_id:
                db.close()
                return False
            
            test_record = db.query(DescriptionABTest).filter(
                DescriptionABTest.id == variant_id
            ).first()
            
            if test_record:
                if event_type == "view":
                    test_record.views += 1
                elif event_type == "click":
                    test_record.clicks += 1
                elif event_type == "conversion":
                    test_record.conversions += 1
                    if event_data and "revenue" in event_data:
                        test_record.revenue += float(event_data["revenue"])
                
                # Update conversion rate
                if test_record.views > 0:
                    test_record.conversion_rate = test_record.conversions / test_record.views
                
                db.commit()
            
            db.close()
            
            # Check if test should be concluded
            await self._check_test_completion(product_id)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to record test event: {e}")
            return False
    
    async def _check_test_completion(self, product_id: str):
        """
        Check if test has sufficient data for statistical significance
        """
        try:
            if product_id not in self.active_tests:
                return
            
            test_config = self.active_tests[product_id]
            
            # Get current test data
            db = next(get_db())
            
            test_records = []
            for variant_data in test_config["variants"].values():
                record = db.query(DescriptionABTest).filter(
                    DescriptionABTest.id == variant_data["id"]
                ).first()
                if record:
                    test_records.append(record)
            
            db.close()
            
            if len(test_records) < 2:
                return
            
            # Check sample size
            min_sample_size = test_config.get("sample_size_per_variant", 100)
            all_variants_sufficient = all(
                record.participant_count >= min_sample_size 
                for record in test_records
            )
            
            if not all_variants_sufficient:
                return
            
            # Perform statistical analysis
            analysis_result = await self._analyze_test_significance(test_records)
            
            # Check if we have a significant winner
            if (analysis_result["statistical_significance"] >= test_config["confidence_threshold"] and
                analysis_result["effect_size"] >= self.statistical_methods["minimum_effect_size"]):
                
                await self._conclude_test(product_id, analysis_result)
            
        except Exception as e:
            logger.error(f"Test completion check failed: {e}")
    
    async def _analyze_test_significance(
        self,
        test_records: List[DescriptionABTest]
    ) -> Dict[str, Any]:
        """
        Perform statistical significance analysis
        """
        try:
            if len(test_records) < 2:
                return {"statistical_significance": 0.0, "effect_size": 0.0}
            
            # Calculate conversion rates
            conversion_rates = []
            sample_sizes = []
            
            for record in test_records:
                if record.views > 0:
                    conversion_rate = record.conversions / record.views
                    conversion_rates.append(conversion_rate)
                    sample_sizes.append(record.views)
                else:
                    conversion_rates.append(0.0)
                    sample_sizes.append(0)
            
            if len([cr for cr in conversion_rates if cr > 0]) < 2:
                return {"statistical_significance": 0.0, "effect_size": 0.0}
            
            # Perform two-proportion z-test for the top two variants
            sorted_variants = sorted(
                zip(conversion_rates, sample_sizes, test_records),
                key=lambda x: x[0],
                reverse=True
            )
            
            # Compare best vs second best
            cr1, n1, record1 = sorted_variants[0]
            cr2, n2, record2 = sorted_variants[1]
            
            if n1 == 0 or n2 == 0:
                return {"statistical_significance": 0.0, "effect_size": 0.0}
            
            # Calculate pooled conversion rate
            x1, x2 = record1.conversions, record2.conversions
            p_pooled = (x1 + x2) / (n1 + n2)
            
            # Calculate standard error
            se = math.sqrt(p_pooled * (1 - p_pooled) * (1/n1 + 1/n2))
            
            if se == 0:
                return {"statistical_significance": 0.0, "effect_size": 0.0}
            
            # Calculate z-score
            z_score = abs(cr1 - cr2) / se
            
            # Convert to confidence level (simplified)
            confidence_level = min(0.99, 1 - 2 * (1 - self._normal_cdf(abs(z_score))))
            
            # Calculate effect size (relative improvement)
            effect_size = abs(cr1 - cr2) / max(cr2, 0.001)  # Avoid division by zero
            
            return {
                "statistical_significance": confidence_level,
                "effect_size": effect_size,
                "z_score": z_score,
                "winner_variant": record1.variant_name,
                "winner_conversion_rate": cr1,
                "baseline_conversion_rate": cr2,
                "relative_improvement": effect_size
            }
            
        except Exception as e:
            logger.error(f"Statistical analysis failed: {e}")
            return {"statistical_significance": 0.0, "effect_size": 0.0}
    
    def _normal_cdf(self, x: float) -> float:
        """
        Approximate normal cumulative distribution function
        """
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))
    
    async def _conclude_test(
        self,
        product_id: str,
        analysis_result: Dict[str, Any]
    ):
        """
        Conclude test and mark winner
        """
        try:
            test_config = self.active_tests.get(product_id)
            if not test_config:
                return
            
            db = next(get_db())
            
            # Update all test records
            winner_variant = analysis_result.get("winner_variant")
            
            for variant_name, variant_data in test_config["variants"].items():
                record = db.query(DescriptionABTest).filter(
                    DescriptionABTest.id == variant_data["id"]
                ).first()
                
                if record:
                    record.status = "completed"
                    record.end_date = datetime.utcnow()
                    record.confidence_level = analysis_result["statistical_significance"]
                    record.statistical_significance = analysis_result["statistical_significance"]
                    record.effect_size = analysis_result["effect_size"]
                    
                    if variant_name == winner_variant:
                        record.is_winner = True
                    
                    db.commit()
            
            db.close()
            
            # Remove from active tests
            if product_id in self.active_tests:
                del self.active_tests[product_id]
            
            # Clean up test assignments
            keys_to_remove = [k for k in self.test_assignments.keys() if k.startswith(f"{product_id}:")]
            for key in keys_to_remove:
                del self.test_assignments[key]
            
            logger.info(f"Test concluded for product {product_id}, winner: {winner_variant}")
            
        except Exception as e:
            logger.error(f"Test conclusion failed: {e}")
    
    async def get_test_results(self, test_id: str) -> Optional[TestResults]:
        """
        Get comprehensive test results
        """
        try:
            db = next(get_db())
            
            # Get all test records for this test
            test_records = db.query(DescriptionABTest).filter(
                DescriptionABTest.test_name.contains(test_id)  # Simplified lookup
            ).all()
            
            if not test_records:
                db.close()
                return None
            
            # Calculate metrics for each variant
            variant_metrics = []
            conversion_rates = {}
            sample_sizes = {}
            
            total_participants = 0
            winner = None
            max_conversion_rate = 0.0
            
            for record in test_records:
                conversion_rate = record.conversion_rate or 0.0
                sample_sizes[record.variant_name] = record.participant_count
                conversion_rates[record.variant_name] = conversion_rate
                total_participants += record.participant_count
                
                if conversion_rate > max_conversion_rate:
                    max_conversion_rate = conversion_rate
                    winner = record.variant_name
                
                # Calculate confidence interval
                if record.views > 0 and record.conversions > 0:
                    p = conversion_rate
                    n = record.views
                    margin_of_error = 1.96 * math.sqrt((p * (1 - p)) / n)
                    ci_lower = max(0, p - margin_of_error)
                    ci_upper = min(1, p + margin_of_error)
                else:
                    ci_lower = ci_upper = 0.0
                
                variant_metric = VariantMetrics(
                    variant_name=record.variant_name,
                    participants=record.participant_count,
                    views=record.views,
                    clicks=record.clicks,
                    conversions=record.conversions,
                    revenue=record.revenue,
                    conversion_rate=conversion_rate,
                    click_through_rate=record.clicks / max(record.views, 1),
                    revenue_per_visitor=record.revenue / max(record.views, 1),
                    confidence_interval=(ci_lower, ci_upper)
                )
                variant_metrics.append(variant_metric)
            
            # Get statistical analysis
            analysis_result = await self._analyze_test_significance(test_records)
            
            # Calculate test duration
            start_date = min(r.start_date for r in test_records if r.start_date)
            end_date = max(r.end_date for r in test_records if r.end_date)
            if start_date and end_date:
                duration_days = (end_date - start_date).days
            else:
                duration_days = 0
            
            # Generate recommendations
            recommendations = await self._generate_test_recommendations(
                variant_metrics, analysis_result
            )
            
            db.close()
            
            return TestResults(
                test_id=test_id,
                status=test_records[0].status,
                duration_days=duration_days,
                total_participants=total_participants,
                confidence_level=analysis_result.get("statistical_significance", 0.0),
                winner=winner,
                statistical_significance=analysis_result.get("statistical_significance", 0.0),
                effect_size=analysis_result.get("effect_size", 0.0),
                conversion_rates=conversion_rates,
                sample_sizes=sample_sizes,
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"Failed to get test results: {e}")
            return None
    
    async def _generate_test_recommendations(
        self,
        variant_metrics: List[VariantMetrics],
        analysis_result: Dict[str, Any]
    ) -> List[str]:
        """
        Generate actionable recommendations based on test results
        """
        recommendations = []
        
        if not variant_metrics:
            return recommendations
        
        # Sort variants by conversion rate
        sorted_variants = sorted(variant_metrics, key=lambda x: x.conversion_rate, reverse=True)
        best_variant = sorted_variants[0]
        
        # Statistical significance recommendations
        significance = analysis_result.get("statistical_significance", 0.0)
        
        if significance >= 0.95:
            recommendations.append(f"Implement {best_variant.variant_name} - results are statistically significant")
        elif significance >= 0.90:
            recommendations.append(f"Consider implementing {best_variant.variant_name} - strong statistical evidence")
        elif significance >= 0.80:
            recommendations.append("Results show promising trends but need more data for confidence")
        else:
            recommendations.append("No clear winner - consider running test longer or redesigning variants")
        
        # Sample size recommendations
        min_participants = min(v.participants for v in variant_metrics)
        if min_participants < 100:
            recommendations.append("Increase sample size for more reliable results")
        
        # Effect size recommendations
        effect_size = analysis_result.get("effect_size", 0.0)
        if effect_size < 0.02:
            recommendations.append("Small effect size - consider testing more different variants")
        elif effect_size > 0.20:
            recommendations.append("Large effect size detected - significant business impact expected")
        
        # Conversion rate analysis
        if best_variant.conversion_rate < 0.02:
            recommendations.append("Low overall conversion rates - consider fundamental changes to approach")
        
        # Revenue impact
        revenue_variants = [v for v in variant_metrics if v.revenue > 0]
        if len(revenue_variants) >= 2:
            revenue_sorted = sorted(revenue_variants, key=lambda x: x.revenue_per_visitor, reverse=True)
            best_revenue = revenue_sorted[0]
            if best_revenue.variant_name == best_variant.variant_name:
                recommendations.append("Winner has both highest conversion and revenue - clear choice")
            else:
                recommendations.append("Different winners for conversion vs revenue - consider business priorities")
        
        return recommendations
    
    async def start_test(self, test_id: str) -> bool:
        """
        Start a created test
        """
        try:
            db = next(get_db())
            
            test_records = db.query(DescriptionABTest).filter(
                DescriptionABTest.test_name.contains(test_id),
                DescriptionABTest.status == "draft"
            ).all()
            
            if not test_records:
                db.close()
                return False
            
            # Update status and start date
            for record in test_records:
                record.status = "running"
                record.start_date = datetime.utcnow()
                db.commit()
            
            # Add to active tests
            product_id = test_records[0].product_id
            test_config = {
                "test_id": test_id,
                "product_id": product_id,
                "variants": {
                    record.variant_name: {
                        "id": str(record.id),
                        "description_id": record.description_id,
                        "traffic_allocation": record.traffic_allocation
                    }
                    for record in test_records
                },
                "status": "running"
            }
            
            self.active_tests[product_id] = test_config
            
            db.close()
            
            logger.info(f"A/B test {test_id} started for product {product_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start test: {e}")
            return False
    
    async def stop_test(self, test_id: str, reason: str = "manual_stop") -> bool:
        """
        Stop a running test
        """
        try:
            db = next(get_db())
            
            test_records = db.query(DescriptionABTest).filter(
                DescriptionABTest.test_name.contains(test_id),
                DescriptionABTest.status == "running"
            ).all()
            
            if not test_records:
                db.close()
                return False
            
            # Update status and end date
            for record in test_records:
                record.status = "completed"
                record.end_date = datetime.utcnow()
                db.commit()
            
            # Remove from active tests
            product_id = test_records[0].product_id
            if product_id in self.active_tests:
                del self.active_tests[product_id]
            
            # Clean up assignments
            keys_to_remove = [k for k in self.test_assignments.keys() if k.startswith(f"{product_id}:")]
            for key in keys_to_remove:
                del self.test_assignments[key]
            
            db.close()
            
            logger.info(f"A/B test {test_id} stopped: {reason}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop test: {e}")
            return False
    
    async def get_active_tests(self) -> List[Dict[str, Any]]:
        """
        Get list of all active tests
        """
        try:
            db = next(get_db())
            
            active_records = db.query(DescriptionABTest).filter(
                DescriptionABTest.status == "running"
            ).all()
            
            # Group by test
            tests = {}
            for record in active_records:
                test_name = record.test_name
                if test_name not in tests:
                    tests[test_name] = {
                        "test_name": test_name,
                        "product_id": record.product_id,
                        "start_date": record.start_date.isoformat() if record.start_date else None,
                        "variants": [],
                        "total_participants": 0
                    }
                
                tests[test_name]["variants"].append({
                    "variant_name": record.variant_name,
                    "participants": record.participant_count,
                    "conversion_rate": record.conversion_rate or 0.0
                })
                tests[test_name]["total_participants"] += record.participant_count
            
            db.close()
            
            return list(tests.values())
            
        except Exception as e:
            logger.error(f"Failed to get active tests: {e}")
            return []
    
    async def get_test_analytics(
        self,
        product_id: Optional[str] = None,
        date_range: Optional[Tuple[datetime, datetime]] = None
    ) -> Dict[str, Any]:
        """
        Get analytics across all tests
        """
        try:
            db = next(get_db())
            
            query = db.query(DescriptionABTest)
            
            if product_id:
                query = query.filter(DescriptionABTest.product_id == product_id)
            
            if date_range:
                start_date, end_date = date_range
                query = query.filter(
                    DescriptionABTest.start_date >= start_date,
                    DescriptionABTest.start_date <= end_date
                )
            
            test_records = query.all()
            
            if not test_records:
                return {"total_tests": 0}
            
            # Calculate overall analytics
            total_tests = len(set(record.test_name for record in test_records))
            completed_tests = len(set(
                record.test_name for record in test_records 
                if record.status == "completed"
            ))
            
            total_participants = sum(record.participant_count for record in test_records)
            total_conversions = sum(record.conversions for record in test_records)
            total_revenue = sum(record.revenue for record in test_records)
            
            # Average metrics
            conversion_rates = [r.conversion_rate for r in test_records if r.conversion_rate]
            avg_conversion_rate = statistics.mean(conversion_rates) if conversion_rates else 0.0
            
            # Winner analysis
            winners = [r for r in test_records if r.is_winner]
            winner_improvements = []
            
            for winner in winners:
                # Find other variants in the same test
                other_variants = [
                    r for r in test_records 
                    if r.test_name == winner.test_name and r.id != winner.id
                ]
                if other_variants:
                    baseline_rate = max(v.conversion_rate or 0.0 for v in other_variants)
                    if baseline_rate > 0:
                        improvement = ((winner.conversion_rate or 0.0) - baseline_rate) / baseline_rate
                        winner_improvements.append(improvement)
            
            avg_improvement = statistics.mean(winner_improvements) if winner_improvements else 0.0
            
            db.close()
            
            return {
                "total_tests": total_tests,
                "completed_tests": completed_tests,
                "running_tests": total_tests - completed_tests,
                "total_participants": total_participants,
                "total_conversions": total_conversions,
                "total_revenue": total_revenue,
                "average_conversion_rate": avg_conversion_rate,
                "average_improvement": avg_improvement,
                "test_success_rate": completed_tests / max(total_tests, 1),
                "analysis_date": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get test analytics: {e}")
            return {"error": str(e)}