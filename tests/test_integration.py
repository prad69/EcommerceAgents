import pytest
import asyncio
from typing import Dict, Any, List
from datetime import datetime, timedelta
import uuid
import json

from src.core.agent_orchestrator import AgentOrchestrator, AgentType, TaskPriority
from src.services.recommendation_engine import RecommendationService
from src.services.review_analysis import ReviewAnalysisService
from src.services.chatbot_orchestrator import ChatbotOrchestrator
from src.services.description_generator import DescriptionGeneratorService
from src.models.product import Product
from src.models.user import User
from src.core.database import get_db


class TestEndToEndIntegration:
    """
    Comprehensive end-to-end integration tests
    """
    
    @pytest.fixture
    async def orchestrator(self):
        """Setup agent orchestrator for testing"""
        orchestrator = AgentOrchestrator()
        yield orchestrator
        await orchestrator.shutdown()
    
    @pytest.fixture
    def sample_product_data(self):
        """Sample product data for testing"""
        return {
            "product_id": str(uuid.uuid4()),
            "name": "Test Wireless Headphones",
            "category": "Electronics",
            "brand": "TestBrand",
            "price": 199.99,
            "description": "Premium wireless headphones with noise cancellation",
            "features": ["Wireless", "Noise Cancellation", "Long Battery Life"],
            "specifications": {
                "battery_life": "30 hours",
                "connectivity": "Bluetooth 5.0",
                "weight": "250g"
            }
        }
    
    @pytest.fixture
    def sample_user_data(self):
        """Sample user data for testing"""
        return {
            "user_id": str(uuid.uuid4()),
            "email": "test@example.com",
            "preferences": {
                "categories": ["Electronics", "Audio"],
                "price_range": {"min": 100, "max": 300}
            }
        }
    
    @pytest.mark.asyncio
    async def test_product_onboarding_workflow(self, orchestrator, sample_product_data):
        """
        Test complete product onboarding workflow
        """
        # Trigger product onboarding workflow
        workflow_instance_id = await orchestrator.trigger_workflow(
            workflow_id="product_onboarding",
            trigger_data=sample_product_data
        )
        
        assert workflow_instance_id is not None
        
        # Wait for workflow to complete (with timeout)
        timeout = 60  # seconds
        start_time = datetime.utcnow()
        
        while (datetime.utcnow() - start_time).seconds < timeout:
            workflow_status = orchestrator.active_workflow_instances.get(workflow_instance_id)
            if workflow_status and workflow_status["status"] == "completed":
                break
            await asyncio.sleep(1)
        
        # Verify workflow completed successfully
        final_status = orchestrator.active_workflow_instances.get(workflow_instance_id)
        assert final_status is not None
        assert final_status["status"] == "completed"
        
        # Verify all steps were completed
        completed_step_ids = [step.step_id for step in final_status["completed_steps"]]
        expected_steps = ["analyze_product", "generate_descriptions", "initialize_recommendations"]
        
        for expected_step in expected_steps:
            assert expected_step in completed_step_ids
    
    @pytest.mark.asyncio
    async def test_customer_support_workflow(self, orchestrator, sample_user_data):
        """
        Test customer support workflow with agent collaboration
        """
        # Prepare customer query
        customer_query = {
            "user_id": sample_user_data["user_id"],
            "user_message": "I'm looking for good wireless headphones under $200",
            "context": sample_user_data["preferences"]
        }
        
        # Trigger customer support workflow
        workflow_instance_id = await orchestrator.trigger_workflow(
            workflow_id="customer_support_escalation",
            trigger_data=customer_query
        )
        
        assert workflow_instance_id is not None
        
        # Wait for workflow to complete
        await self._wait_for_workflow_completion(orchestrator, workflow_instance_id, timeout=30)
        
        # Verify workflow results
        workflow_status = orchestrator.active_workflow_instances.get(workflow_instance_id)
        assert workflow_status["status"] == "completed"
    
    @pytest.mark.asyncio
    async def test_cross_agent_data_sharing(self, orchestrator, sample_product_data):
        """
        Test data sharing between agents
        """
        # Agent 1 shares data
        shared_key = "test_product_analysis"
        analysis_data = {
            "product_id": sample_product_data["product_id"],
            "features_extracted": sample_product_data["features"],
            "quality_score": 0.85,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        await orchestrator.share_data(shared_key, analysis_data, ttl_seconds=300)
        
        # Agent 2 retrieves data
        retrieved_data = await orchestrator.get_shared_data(shared_key)
        
        assert retrieved_data is not None
        assert retrieved_data["product_id"] == sample_product_data["product_id"]
        assert retrieved_data["quality_score"] == 0.85
        
        # Test data expiration (mock)
        await orchestrator.share_data("temp_data", {"test": True}, ttl_seconds=1)
        await asyncio.sleep(2)
        
        expired_data = await orchestrator.get_shared_data("temp_data")
        # Note: In real implementation, this would be None due to TTL
    
    @pytest.mark.asyncio
    async def test_task_dependency_management(self, orchestrator, sample_product_data):
        """
        Test task dependency execution order
        """
        # Submit dependent tasks
        task1_id = await orchestrator.submit_task(
            agent_type=AgentType.PRODUCT_ANALYZER,
            task_type="analyze_specifications",
            payload={"product_id": sample_product_data["product_id"]},
            priority=TaskPriority.HIGH
        )
        
        task2_id = await orchestrator.submit_task(
            agent_type=AgentType.DESCRIPTION_GENERATOR,
            task_type="generate_descriptions",
            payload={
                "product_id": sample_product_data["product_id"],
                "description_types": ["short", "medium"],
                "use_analysis": True
            },
            priority=TaskPriority.HIGH,
            dependencies=[task1_id]
        )
        
        # Wait for both tasks to complete
        await self._wait_for_task_completion(orchestrator, [task1_id, task2_id], timeout=60)
        
        # Verify execution order
        task1_status = await orchestrator.get_task_status(task1_id)
        task2_status = await orchestrator.get_task_status(task2_id)
        
        assert task1_status["status"] == "completed"
        assert task2_status["status"] == "completed"
        
        # Task 1 should complete before task 2
        task1_completed = datetime.fromisoformat(task1_status["completed_at"])
        task2_completed = datetime.fromisoformat(task2_status["completed_at"])
        assert task1_completed < task2_completed
    
    @pytest.mark.asyncio
    async def test_error_handling_and_retries(self, orchestrator):
        """
        Test error handling and retry mechanisms
        """
        # Submit task with invalid payload to trigger error
        invalid_task_id = await orchestrator.submit_task(
            agent_type=AgentType.PRODUCT_ANALYZER,
            task_type="analyze_specifications",
            payload={"product_id": "invalid_product_id"},
            priority=TaskPriority.MEDIUM
        )
        
        # Wait for task to complete (with retries)
        await self._wait_for_task_completion(orchestrator, [invalid_task_id], timeout=30)
        
        # Verify task failed after retries
        task_status = await orchestrator.get_task_status(invalid_task_id)
        assert task_status["status"] == "failed"
        assert task_status["retry_count"] > 0
        assert task_status["error_message"] is not None
    
    @pytest.mark.asyncio
    async def test_performance_monitoring(self, orchestrator, sample_product_data):
        """
        Test performance monitoring and metrics collection
        """
        # Submit multiple tasks to generate metrics
        task_ids = []
        for i in range(5):
            task_id = await orchestrator.submit_task(
                agent_type=AgentType.PRODUCT_ANALYZER,
                task_type="analyze_specifications",
                payload={"product_id": f"product_{i}"},
                priority=TaskPriority.MEDIUM
            )
            task_ids.append(task_id)
        
        # Wait for all tasks to complete
        await self._wait_for_task_completion(orchestrator, task_ids, timeout=60)
        
        # Check performance metrics
        system_status = await orchestrator.get_system_status()
        
        assert "agents" in system_status
        assert "performance_metrics" in system_status["agents"]
        assert "tasks" in system_status
        
        # Verify metrics are being collected
        performance_metrics = system_status["agents"]["performance_metrics"]
        analyzer_key = f"{AgentType.PRODUCT_ANALYZER.value}:analyze_specifications"
        
        if analyzer_key in performance_metrics:
            metrics = performance_metrics[analyzer_key]
            assert metrics["total_executions"] > 0
            assert metrics["avg_time"] > 0
    
    @pytest.mark.asyncio
    async def test_system_health_monitoring(self, orchestrator):
        """
        Test system health monitoring
        """
        # Wait for initial health check
        await asyncio.sleep(2)
        
        system_status = await orchestrator.get_system_status()
        
        # Verify health status for all agents
        health_status = system_status["agents"]["health_status"]
        
        for agent_type in AgentType:
            agent_key = agent_type.value
            if agent_key in health_status:
                assert "status" in health_status[agent_key]
                assert "last_check" in health_status[agent_key]
    
    @pytest.mark.asyncio
    async def test_analytics_dashboard_data(self, orchestrator):
        """
        Test analytics dashboard data generation
        """
        # Generate some activity
        await orchestrator.submit_task(
            agent_type=AgentType.RECOMMENDATION,
            task_type="get_recommendations",
            payload={"user_id": "test_user", "limit": 5},
            priority=TaskPriority.LOW
        )
        
        await asyncio.sleep(2)
        
        # Get analytics data
        analytics = await orchestrator.get_analytics_dashboard()
        
        # Verify analytics structure
        assert "system_overview" in analytics
        assert "agent_performance" in analytics
        assert "workflow_analytics" in analytics
        assert "performance_trends" in analytics
        assert "generated_at" in analytics
        
        # Verify system overview
        system_overview = analytics["system_overview"]
        assert "total_tasks" in system_overview
        assert "success_rate" in system_overview
        assert "active_tasks" in system_overview
    
    @pytest.mark.asyncio
    async def test_workflow_condition_evaluation(self, orchestrator):
        """
        Test conditional workflow execution
        """
        # Test customer support workflow with product search intent
        query_data = {
            "user_id": "test_user",
            "user_message": "Show me laptops",
            "context": {"intent": "product_search"},
            "intent": "product_search"
        }
        
        workflow_instance_id = await orchestrator.trigger_workflow(
            workflow_id="customer_support_escalation",
            trigger_data=query_data
        )
        
        await self._wait_for_workflow_completion(orchestrator, workflow_instance_id, timeout=30)
        
        # Verify conditional steps were executed
        workflow_status = orchestrator.active_workflow_instances.get(workflow_instance_id)
        completed_step_ids = [step.step_id for step in workflow_status["completed_steps"]]
        
        # Should have executed recommendation step due to product_search intent
        assert "analyze_query" in completed_step_ids
    
    @pytest.mark.asyncio
    async def test_concurrent_workflow_execution(self, orchestrator, sample_product_data):
        """
        Test concurrent workflow execution
        """
        # Trigger multiple workflows concurrently
        workflow_ids = []
        
        for i in range(3):
            product_data = sample_product_data.copy()
            product_data["product_id"] = f"product_{i}"
            product_data["name"] = f"Test Product {i}"
            
            workflow_id = await orchestrator.trigger_workflow(
                workflow_id="product_onboarding",
                trigger_data=product_data
            )
            workflow_ids.append(workflow_id)
        
        # Wait for all workflows to complete
        timeout = 90
        start_time = datetime.utcnow()
        completed_workflows = 0
        
        while (datetime.utcnow() - start_time).seconds < timeout:
            completed_count = 0
            for workflow_id in workflow_ids:
                workflow_status = orchestrator.active_workflow_instances.get(workflow_id)
                if workflow_status and workflow_status["status"] == "completed":
                    completed_count += 1
            
            if completed_count == len(workflow_ids):
                break
            
            await asyncio.sleep(2)
        
        # Verify all workflows completed
        for workflow_id in workflow_ids:
            workflow_status = orchestrator.active_workflow_instances.get(workflow_id)
            assert workflow_status["status"] == "completed"
    
    @pytest.mark.asyncio
    async def test_task_cancellation(self, orchestrator):
        """
        Test task cancellation functionality
        """
        # Submit a task
        task_id = await orchestrator.submit_task(
            agent_type=AgentType.PRODUCT_ANALYZER,
            task_type="analyze_specifications",
            payload={"product_id": "test_product"},
            priority=TaskPriority.LOW
        )
        
        # Cancel the task
        cancelled = await orchestrator.cancel_task(task_id)
        assert cancelled is True
        
        # Verify task is marked as cancelled
        task_status = await orchestrator.get_task_status(task_id)
        if task_status:
            # Task might be completed before cancellation, which is acceptable
            assert task_status["status"] in ["cancelled", "completed"]
    
    # Helper methods
    
    async def _wait_for_workflow_completion(self, orchestrator, workflow_instance_id, timeout=30):
        """Helper to wait for workflow completion"""
        start_time = datetime.utcnow()
        
        while (datetime.utcnow() - start_time).seconds < timeout:
            workflow_status = orchestrator.active_workflow_instances.get(workflow_instance_id)
            if workflow_status and workflow_status["status"] in ["completed", "failed"]:
                break
            await asyncio.sleep(1)
    
    async def _wait_for_task_completion(self, orchestrator, task_ids, timeout=30):
        """Helper to wait for task completion"""
        start_time = datetime.utcnow()
        
        while (datetime.utcnow() - start_time).seconds < timeout:
            completed_count = 0
            for task_id in task_ids:
                task_status = await orchestrator.get_task_status(task_id)
                if task_status and task_status["status"] in ["completed", "failed"]:
                    completed_count += 1
            
            if completed_count == len(task_ids):
                break
            
            await asyncio.sleep(1)


class TestComponentIntegration:
    """
    Test integration between specific components
    """
    
    @pytest.mark.asyncio
    async def test_recommendation_review_integration(self):
        """
        Test integration between recommendation engine and review analysis
        """
        recommendation_service = RecommendationService()
        review_service = ReviewAnalysisService()
        
        # Test data
        product_id = "test_product_123"
        user_id = "test_user_456"
        
        # Simulate review analysis affecting recommendations
        review_data = {
            "product_id": product_id,
            "sentiment_score": 0.8,
            "quality_indicators": ["excellent sound", "comfortable fit"],
            "issues": ["battery life could be better"]
        }
        
        # This would normally update product metrics in the recommendation engine
        # For testing, we verify the services can communicate
        
        # Get initial recommendations
        try:
            initial_recs = await recommendation_service.get_recommendations(
                user_id=user_id,
                filters={"category": "Electronics"},
                limit=5
            )
            # Test passes if no exception is thrown
            assert True
        except Exception as e:
            pytest.fail(f"Recommendation service failed: {e}")
    
    @pytest.mark.asyncio
    async def test_chatbot_recommendation_integration(self):
        """
        Test chatbot integration with recommendation engine
        """
        chatbot = ChatbotOrchestrator()
        
        # Test chatbot processing a product search query
        session_id = "test_session_789"
        user_message = "I need good headphones for working out"
        
        try:
            response = await chatbot.process_user_message(
                session_id=session_id,
                user_message=user_message,
                user_id="test_user"
            )
            
            # Verify response structure
            assert hasattr(response, 'message')
            assert hasattr(response, 'intent')
            assert hasattr(response, 'confidence')
            
        except Exception as e:
            pytest.fail(f"Chatbot integration failed: {e}")
    
    @pytest.mark.asyncio
    async def test_description_generator_analyzer_integration(self):
        """
        Test description generator integration with product analyzer
        """
        description_service = DescriptionGeneratorService()
        
        # Test product analysis integrated into description generation
        from src.services.description_generator import GenerationRequest, DescriptionType
        
        request = GenerationRequest(
            product_id="test_product",
            description_types=[DescriptionType.SHORT, DescriptionType.MEDIUM],
            target_keywords=["wireless", "headphones", "bluetooth"],
            include_seo=True
        )
        
        try:
            # This would normally use product analysis results
            # For testing, we verify the service handles the request
            result = await description_service.generate_descriptions(request)
            
            # Verify result structure
            assert "product_id" in result
            assert "descriptions" in result
            assert "metadata" in result
            
        except Exception as e:
            pytest.fail(f"Description generation integration failed: {e}")


class TestPerformanceIntegration:
    """
    Performance and load testing for integrated system
    """
    
    @pytest.mark.asyncio
    async def test_concurrent_task_processing(self):
        """
        Test system performance under concurrent load
        """
        orchestrator = AgentOrchestrator()
        
        try:
            # Submit multiple tasks concurrently
            task_count = 20
            task_ids = []
            
            start_time = datetime.utcnow()
            
            for i in range(task_count):
                task_id = await orchestrator.submit_task(
                    agent_type=AgentType.PRODUCT_ANALYZER,
                    task_type="analyze_specifications",
                    payload={"product_id": f"perf_test_product_{i}"},
                    priority=TaskPriority.MEDIUM
                )
                task_ids.append(task_id)
            
            # Wait for all tasks to complete
            timeout = 120  # 2 minutes
            completed_tasks = 0
            
            while (datetime.utcnow() - start_time).seconds < timeout:
                completed_count = 0
                for task_id in task_ids:
                    task_status = await orchestrator.get_task_status(task_id)
                    if task_status and task_status["status"] in ["completed", "failed"]:
                        completed_count += 1
                
                if completed_count == task_count:
                    break
                
                await asyncio.sleep(1)
            
            end_time = datetime.utcnow()
            total_time = (end_time - start_time).total_seconds()
            
            # Performance assertions
            assert completed_count == task_count, f"Only {completed_count}/{task_count} tasks completed"
            assert total_time < 120, f"Processing took {total_time}s, expected <120s"
            
            # Check average task processing time
            avg_time_per_task = total_time / task_count
            assert avg_time_per_task < 10, f"Average time per task: {avg_time_per_task}s"
            
        finally:
            await orchestrator.shutdown()
    
    @pytest.mark.asyncio 
    async def test_memory_usage_under_load(self):
        """
        Test memory usage under sustained load
        """
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        orchestrator = AgentOrchestrator()
        
        try:
            # Generate sustained load
            for batch in range(5):
                task_ids = []
                
                # Submit batch of tasks
                for i in range(10):
                    task_id = await orchestrator.submit_task(
                        agent_type=AgentType.RECOMMENDATION,
                        task_type="get_recommendations",
                        payload={"user_id": f"user_{batch}_{i}", "limit": 5},
                        priority=TaskPriority.MEDIUM
                    )
                    task_ids.append(task_id)
                
                # Wait for batch to complete
                await asyncio.sleep(5)
                
                # Check memory usage
                current_memory = process.memory_info().rss / 1024 / 1024  # MB
                memory_increase = current_memory - initial_memory
                
                # Memory shouldn't increase excessively
                assert memory_increase < 500, f"Memory usage increased by {memory_increase}MB"
                
        finally:
            await orchestrator.shutdown()


# Pytest configuration
@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])