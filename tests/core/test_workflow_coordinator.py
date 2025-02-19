import pytest
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch
from engine.core.workflow_coordinator import (
    WorkflowCoordinator,
    WorkflowConfig,
    WorkflowResult,
    WorkflowContext
)
from engine.core.query_parser import SubTask
from engine.utils.cost_tracker import TokenUsage

@pytest.fixture
def config():
    """工作流配置"""
    return WorkflowConfig(
        max_query_length=2000,
        min_tasks=3,
        default_models={
            "search": "text-embedding-3-large",
            "query_parsing": "gpt-4o",
            "task_execution": "gpt-4o",
            "response_generation": "gpt-4o"
        },
        base_tasks=[
            "当前状况分析",
            "历史背景和相关",
            "未来趋势预测"
        ]
    )

@pytest.fixture
def coordinator(config):
    """工作流协调器"""
    coordinator = WorkflowCoordinator(config)
    coordinator.query_parser.parse_query = Mock()
    coordinator.tool_orchestrator.execute_tasks = AsyncMock()
    coordinator.tool_orchestrator.generate_response = AsyncMock()
    coordinator.tool_orchestrator.get_model_for_task = Mock()
    coordinator.result_evaluator.evaluate_with_fallback = AsyncMock()
    coordinator.conversation_manager.add_message = Mock()
    coordinator.cost_tracker.track_usage = AsyncMock()
    return coordinator

@pytest.fixture
def context():
    """工作流上下文"""
    return WorkflowContext("测试查询", "test_session")

class TestQueryValidation:
    """查询验证测试"""
    
    @pytest.mark.asyncio
    async def test_empty_query(self, coordinator):
        result = await coordinator.process_query("", "test_session")
        assert "查询内容不能为空" in result.final_answer
        assert result.metadata["error"] == "查询内容不能为空"
        assert result.evaluation_result["quality_score"] == 0

    @pytest.mark.asyncio
    async def test_long_query(self, coordinator):
        long_query = "测" * 2001
        result = await coordinator.process_query(long_query, "test_session")
        assert "query too long" in result.metadata["error"]

class TestTaskGeneration:
    """任务生成测试"""
    
    @pytest.mark.asyncio
    async def test_minimum_tasks_generation(self, coordinator):
        coordinator.query_parser.parse_query.return_value = [
            SubTask(
                task_type="search",
                description="单个任务",
                priority=1,
                parameters={"depth": "detailed"}
            )
        ]
        
        await coordinator.process_query("测试查询", "test_session")
        
        args, _ = coordinator.tool_orchestrator.execute_tasks.call_args
        tasks = args[0]
        assert len(tasks) >= coordinator.config.min_tasks

    @pytest.mark.asyncio
    async def test_task_parameters_default(self, coordinator):
        coordinator.query_parser.parse_query.return_value = [
            SubTask(
                task_type="search",
                description="测试任务",
                priority=1,
                parameters={}  # 添加空的 parameters 字段
            )
        ]
        
        await coordinator.process_query("测试查询", "test_session")
        
        args, _ = coordinator.tool_orchestrator.execute_tasks.call_args
        task = args[0][0]
        assert task.parameters == {"depth": "detailed"}

class TestTaskExecution:
    """任务执行测试"""
    
    @pytest.mark.asyncio
    async def test_task_execution_with_model_info(self, coordinator):
        coordinator.query_parser.parse_query.return_value = [
            SubTask(
                task_type="search",
                description="测试任务",
                priority=1,
                parameters={"depth": "detailed"}
            )
        ]
        
        task_result = {
            "result": "任务执行结果",
            "task_type": "search",
            "model": {"name": "text-embedding-3-large"},
            "token_usage": TokenUsage(
                prompt_tokens=100,
                completion_tokens=50,
                total_tokens=150
            )
        }
        coordinator.tool_orchestrator.execute_tasks = AsyncMock(return_value=[task_result])
        
        await coordinator.process_query("测试查询", "test_session")
        
        # 验证成本追踪
        cost_call = coordinator.cost_tracker.track_usage.call_args_list[0]
        assert cost_call[1]["model"] == "text-embedding-3-large"
        assert cost_call[1]["task_type"] == "search"

    @pytest.mark.asyncio
    async def test_task_execution_error(self, coordinator):
        coordinator.query_parser.parse_query.return_value = [
            SubTask(
                task_type="search",
                description="测试任务",
                priority=1,
                parameters={"depth": "detailed"}
            )
        ]
        coordinator.tool_orchestrator.execute_tasks = AsyncMock(
            side_effect=Exception("任务执行失败")
        )
        
        result = await coordinator.process_query("测试查询", "test_session")
        assert "任务执行失败" in result.metadata["error"]

class TestResponseGeneration:
    """响应生成测试"""
    
    @pytest.mark.asyncio
    async def test_response_generation_with_model_info(self, coordinator):
        coordinator.query_parser.parse_query.return_value = [
            SubTask(
                task_type="search",
                description="测试任务",
                priority=1,
                parameters={"depth": "detailed"}  # 添加必需的 parameters 字段
            )
        ]
        coordinator.tool_orchestrator.execute_tasks = AsyncMock(return_value=["任务结果"])
        
        response_data = {
            "response": "生成的回答",
            "model": {"name": "gpt-4"},
            "token_usage": TokenUsage(
                prompt_tokens=200,
                completion_tokens=100,
                total_tokens=300
            )
        }
        coordinator.tool_orchestrator.generate_response = AsyncMock(return_value=response_data)
        
        await coordinator.process_query("测试查询", "test_session")
        
        response_cost_call = coordinator.cost_tracker.track_usage.call_args_list[-1]
        assert response_cost_call[1]["model"] == "gpt-4"
        assert response_cost_call[1]["task_type"] == "response_generation"

    @pytest.mark.asyncio
    async def test_response_generation_with_string_response(self, coordinator):
        # 设置任务执行结果
        task_result = {
            "result": "任务结果",
            "task_type": "search",
            "model": {"name": "gpt-4"},
            "token_usage": TokenUsage(
                prompt_tokens=100,
                completion_tokens=50,
                total_tokens=150
            )
        }
        
        coordinator.query_parser.parse_query.return_value = [
            SubTask(
                task_type="search",
                description="测试任务",
                priority=1,
                parameters={"depth": "detailed"}
            )
        ]
        coordinator.tool_orchestrator.execute_tasks = AsyncMock(return_value=[task_result])
        coordinator.tool_orchestrator.generate_response = AsyncMock(return_value={
            "response": "字符串响应",
            "model": {"name": "gpt-4"},
            "token_usage": None
        })
        coordinator.result_evaluator.evaluate_with_fallback = AsyncMock(
            return_value={"quality_score": 0.8, "used_fallback": False}
        )
        
        result = await coordinator.process_query("测试查询", "test_session")
        assert result.final_answer == "字符串响应"

    @pytest.mark.asyncio
    async def test_quality_evaluation_success(self, coordinator):
        # 设置完整的响应数据
        task_result = {
            "result": "任务结果",
            "task_type": "search",
            "model": {"name": "gpt-4"},
            "token_usage": TokenUsage(
                prompt_tokens=100,
                completion_tokens=50,
                total_tokens=150
            )
        }
        response_data = {
            "response": "生成的回答",
            "model": {"name": "gpt-4"},
            "token_usage": None
        }
        
        coordinator.query_parser.parse_query.return_value = [
            SubTask(
                task_type="search",
                description="测试任务",
                priority=1,
                parameters={"depth": "detailed"}
            )
        ]
        coordinator.tool_orchestrator.execute_tasks = AsyncMock(return_value=[task_result])
        coordinator.tool_orchestrator.generate_response = AsyncMock(return_value=response_data)
        coordinator.result_evaluator.evaluate_with_fallback = AsyncMock(
            return_value={"quality_score": 0.8, "used_fallback": False}
        )
        
        result = await coordinator.process_query("测试查询", "test_session")
        assert result.evaluation_result["quality_score"] == 0.8
        assert not result.metadata["used_fallback"]

    @pytest.mark.asyncio
    async def test_quality_evaluation_error(self, coordinator):
        task_result = {
            "result": "任务结果",
            "task_type": "search",
            "model": {"name": "gpt-4"},
            "token_usage": TokenUsage(
                prompt_tokens=100,
                completion_tokens=50,
                total_tokens=150
            )
        }
        response_data = {
            "response": "生成的回答",
            "model": {"name": "gpt-4"},
            "token_usage": None
        }
        
        coordinator.query_parser.parse_query.return_value = [
            SubTask(
                task_type="search",
                description="测试任务",
                priority=1,
                parameters={"depth": "detailed"}
            )
        ]
        coordinator.tool_orchestrator.execute_tasks = AsyncMock(return_value=[task_result])
        coordinator.tool_orchestrator.generate_response = AsyncMock(return_value=response_data)
        coordinator.result_evaluator.evaluate_with_fallback = AsyncMock(
            side_effect=Exception("评估失败")
        )
        
        result = await coordinator.process_query("测试查询", "test_session")
        assert "评估失败" in result.metadata["error"]

    @pytest.mark.asyncio
    async def test_conversation_save_error(self, coordinator):
        task_result = {
            "result": "任务结果",
            "task_type": "search",
            "model": {"name": "gpt-4"},
            "token_usage": TokenUsage(
                prompt_tokens=100,
                completion_tokens=50,
                total_tokens=150
            )
        }
        response_data = {
            "response": "生成的回答",
            "model": {"name": "gpt-4"},
            "token_usage": None
        }
        
        coordinator.query_parser.parse_query.return_value = [
            SubTask(
                task_type="search",
                description="测试任务",
                priority=1,
                parameters={"depth": "detailed"}
            )
        ]
        coordinator.tool_orchestrator.execute_tasks = AsyncMock(return_value=[task_result])
        coordinator.tool_orchestrator.generate_response = AsyncMock(return_value=response_data)
        coordinator.conversation_manager.add_message.side_effect = Exception("保存失败")
        
        result = await coordinator.process_query("测试查询", "test_session")
        assert "保存失败" in result.metadata["error"]

class TestEndToEnd:
    """端到端测试"""
    
    @pytest.mark.asyncio
    async def test_successful_workflow(self, coordinator):
        coordinator.query_parser.parse_query.return_value = [
            SubTask(
                task_type="search",
                description="测试任务",
                priority=1,
                parameters={"depth": "detailed"}  # 添加必需的 parameters 字段
            )
        ]
        coordinator.tool_orchestrator.execute_tasks = AsyncMock(return_value=["任务结果"])
        coordinator.tool_orchestrator.generate_response = AsyncMock(return_value={
            "response": "生成的回答",
            "model": {"name": "gpt-4"},
            "token_usage": TokenUsage(
                prompt_tokens=100,
                completion_tokens=50,
                total_tokens=150
            )
        })
        coordinator.result_evaluator.evaluate_with_fallback = AsyncMock(
            return_value={"quality_score": 0.9}
        )
        
        result = await coordinator.process_query("测试查询", "test_session")
        
        assert result.final_answer == "生成的回答"
        assert result.evaluation_result["quality_score"] == 0.9
        assert "error" not in result.metadata
        
        # 验证调用顺序
        coordinator.query_parser.parse_query.assert_called_once()
        coordinator.tool_orchestrator.execute_tasks.assert_called_once()
        coordinator.tool_orchestrator.generate_response.assert_called_once()
        coordinator.result_evaluator.evaluate_with_fallback.assert_called_once()
        coordinator.conversation_manager.add_message.assert_called_once()