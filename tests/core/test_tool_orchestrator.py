import pytest
from unittest.mock import Mock, patch, AsyncMock
import json  # 确保导入 json
from typing import List, Dict
from engine.core.tool_orchestrator import ToolOrchestrator
from engine.core.query_parser import SubTask
from langchain.schema import AIMessage

@pytest.fixture
def orchestrator():
    orchestrator = ToolOrchestrator()
    orchestrator.llm = AsyncMock()
    orchestrator.agent_executor = AsyncMock()
    orchestrator.query_parser = Mock()
    
    mock_response = AIMessage(content="测试回答")
    orchestrator.llm.ainvoke.return_value = mock_response
    return orchestrator, mock_response

@pytest.mark.asyncio
async def test_execute_tasks_success(orchestrator):
    orchestrator, _ = orchestrator
    tasks = [
        SubTask(task_type="search", description="搜索任务", priority=1, parameters={"depth": "detailed"}),
        SubTask(task_type="analysis", description="分析任务", priority=2, parameters={"method": "comprehensive"})
    ]
    
    orchestrator.agent_executor.ainvoke.side_effect = [
        {"output": "搜索结果"},
        {"output": "分析结果"}
    ]
    
    results = await orchestrator.execute_tasks(tasks)
    
    assert len(results) == 2
    assert results[0]["task_type"] == "search"
    assert results[1]["task_type"] == "analysis"
    assert orchestrator.agent_executor.ainvoke.call_count == 2

@pytest.mark.asyncio
async def test_generate_response_direct_answer(orchestrator):
    orchestrator, mock_response = orchestrator
    query = "测试查询"
    task_results = [{"task_type": "search", "result": {"output": "测试结果"}}]
    
    result = await orchestrator.generate_response(query, task_results)
    
    assert result == "测试回答"
    orchestrator.llm.ainvoke.assert_called_once()

@pytest.mark.asyncio
async def test_generate_response_with_additional_info(orchestrator):
    orchestrator, _ = orchestrator
    query = "测试查询"
    initial_results = [{"task_type": "search", "result": {"output": "初始结果"}}]
    new_tasks = [
        SubTask(task_type="search", description="新任务", priority=1, parameters={})
    ]
    
    orchestrator.llm.ainvoke.side_effect = [
        AIMessage(content="需要更多信息：查询更多数据"),
        AIMessage(content="最终答案")
    ]
    
    orchestrator.query_parser.parse_query.return_value = new_tasks
    orchestrator.agent_executor.ainvoke.return_value = {"output": "新的结果"}
    
    result = await orchestrator.generate_response(query, initial_results)
    
    assert result == "最终答案"
    assert orchestrator.llm.ainvoke.call_count == 2

@pytest.mark.asyncio
async def test_generate_response_max_iterations(orchestrator):
    orchestrator, _ = orchestrator
    query = "测试查询"
    task_results = [{"task_type": "search", "result": {"output": "初始结果"}}]
    
    orchestrator.llm.ainvoke.return_value = AIMessage(content="需要更多信息：继续查询")
    orchestrator.query_parser.parse_query.return_value = [
        SubTask(task_type="search", description="新任务", priority=1, parameters={})
    ]
    orchestrator.agent_executor.ainvoke.return_value = {"output": "新的结果"}
    
    result = await orchestrator.generate_response(query, task_results)
    
    assert "需要更多信息" in result
    assert orchestrator.llm.ainvoke.call_count == 3

@pytest.mark.asyncio
async def test_generate_response_empty_results(orchestrator):
    orchestrator, mock_response = orchestrator
    query = "测试查询"
    task_results = []
    
    mock_response.content = "没有找到相关信息"
    
    result = await orchestrator.generate_response(query, task_results)
    
    assert result == "没有找到相关信息"
    orchestrator.llm.ainvoke.assert_called_once()