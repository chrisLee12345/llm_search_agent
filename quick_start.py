from engine.core.workflow_coordinator import WorkflowCoordinator
import asyncio

async def analyze_query(query: str) -> None:
    # 创建工作流协调器
    coordinator = WorkflowCoordinator()
    
    # 生成会话ID
    session_id = "test_session"
    
    # 处理查询
    result = await coordinator.process_query(query, session_id)
    
    # 打印结果
    print("\n=== 分析结果 ===")
    print(f"答案：{result.final_answer}\n")
    print("=== 子任务执行情况 ===")
    for task in result.subtasks:
        print(f"- {task['description']}")
    print(f"\n质量评分：{result.evaluation_result.get('quality_score', 'N/A')}")

# 使用示例
if __name__ == "__main__":
    # 示例查询
    query = "分析中国新能源汽车行业的发展趋势"
    
    # 运行分析
    asyncio.run(analyze_query(query))