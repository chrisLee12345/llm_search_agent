import asyncio
from engine.core.result_evaluator import ResultEvaluator  # 移除 'src.' 前缀
from langchain.schema import Document  # 添加这行导入

async def test_result_evaluator():
    evaluator = ResultEvaluator()

    # 测试场景1：高质量答案
    test_context = [
        Document(page_content="人工智能是计算机科学的一个分支，致力于开发能够模拟人类智能的系统。"),
        Document(page_content="机器学习是人工智能的核心技术之一，它使计算机能够从数据中学习并改进。")
    ]
    test_answer = "人工智能是计算机科学的重要分支，其核心技术是机器学习，通过数据学习来模拟人类智能。"

    print("测试场景1：评估高质量答案")
    result1 = await evaluator.evaluate_with_fallback(
        answer=test_answer,
        context=test_context,
        query="什么是人工智能？"
    )
    print(f"评估结果：{result1}\n")
    
    # 测试场景2：低质量答案（需要触发备用搜索）
    test_answer_poor = "人工智能可以取代所有人类工作，并且已经具备完全的自主意识。"
    
    print("测试场景2：评估存在幻觉的答案")
    result2 = await evaluator.evaluate_with_fallback(
        answer=test_answer_poor,
        context=test_context,
        query="人工智能的能力如何？"
    )
    print(f"评估结果：{result2}\n")
    
    # 测试场景3：完全不相关的答案
    test_answer_irrelevant = "今天天气很好，适合出去散步。"
    
    print("测试场景3：评估不相关答案")
    result3 = await evaluator.evaluate_with_fallback(
        answer=test_answer_irrelevant,
        context=test_context,
        query="什么是人工智能？"
    )
    print(f"评估结果：{result3}")

if __name__ == "__main__":
    asyncio.run(test_result_evaluator())