import unittest
from unittest.mock import Mock, AsyncMock, patch
from typing import List, Dict
from dataclasses import dataclass
from engine.core.result_evaluator import ResultEvaluator

@dataclass
class WebResult:
    content: str
    url: str

@dataclass
class EvalResult:
    score: float
    hallucination_risk: float

class TestResultEvaluator(unittest.TestCase):
    def setUp(self):
        self.evaluator = ResultEvaluator()
        self.evaluator.evaluate = Mock()
        self.evaluator.fallback_search = Mock()
        self.evaluator._regenerate_answer = AsyncMock()

    async def test_evaluate_with_fallback_good_quality(self):
        # 准备测试数据
        answer = "这是一个高质量的答案"
        context = ["上下文1", "上下文2"]
        query = "测试查询"
        
        # 设置评估结果为高质量
        self.evaluator.evaluate.return_value = EvalResult(
            score=0.8,
            hallucination_risk=0.1
        )
        
        # 执行测试
        result = await self.evaluator.evaluate_with_fallback(
            answer=answer,
            context=context,
            query=query
        )
        
        # 验证结果
        self.assertEqual(result["answer"], answer)
        self.assertEqual(result["used_fallback"], False)
        self.assertEqual(result["web_sources"], [])
        self.evaluator.fallback_search.fallback_search.assert_not_called()
        self.evaluator._regenerate_answer.assert_not_called()

    async def test_evaluate_with_fallback_low_score(self):
        # 准备测试数据
        answer = "这是一个低质量的答案"
        context = ["上下文1", "上下文2"]
        query = "测试查询"
        
        # 设置初始评估结果为低质量
        self.evaluator.evaluate.side_effect = [
            EvalResult(score=0.5, hallucination_risk=0.2),  # 第一次评估
            EvalResult(score=0.9, hallucination_risk=0.1)   # 第二次评估
        ]
        
        # 设置备用搜索结果
        web_results = [
            WebResult(content="网页内容1", url="http://test1.com"),
            WebResult(content="网页内容2", url="http://test2.com")
        ]
        self.evaluator.fallback_search.fallback_search = AsyncMock(
            return_value=web_results
        )
        
        # 设置重新生成的答案
        new_answer = "这是重新生成的高质量答案"
        self.evaluator._regenerate_answer.return_value = new_answer
        
        # 执行测试
        result = await self.evaluator.evaluate_with_fallback(
            answer=answer,
            context=context,
            query=query
        )
        
        # 验证结果
        self.assertEqual(result["answer"], new_answer)
        self.assertEqual(result["used_fallback"], True)
        self.assertEqual(result["web_sources"], 
                        ["http://test1.com", "http://test2.com"])
        
        # 验证方法调用
        self.evaluator.fallback_search.fallback_search.assert_called_once_with(query)
        self.evaluator._regenerate_answer.assert_called_once_with(
            query, 
            context + ["网页内容1", "网页内容2"]
        )

    async def test_evaluate_with_fallback_high_hallucination(self):
        # 准备测试数据
        answer = "这是一个幻觉风险高的答案"
        context = ["上下文1", "上下文2"]
        query = "测试查询"
        
        # 设置评估结果为高幻觉风险
        self.evaluator.evaluate.side_effect = [
            EvalResult(score=0.8, hallucination_risk=0.4),  # 第一次评估
            EvalResult(score=0.9, hallucination_risk=0.1)   # 第二次评估
        ]
        
        # 设置备用搜索结果
        web_results = [
            WebResult(content="网页内容1", url="http://test1.com")
        ]
        self.evaluator.fallback_search.fallback_search = AsyncMock(
            return_value=web_results
        )
        
        # 设置重新生成的答案
        new_answer = "这是重新生成的低幻觉风险答案"
        self.evaluator._regenerate_answer.return_value = new_answer
        
        # 执行测试
        result = await self.evaluator.evaluate_with_fallback(
            answer=answer,
            context=context,
            query=query
        )
        
        # 验证结果
        self.assertEqual(result["answer"], new_answer)
        self.assertEqual(result["used_fallback"], True)
        self.assertEqual(result["web_sources"], ["http://test1.com"])
        
        # 验证方法调用
        self.evaluator.fallback_search.fallback_search.assert_called_once_with(query)
        self.evaluator._regenerate_answer.assert_called_once_with(
            query, 
            context + ["网页内容1"]
        )

if __name__ == '__main__':
    unittest.main()