import unittest
from unittest.mock import Mock, patch
from typing import List
from dataclasses import dataclass

@dataclass
class Document:
    page_content: str
    metadata: dict = None

class AnswerGenerator:
    def __init__(self):
        self.llm = None
        self.answer_prompt = None

    def generate_answer(self, query: str, context: List[Document]) -> str:
        """根据上下文生成答案"""
        # 从 Document 对象中提取文本内容
        context_text = "\n".join([doc.page_content for doc in context])
        
        response = self.llm.invoke(
            self.answer_prompt.format(
                query=query,
                context=context_text
            )
        )
        return response.content

class TestAnswerGenerator(unittest.TestCase):
    def setUp(self):
        self.generator = AnswerGenerator()
        self.generator.llm = Mock()
        self.generator.answer_prompt = "{query}\n\n上下文：\n{context}\n\n请基于上述上下文回答问题。"
        
        # 模拟 LLM 响应对象
        self.mock_response = Mock()
        self.mock_response.content = "这是生成的答案"
        self.generator.llm.invoke.return_value = self.mock_response

    def test_generate_answer_with_single_document(self):
        # 准备测试数据
        query = "测试问题"
        context = [Document(page_content="测试文档内容")]
        
        # 执行测试
        result = self.generator.generate_answer(query, context)
        
        # 验证结果
        self.assertEqual(result, "这是生成的答案")
        self.generator.llm.invoke.assert_called_once_with(
            self.generator.answer_prompt.format(
                query=query,
                context="测试文档内容"
            )
        )

    def test_generate_answer_with_multiple_documents(self):
        # 准备测试数据
        query = "测试问题"
        context = [
            Document(page_content="文档1内容"),
            Document(page_content="文档2内容"),
            Document(page_content="文档3内容")
        ]
        
        # 执行测试
        result = self.generator.generate_answer(query, context)
        
        # 验证结果
        self.assertEqual(result, "这是生成的答案")
        self.generator.llm.invoke.assert_called_once_with(
            self.generator.answer_prompt.format(
                query=query,
                context="文档1内容\n文档2内容\n文档3内容"
            )
        )

    def test_generate_answer_with_empty_context(self):
        # 准备测试数据
        query = "测试问题"
        context = []
        
        # 执行测试
        result = self.generator.generate_answer(query, context)
        
        # 验证结果
        self.assertEqual(result, "这是生成的答案")
        self.generator.llm.invoke.assert_called_once_with(
            self.generator.answer_prompt.format(
                query=query,
                context=""
            )
        )

    def test_generate_answer_with_empty_content(self):
        # 准备测试数据
        query = "测试问题"
        context = [Document(page_content="")]
        
        # 执行测试
        result = self.generator.generate_answer(query, context)
        
        # 验证结果
        self.assertEqual(result, "这是生成的答案")
        self.generator.llm.invoke.assert_called_once_with(
            self.generator.answer_prompt.format(
                query=query,
                context=""
            )
        )

if __name__ == '__main__':
    unittest.main()