import unittest
from unittest.mock import Mock, patch
from engine.core.query_parser import QueryParser, SubTask
from pydantic import ValidationError  # 修改导入语句

class TestQueryParser(unittest.TestCase):
    def setUp(self):
        self.parser = QueryParser()
        self.parser.llm = Mock()
        self.parser.task_prompt = "分析查询：{query}"

    def test_parse_query_normal(self):
        # 准备测试数据
        mock_response = Mock()
        mock_response.content = """{"tasks": [
            {"task_type": "search", "description": "分析当前状况", "priority": 1, "parameters": {"depth": "detailed"}},
            {"task_type": "search", "description": "研究历史背景", "priority": 2, "parameters": {"depth": "normal"}}
        ]}"""
        self.parser.llm.invoke.return_value = mock_response

        # 执行测试
        result = self.parser.parse_query("人工智能发展")

        # 验证结果
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], SubTask)
        self.assertEqual(result[0].task_type, "search")
        self.assertEqual(result[0].description, "分析当前状况")
        self.assertEqual(result[0].priority, 1)
        self.assertEqual(result[0].parameters["depth"], "detailed")

    def test_parse_query_empty_tasks(self):
        # 准备测试数据
        mock_response = Mock()
        mock_response.content = """{"tasks": []}"""
        self.parser.llm.invoke.return_value = mock_response

        # 执行测试
        result = self.parser.parse_query("测试查询")

        # 验证结果
        self.assertEqual(len(result), 0)

    def test_parse_query_invalid_json(self):
        # 准备测试数据
        mock_response = Mock()
        mock_response.content = "invalid json"
        self.parser.llm.invoke.return_value = mock_response

        # 验证异常
        with self.assertRaises(SyntaxError):
            self.parser.parse_query("测试查询")

    def test_parse_query_missing_tasks_key(self):
        # 准备测试数据
        mock_response = Mock()
        mock_response.content = """{"other": []}"""
        self.parser.llm.invoke.return_value = mock_response

        # 验证异常
        with self.assertRaises(KeyError):
            self.parser.parse_query("测试查询")

    def test_parse_query_invalid_task_format(self):
        # 准备测试数据
        mock_response = Mock()
        mock_response.content = """{"tasks": [
            {"invalid_field": "value"}
        ]}"""
        self.parser.llm.invoke.return_value = mock_response

        # 验证异常
        with self.assertRaises(ValidationError):  # 修改异常类型
            self.parser.parse_query("测试查询")

if __name__ == '__main__':
    unittest.main()