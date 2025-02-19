import pytest
import os
import sys

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 设置异步测试的默认配置
pytest_plugins = ["pytest_asyncio"]

# 如果需要 mock 某些外部服务，可以在这里添加 fixture
@pytest.fixture
def mock_bing_search(monkeypatch):
    async def mock_search(*args, **kwargs):
        return [{
            "source": "mock",
            "content": "这是测试数据",
            "url": "http://example.com",
            "relevance_score": 0.9
        }]
    
    from llm_search.core.fallback_search import FallbackSearchEngine
    monkeypatch.setattr(FallbackSearchEngine, "fallback_search", mock_search)