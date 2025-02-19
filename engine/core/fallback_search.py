from typing import List, Dict, Optional
from pydantic import BaseModel
import aiohttp
import json
from ..web.apiconfig import config

class SearchResult(BaseModel):
    source: str
    content: str
    url: Optional[str] = None
    relevance_score: Optional[float] = None

class FallbackSearchEngine:
    def __init__(self):
        self.subscription_key = config.api.bing_search["api_key"]
        self.endpoint = config.api.bing_search["endpoint"]
    
    async def fallback_search(self, 
                            query: str, 
                            max_results: int = 5,
                            min_relevance_score: float = 0.7) -> List[SearchResult]:
        """执行备用网页搜索"""
        params = {
            'q': query,
            'mkt': 'zh-CN' if any('\u4e00-\u9fff' in char for char in query) else 'en-US',
            'count': max_results,
            'textFormat': 'HTML'
        }
        headers = {'Ocp-Apim-Subscription-Key': self.subscription_key}
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.endpoint, headers=headers, params=params) as response:
                    if response.status == 200:
                        result = await response.json()
                        if "webPages" in result and "value" in result["webPages"]:
                            filtered_results = []
                            for item in result["webPages"]["value"]:
                                relevance_score = self._calculate_relevance(query, item["snippet"])
                                if relevance_score >= min_relevance_score:
                                    filtered_results.append(SearchResult(
                                        source="bing",
                                        content=item["snippet"],
                                        url=item["url"],
                                        relevance_score=relevance_score
                                    ))
                            return filtered_results
                    return []
                    
        except Exception as e:
            print(f"Bing 搜索错误: {str(e)}")
            return []
    
    def _calculate_relevance(self, query: str, content: str) -> float:
        """计算搜索结果与查询的相关性分数"""
        query_words = set(query.lower().split())
        content_words = set(content.lower().split())
        overlap = len(query_words.intersection(content_words))
        return min(1.0, overlap / len(query_words) + 0.3)  # 基础分 0.3