import os
import json
from pprint import pprint
import requests
import asyncio
import aiohttp

async def test_bing_search_async():
    """测试 Bing 搜索的异步调用"""
    subscription_key = "31dae6e97e1545218c4b0b950b419e11"
    endpoint = "https://api.bing.microsoft.com/v7.0/search"
    
    # 测试查询
    query = "人工智能"
    
    # 构造请求参数，移除布尔值参数
    params = {
        'q': query,
        'mkt': 'zh-CN',
        'count': 3,
        'textFormat': 'HTML'
    }
    headers = {'Ocp-Apim-Subscription-Key': subscription_key}
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(endpoint, headers=headers, params=params) as response:
                if response.status == 200:
                    result = await response.json()
                    print("\nBing 搜索结果:")
                    if "webPages" in result and "value" in result["webPages"]:
                        for item in result["webPages"]["value"]:
                            print(f"\n标题: {item['name']}")
                            print(f"URL: {item['url']}")
                            print(f"摘要: {item['snippet']}\n")
                    else:
                        print("未找到搜索结果")
                else:
                    print(f"API 调用失败: {response.status}")
                    
    except Exception as e:
        print(f"发生错误: {str(e)}")

def test_bing_search_sync():
    """测试 Bing 搜索的同步调用"""
    subscription_key = "31dae6e97e1545218c4b0b950b419e11"
    endpoint = "https://api.bing.microsoft.com/v7.0/search"
    
    query = "人工智能"
    params = {
        'q': query,
        'mkt': 'zh-CN',
        'count': 3
    }
    headers = {'Ocp-Apim-Subscription-Key': subscription_key}
    
    try:
        response = requests.get(endpoint, headers=headers, params=params)
        response.raise_for_status()
        
        print("\nHeaders:")
        print(response.headers)
        
        print("\nJSON Response:")
        pprint(response.json())
        
    except Exception as e:
        print(f"发生错误: {str(e)}")

if __name__ == "__main__":
    print("开始测试 Bing 搜索 API...")
    
    # 测试同步调用
    print("\n=== 同步调用测试 ===")
    test_bing_search_sync()
    
    # 测试异步调用
    print("\n=== 异步调用测试 ===")
    asyncio.run(test_bing_search_async())