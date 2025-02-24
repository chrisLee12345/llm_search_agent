from typing import Dict
import os
from pydantic import BaseModel

class APIConfig(BaseModel):
    """API 配置类"""
    azure_openai: Dict[str, str] = {
        "api_key": "enter your api key",
        "api_version": "2024-08-01-preview",
        "azure_endpoint": "enter your endpoint",
        "model": "enter your model name"
    }
    
    bing_search: Dict[str, str] = {
        "api_key": "your_bing_api_key",
        "endpoint": "https://api.bing.microsoft.com/v7.0/search"
    }
    
    embedding: Dict[str, str] = {
        "azure_endpoint": "enter your endpoint",
        "api_key": "enter your api key",
        "api_version": "2024-02-15-preview",
        "model": "text-embedding-3-large"
    }

class Config:
    """全局配置类"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_config()
        return cls._instance
    
    def _load_config(self):
        """加载配置"""
        self.api = APIConfig()
        
        # 从环境变量加载配置（如果存在）
        if os.getenv("AZURE_OPENAI_API_KEY"):
            self.api.azure_openai["api_key"] = os.getenv("AZURE_OPENAI_API_KEY")
            self.api.embedding["api_key"] = os.getenv("AZURE_OPENAI_API_KEY")
        if os.getenv("AZURE_OPENAI_ENDPOINT"):
            self.api.azure_openai["azure_endpoint"] = os.getenv("AZURE_OPENAI_ENDPOINT")
            self.api.embedding["azure_endpoint"] = os.getenv("AZURE_OPENAI_ENDPOINT")
        
        if os.getenv("BING_API_KEY"):
            self.api.bing_search["api_key"] = os.getenv("BING_API_KEY")

# 全局配置实例
config = Config()
