from typing import Dict
from pydantic import BaseModel, Field

class Config(BaseModel):
    """全局配置"""
    models: Dict[str, str] = Field(
        default={
            "search": "text-embedding-3-large",
            "query_parsing": "gpt-4o",
            "task_execution": "gpt-4o",
            "response_generation": "gpt-4o",
            "text_embedding": "text-embedding-3-large"
        },
        description="各任务类型的默认模型"
    )