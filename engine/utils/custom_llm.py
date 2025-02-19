from typing import Any, List, Optional
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from openai import AzureOpenAI
import os

class AzureGPT4LLM(LLM):  # 更改类名
    client: Any = None
    
    def __init__(self):
        super().__init__()
        self.client = AzureOpenAI(
            api_key="DuyZ0hHrPDcKN5FswiEyi43Reqqu2qLulqfQMglca8xDYA79URG3JQQJ99BAACYeBjFXJ3w3AAABACOGvlW6",
            api_version="2024-08-01-preview",
            azure_endpoint="https://ai-search-gpt4.openai.azure.com"
        )
    
    @property
    def _llm_type(self) -> str:
        return "azure_gpt4"  # 更新返回值
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            max_tokens=500,
            stream=False
        )
        return response.choices[0].message.content