from typing import List, Dict, Optional
from pydantic import BaseModel
from langchain.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI
from langchain.docstore.document import Document
from ..web.apiconfig import config

class SubTask(BaseModel):
    task_type: str  # 任务类型：doc_qa, db_query, calculation, analysis
    description: str  # 任务描述
    parameters: Dict  # 任务参数
    priority: int  # 任务优先级

class QueryParser:
    def __init__(self):
        self.llm = AzureChatOpenAI(
            azure_endpoint=config.api.azure_openai["azure_endpoint"],
            api_key=config.api.azure_openai["api_key"],
            api_version=config.api.azure_openai["api_version"],
            model=config.api.azure_openai["model"],
            temperature=0
        )
        
        self.task_prompt = ChatPromptTemplate.from_messages([
            ("system", "你是一个专业的查询解析器。你需要将用户的查询分解为具体的子任务。"),
            ("user", "{query}"),
            ("system", "请将查询分解为子任务，并以JSON格式返回，包含task_type、description、parameters和priority字段。")
        ])
        
        self.answer_prompt = ChatPromptTemplate.from_messages([
            ("system", "你是一个专业的问答助手。请基于提供的上下文信息，生成准确、完整的回答。如果上下文中没有相关信息，请明确指出。"),
            ("user", "问题：{query}\n\n上下文信息：\n{context}")
        ])
    
    def parse_query(self, query: str) -> List[SubTask]:
        """解析用户查询并分解为子任务"""
        response = self.llm.invoke(self.task_prompt.format(query=query))
        tasks_dict = eval(response.content)  # 将JSON字符串转换为Python对象
        return [SubTask(**task) for task in tasks_dict["tasks"]]
    
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