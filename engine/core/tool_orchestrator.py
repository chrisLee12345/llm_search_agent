from typing import List, Dict, Any
import json
from .query_parser import SubTask, QueryParser  # 添加 QueryParser 导入
from ..indexer.document_store import DocumentStore
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain_openai import AzureChatOpenAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from pydantic import BaseModel
from ..web.apiconfig import config
import tiktoken  # 添加 tiktoken 导入

class ToolOrchestrator:
    def __init__(self):
        # 初始化基本组件
        self.response_prompt = ChatPromptTemplate.from_messages([
            {"role": "system", "content": "基于提供的上下文信息生成回答。"},
            {"role": "user", "content": "问题：{query}\n\n子任务结果：{results}"}
        ])
        self.doc_store = DocumentStore()
        self.llm = AzureChatOpenAI(
            azure_endpoint=config.api.azure_openai["azure_endpoint"],
            api_key=config.api.azure_openai["api_key"],
            api_version=config.api.azure_openai["api_version"],
            model=config.api.azure_openai["model"]
        )
        
        # 初始化 tokenizer
        try:
            self.tokenizer = tiktoken.encoding_for_model(config.api.azure_openai["model"])
        except Exception as e:
            print(f"初始化 tokenizer 失败: {e}")
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        # 初始化工具和代理
        self.query_parser = QueryParser()
        
        # 创建同步版本的文档搜索函数
        async def sync_document_search(query: str):
            try:
                results = await self.doc_store.search(query)
                return [doc.page_content for doc in results]
            except Exception as e:
                print(f"文档搜索错误: {e}")
                return []
        
        self.tools = [
            Tool(
                name="document_search",
                func=sync_document_search,
                description="搜索文档知识库，返回相关文档内容",
                coroutine=sync_document_search
            )
        ]
        
        # 创建 React 代理
        react_prompt = PromptTemplate.from_template("""Answer the following questions as best you can...

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought: {agent_scratchpad}""")
        
        # 初始化代理执行器
        self.agent_executor = AgentExecutor(
            agent=create_react_agent(
                llm=self.llm,
                tools=self.tools,
                prompt=react_prompt
            ),
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=3
        )

    def get_model_for_task(self, task_type: str) -> str:
        """获取任务类型对应的模型名称"""
        model_mapping = {
            "search": config.api.azure_openai["model"],
            "analyze": config.api.azure_openai["model"],
            "default": config.api.azure_openai["model"]
        }
        return model_mapping.get(task_type, model_mapping["default"])

    # 删除以下重复的代码块:
    # - 重复的 query_parser 初始化
    # - 重复的 tools 定义
    # - 重复的 react_prompt 定义
    # - 重复的 agent_executor 初始化
        
        # 初始化 query_parser
        self.query_parser = QueryParser()
        
        self.tools = [
            Tool(
                name="document_search",
                func=self.doc_store.search,
                description="搜索文档知识库"
            ),
            # 可以添加更多工具，如数据库查询、计算工具等
        ]
        
        # 添加 React 代理的提示模板
        react_prompt = PromptTemplate.from_template("""Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought: {agent_scratchpad}""")
        
        # 修改 React 代理的创建方式
        self.agent_executor = AgentExecutor(
            agent=create_react_agent(
                llm=self.llm,
                tools=self.tools,
                prompt=react_prompt
            ),
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=3
        )

    def count_tokens(self, text: str) -> int:
        """计算文本的 token 数量"""
        return len(self.tokenizer.encode(text))

    async def execute_tasks(self, tasks: List[SubTask]) -> List[Dict[str, Any]]:
        """执行任务列表并返回结果"""
        results = []
        for task in sorted(tasks, key=lambda x: x.priority):
            try:
                # 使用 Agent 执行任务
                result = await self.agent_executor.ainvoke({
                    "input": task.description
                })
                print("Agent 响应:", json.dumps(result, ensure_ascii=False, indent=2))
                
                # 处理响应
                if isinstance(result, dict):
                    output = result.get("output", "")
                    # 使用 tiktoken 计算 token 使用量
                    input_tokens = self.count_tokens(task.description)
                    output_tokens = self.count_tokens(output)
                    
                    token_usage = {
                        "completion_tokens": output_tokens,
                        "total_tokens": input_tokens + output_tokens
                    }
                else:
                    output = str(result)
                    input_tokens = self.count_tokens(task.description)
                    output_tokens = self.count_tokens(output)
                    token_usage = {
                        "completion_tokens": output_tokens,
                        "total_tokens": input_tokens + output_tokens
                    }
                
                results.append({
                    "task_type": task.task_type,
                    "result": output,
                    "token_usage": token_usage
                })
            except Exception as e:
                print(f"任务执行错误: {str(e)}")
                results.append({
                    "task_type": task.task_type,
                    "result": f"执行失败: {str(e)}",
                    "token_usage": {
                        "completion_tokens": 0,
                        "total_tokens": 0
                    }
                })
        return results

    async def generate_response(self, query: str, task_results: List[Dict]) -> Dict:
        max_iterations = 3
        current_iteration = 0
        
        while current_iteration < max_iterations:
            try:
                # 使用 ChatPromptTemplate 格式化消息
                messages = self.response_prompt.format_messages(
                    query=query,
                    results=json.dumps(task_results, ensure_ascii=False)
                )
                
                response = await self.llm.ainvoke(messages)
                response_content = response.content if hasattr(response, 'content') else str(response)
                
                # 计算 token 使用量
                input_text = query + json.dumps(task_results, ensure_ascii=False)
                input_tokens = self.count_tokens(input_text)
                output_tokens = self.count_tokens(response_content)
                
                return {
                    "response": response_content,
                    "token_usage": {
                        "completion_tokens": output_tokens,
                        "total_tokens": input_tokens + output_tokens
                    }
                }
                
            except Exception as e:
                print(f"生成响应错误: {e}")
                return {
                    "response": f"处理响应时发生错误: {str(e)}",
                    "token_usage": {
                        "completion_tokens": 0,
                        "total_tokens": 0
                    }
                }