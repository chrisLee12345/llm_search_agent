from typing import Dict, List, Optional
from pydantic import BaseModel
from openai import AzureOpenAI  # 修改导入
from langchain.prompts import ChatPromptTemplate
from .fallback_search import FallbackSearchEngine
from langchain.docstore.document import Document
from ..web.apiconfig import config
import json
import os
import tiktoken

class EvaluationResult(BaseModel):
    score: float
    hallucination_risk: float
    confidence: float
    issues: List[str]
    token_usage: Dict[str, int] = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0
    }

class ResultEvaluator:
    def __init__(self):
        self.client = AzureOpenAI(
            api_key=config.api.azure_openai["api_key"],
            api_version=config.api.azure_openai["api_version"],
            azure_endpoint=config.api.azure_openai["azure_endpoint"]
        )
        
        self.model = config.api.azure_openai["model"]
        self.fallback_search = FallbackSearchEngine()
        
        try:
            self.tokenizer = tiktoken.encoding_for_model(self.model)
        except Exception as e:
            print(f"初始化 tokenizer 失败: {e}")
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
    
    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))
    
    def _truncate_context(self, context_texts: List[str], max_tokens: int = 4000) -> List[str]:
        truncated = []
        current_tokens = 0
        
        for text in context_texts:
            tokens = self.count_tokens(text)
            if current_tokens + tokens <= max_tokens:
                truncated.append(text)
                current_tokens += tokens
            else:
                break
        
        return truncated
    
    def _call_llm(self, system_prompt: str, user_prompt: str) -> Dict:
        prompt_tokens = self.count_tokens(system_prompt + user_prompt)
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0,
            max_tokens=1000
        )
        
        content = response.choices[0].message.content
        completion_tokens = self.count_tokens(content)
        
        return {
            "content": content,
            "token_usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            }
        }

    def evaluate(self, 
                answer: str, 
                context: List[Document],
                threshold: float = 0.7) -> EvaluationResult:
        context_texts = [doc.page_content for doc in context]
        
        system_prompt = """你是一个专业的答案评估器。你的任务是评估给定答案的质量，并返回一个 JSON 格式的评估结果。

请严格按照以下格式返回（不要添加任何其他内容）：
{
    "score": 0.9,
    "hallucination_risk": 0.1,
    "confidence": 0.9,
    "issues": ["问题1", "问题2"]
}

注意事项：
1. 所有数值必须是 0-1 之间的小数
2. score 表示答案的整体质量，越高越好
3. hallucination_risk 表示答案包含虚构信息的风险，越低越好
4. confidence 表示你对评估结果的确信程度
5. issues 必须是字符串数组，列出发现的具体问题

评估要点：
1. 答案是否包含具体的数据和事实依据
2. 信息是否与上下文一致
3. 论述是否全面且有逻辑性
4. 是否存在未经验证的信息"""
        
        user_prompt = f"请评估以下答案的质量：\n\n上下文：{chr(10).join(context_texts)}\n\n答案：{answer}"
        
        try:
            response = self._call_llm(system_prompt, user_prompt)
            content = response["content"].strip()
            eval_dict = json.loads(content)
            
            # 添加 token 使用统计
            eval_dict["token_usage"] = response["token_usage"]
            
            # 验证数值范围
            for field in ["score", "hallucination_risk", "confidence"]:
                value = float(eval_dict[field])
                if not 0 <= value <= 1:
                    raise ValueError(f"{field} 必须在 0-1 之间，当前值: {value}")
                eval_dict[field] = value
            
            # 确保 issues 是字符串列表
            if not isinstance(eval_dict['issues'], list):
                eval_dict['issues'] = []
            eval_dict['issues'] = [str(issue) for issue in eval_dict['issues']]
            
            return EvaluationResult(**eval_dict)
            
        except Exception as e:
            print(f"评估结果解析失败: {str(e)}\n原始响应: {response['content']}")
            return EvaluationResult(
                score=0.1,
                hallucination_risk=0.9,
                confidence=0.1,
                issues=["评估过程出错，无法获得有效结果"],
                token_usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            )
    
    async def evaluate_with_fallback(self,
                                   answer: str,
                                   query: str,
                                   context: Optional[List[Document]] = None) -> Dict:
        try:
            if context is None:
                context = []
            
            total_tokens = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }
            
            # 评估原始答案
            eval_result = self.evaluate(answer, context)
            for key in total_tokens:
                total_tokens[key] += eval_result.token_usage[key]
            
            if eval_result.score < 0.7 or eval_result.hallucination_risk > 0.3:
                web_results = await self.fallback_search.fallback_search(query)
                new_docs = [Document(page_content=r.content) for r in web_results]
                new_context = context + new_docs
                
                # 重新生成答案
                response = await self._regenerate_answer(query, [doc.page_content for doc in new_context])
                for key in total_tokens:
                    total_tokens[key] += response["token_usage"][key]
                new_answer = response["content"]
                
                # 评估新答案
                new_eval_result = self.evaluate(new_answer, new_context)
                for key in total_tokens:
                    total_tokens[key] += new_eval_result.token_usage[key]
                
                return {
                    "answer": new_answer,
                    "evaluation": new_eval_result.model_dump(),
                    "used_fallback": True,
                    "web_sources": [r.url for r in web_results],
                    "token_usage": total_tokens
                }
            
            return {
                "answer": answer,
                "evaluation": eval_result.model_dump(),
                "used_fallback": False,
                "web_sources": [],
                "token_usage": total_tokens
            }
            
        except Exception as e:
            print(f"评估过程发生错误: {e}")
            error_result = EvaluationResult(
                score=0.1,
                hallucination_risk=0.9,
                confidence=0.1,
                issues=[f"评估过程发生错误: {str(e)}"]
            )
            return {
                "answer": answer,
                "evaluation": error_result.model_dump(),  # 使用 model_dump
                "used_fallback": False,
                "web_sources": []
            }

    async def _regenerate_answer(self, query: str, context: List[str]) -> Dict:
        """使用扩展的上下文重新生成答案"""
        system_prompt = "基于提供的上下文信息，请生成一个准确、完整的回答。"
        user_prompt = f"问题：{query}\n\n上下文信息：{chr(10).join(context)}"
        
        return self._call_llm(system_prompt, user_prompt)  # 直接返回完整的响应字典