from dataclasses import dataclass
from typing import Dict, List, Optional, Union
import json
from datetime import datetime
from pathlib import Path
import tiktoken
import asyncio
from concurrent.futures import ThreadPoolExecutor

@dataclass
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    reasoning_tokens: Optional[int] = None

@dataclass
class APIUsage:  # 添加这个类的定义
    content: str
    token_usage: TokenUsage
    cost: float
    model: str
    task_type: str
    session_id: str
    timestamp: datetime
    thinking_time: float = 0.0

class CostTracker:
    def __init__(self, log_dir: str = "/Users/bojieli/pyproject/logs/costs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.MODEL_PRICING = {
            "gpt-4": {"input": 15.0, "output": 60.0},
            "text-embedding-3-large": {"input": 1.10, "output": 1.10},
            "o3-mini": {"input": 1.10, "output": 4.40},
            "o1": {"input": 15.0, "output": 60.0},
            "gpt-4o": {"input": 15.0, "output": 60.0},  # 添加新模型
        }
        
        # 初始化 tokenizer
        try:
            self.tokenizer = tiktoken.encoding_for_model("gpt-4o")
        except Exception as e:
            print(f"初始化 tokenizer 失败: {e}")
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        # 初始化线程池
        self.executor = ThreadPoolExecutor(max_workers=4)
        self._write_lock = asyncio.Lock()

    def count_tokens(self, text: str) -> int:
        """计算文本的 token 数量"""
        try:
            return len(self.tokenizer.encode(text))
        except Exception as e:
            print(f"计算 token 失败: {e}")
            return 0

    def calculate_cost(self, prompt_tokens: int, completion_tokens: int, model: str) -> float:
        """计算成本，添加错误处理"""
        try:
            pricing = self.MODEL_PRICING.get(model)
            if not pricing:
                print(f"未知模型 {model}，使用默认定价")
                pricing = self.MODEL_PRICING["o3-mini"]
            
            input_cost = (prompt_tokens / 1_000_000) * pricing["input"]
            output_cost = (completion_tokens / 1_000_000) * pricing["output"]
            return round(input_cost + output_cost, 6)
        except Exception as e:
            print(f"计算成本失败: {e}")
            return 0.0

    async def track_usage(self, 
                         content: str,
                         token_usage: Union[Dict, TokenUsage],
                         model: str,
                         session_id: str,
                         task_type: str,
                         thinking_time: float = 0.0):
        """记录API使用情况，使用本地 token 计算"""
        try:
            # 直接使用 tiktoken 计算 token
            total_tokens = self.count_tokens(content)
            # 根据任务类型估算输入输出比例
            if task_type in ["chat", "completion"]:
                prompt_ratio = 0.3
            elif task_type in ["embedding", "search"]:
                prompt_ratio = 0.8
            else:
                prompt_ratio = 0.5
                
            prompt_tokens = int(total_tokens * prompt_ratio)
            completion_tokens = total_tokens - prompt_tokens
            
            token_usage = TokenUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens
            )
            
            cost = self.calculate_cost(
                token_usage.prompt_tokens,
                token_usage.completion_tokens,
                model
            )
            
            usage = APIUsage(
                content=content,
                token_usage=token_usage,
                cost=cost,
                model=model,
                task_type=task_type,
                session_id=session_id,
                timestamp=datetime.now(),
                thinking_time=thinking_time
            )
            
            await self._save_usage(usage)
            
        except Exception as e:
            print(f"记录使用情况失败: {e}")

    async def _save_usage(self, usage: APIUsage):
        """保存使用记录，添加并发控制"""
        try:
            log_file = self.log_dir / f"{usage.timestamp.strftime('%Y-%m')}.json"
            log_entry = {
                "timestamp": usage.timestamp.isoformat(),
                "session_id": usage.session_id,
                "model": usage.model,
                "task_type": usage.task_type,
                "prompt_tokens": usage.token_usage.prompt_tokens,
                "completion_tokens": usage.token_usage.completion_tokens,
                "total_tokens": usage.token_usage.total_tokens,
                "reasoning_tokens": usage.token_usage.reasoning_tokens,
                "cost": usage.cost,
                "thinking_time": usage.thinking_time,
                "content": usage.content
            }
            
            async with self._write_lock:
                await self._append_to_log(log_file, log_entry)
                
        except Exception as e:
            print(f"保存使用记录失败: {e}")
    
    async def _append_to_log(self, log_file: Path, entry: Dict):
        """追加日志记录"""
        entries = []
        if log_file.exists():
            with open(log_file, 'r', encoding='utf-8') as f:
                entries = json.load(f)
        
        entries.append(entry)
        
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(entries, f, ensure_ascii=False, indent=2)