from typing import Dict, List, Optional, Callable, Any
from pydantic import BaseModel, Field
from datetime import datetime
from .events import EventType, WorkflowEvent
from .query_parser import QueryParser, SubTask
from .tool_orchestrator import ToolOrchestrator
from .result_evaluator import ResultEvaluator
from .conversation_manager import ConversationManager, Message
from ..utils.cost_tracker import CostTracker, TokenUsage

class WorkflowConfig(BaseModel):
    """工作流配置"""
    max_query_length: int = Field(default=2000, description="最大查询长度")
    min_tasks: int = Field(default=3, description="最小任务数量")
    default_models: Dict[str, str] = Field(
        default={
            "search": "text-embedding-3-large",
            "query_parsing": "gpt-4o",
            "task_execution": "gpt-4o",
            "response_generation": "gpt-4o",
            "text_embedding": "text-embedding-3-large"
        },
        description="各任务类型的默认模型"
    )
    base_tasks: List[str] = Field(
        default=[
            "当前状况分析",
            "历史背景和相关",
            "未来趋势预测",
            "相关影响评估",
            "专家观点汇总"
        ],
        description="基础任务列表"
    )

class WorkflowResult(BaseModel):
    """工作流结果"""
    final_answer: str
    subtasks: List[Dict]
    evaluation_result: Dict
    conversation_id: str
    metadata: Dict

class WorkflowContext:
    """工作流上下文"""
    def __init__(self, query: str, session_id: str):
        self.query = query
        self.session_id = session_id
        self.tasks: List[SubTask] = []
        self.task_results: List[Dict] = []
        self.response: Optional[Dict] = None
        self.evaluation: Optional[Dict] = None
        self.error: Optional[str] = None
        self.metadata: Dict = {}

class WorkflowCoordinator:
    """工作流协调器"""
    def __init__(self, config: Optional[WorkflowConfig] = None):
        self.config = config or WorkflowConfig()
        self.query_parser = QueryParser()
        self.tool_orchestrator = ToolOrchestrator()
        self.result_evaluator = ResultEvaluator()
        self.conversation_manager = ConversationManager()
        self.cost_tracker = CostTracker()
        
    async def process_query(self, query: str, session_id: str) -> WorkflowResult:
        """处理查询"""
        context = WorkflowContext(query, session_id)
        
        try:
            # 验证查询
            if not await self._validate_query(context):
                return self._create_error_result(context)
            
            # 生成任务
            if not await self._generate_tasks(context):
                return self._create_error_result(context)
            
            # 执行任务
            if not await self._execute_tasks(context):
                return self._create_error_result(context)
            
            # 生成响应
            if not await self._generate_response(context):
                return self._create_error_result(context)
            
            # 评估质量
            if not await self._evaluate_quality(context):
                return self._create_error_result(context)
            
            # 保存对话
            if not await self._save_conversation(context):
                return self._create_error_result(context)
            
            return self._create_success_result(context)
            
        except Exception as e:
            context.error = str(e)
            return self._create_error_result(context)

    async def _validate_query(self, context: WorkflowContext) -> bool:
        """验证查询"""
        if not context.query:
            context.error = "查询内容不能为空"
            return False
        
        if len(context.query) > self.config.max_query_length:
            context.error = "query too long"
            return False
        
        return True

    async def _generate_tasks(self, context: WorkflowContext) -> bool:
        """生成任务"""
        try:
            # 直接使用基础任务列表
            tasks = []
            for i, aspect in enumerate(self.config.base_tasks):
                tasks.append(SubTask(
                    task_type="search",
                    description=f"请全面深入地分析{context.query}的{aspect}，需要包含具体数据和事实依据",
                    priority=i + 1,
                    parameters={"depth": "detailed", "min_length": 200}
                ))
            
            context.tasks = tasks
            return True
            
        except Exception as e:
            context.error = f"任务生成失败: {str(e)}"
            return False

    async def _execute_tasks(self, context: WorkflowContext) -> bool:
        """执行任务"""
        try:
            results = await self.tool_orchestrator.execute_tasks(context.tasks)
            
            # 处理任务结果
            processed_results = []
            for result in results:
                # 检查结果是否有效
                if isinstance(result, dict):
                    # 如果结果中包含错误信息，则标记为失败
                    if isinstance(result.get("output"), str) and (
                        "Agent stopped" in result["output"] or 
                        "执行失败" in result["output"]
                    ):
                        context.error = result["output"]
                        return False
                    
                    # 构建标准化的结果格式
                    processed_result = {
                        "task_type": result.get("task_type", "unknown"),
                        "result": result.get("output", ""),
                        "token_usage": {
                            "completion_tokens": 0,
                            "total_tokens": 0
                        }
                    }
                    
                    # 如果有 token 使用信息，则更新
                    if "token_usage" in result:
                        processed_result["token_usage"] = result["token_usage"]
                    
                    # 记录任务执行成本
                    await self._track_task_cost(processed_result, context.session_id)
                    processed_results.append(processed_result)
                else:
                    context.error = "任务返回结果格式无效"
                    return False
            
            context.task_results = processed_results
            return True
            
        except Exception as e:
            context.error = f"任务执行失败: {str(e)}"
            return False

    def _normalize_response(self, response: Any, context: WorkflowContext) -> Dict:
        """统一响应格式"""
        if isinstance(response, str):
            # 如果响应是空的或者包含错误信息，返回错误
            if not response or "执行失败" in response:
                raise ValueError("响应内容无效")
                
            return {
                "response": response,
                "model": self._get_model_from_context(context),
                "token_usage": {
                    "completion_tokens": 0,
                    "total_tokens": 0
                }
            }
        elif isinstance(response, dict) and "content" in response:
            return {
                "response": response["content"],
                "model": self._get_model_from_context(context),
                "token_usage": response.get("token_usage", {
                    "completion_tokens": 0,
                    "total_tokens": 0
                })
            }
        else:
            raise ValueError("无法解析的响应格式")

    async def _generate_response(self, context: WorkflowContext) -> bool:
        """生成响应"""
        try:
            prompt = self._create_response_prompt(context.query)
            response = await self.tool_orchestrator.generate_response(prompt, context.task_results)
            
            # 统一响应格式
            response_data = self._normalize_response(response, context)
            
            # 记录响应生成成本
            if response_data.get("token_usage"):
                await self._track_response_cost(response_data, context.session_id)
            
            context.response = response_data
            return True
            
        except Exception as e:
            context.error = f"响应生成失败: {str(e)}"
            return False

    async def _evaluate_quality(self, context: WorkflowContext) -> bool:
        """评估质量"""
        try:
            eval_result = await self.result_evaluator.evaluate_with_fallback(
                answer=context.response["response"],
                query=context.query
            )
            context.evaluation = eval_result
            return True
            
        except Exception as e:
            context.error = f"质量评估失败: {str(e)}"
            return False

    async def _save_conversation(self, context: WorkflowContext) -> bool:
        """保存对话"""
        try:
            self.conversation_manager.add_message(
                session_id=context.session_id,
                message=Message(
                    role="assistant",
                    content=context.response["response"],
                    timestamp=datetime.now(),
                    metadata=context.evaluation or {}
                )
            )
            return True
            
        except Exception as e:
            context.error = f"对话保存失败: {str(e)}"
            return False

    def _ensure_minimum_tasks(self, existing_tasks: List[SubTask], query: str) -> List[SubTask]:
        """确保最小任务数量"""
        result = list(existing_tasks)
        tasks_needed = max(0, self.config.min_tasks - len(existing_tasks))
        
        for i, aspect in enumerate(self.config.base_tasks[:tasks_needed]):
            result.append(SubTask(
                task_type="search",
                description=f"请全面深入地分析{query}的{aspect}，需要包含具体数据和事实依据",
                priority=len(existing_tasks) + i + 1,
                parameters={"depth": "detailed", "min_length": 200}
            ))
        
        return result

    async def _track_task_cost(self, result: Dict, session_id: str) -> None:
        """追踪任务成本"""
        used_model = self._get_model_from_result(result)
        if not used_model:
            task_type = result.get("task_type")
            used_model = self.config.default_models.get(task_type, "gpt-4o")
        
        # 使用文本长度计算 token 使用量
        input_text = result.get("input", "")
        output_text = result.get("result", "")
        
        await self.cost_tracker.track_usage(
            content=output_text,
            token_usage=result["token_usage"],  # 已经由 ToolOrchestrator 计算好的 token 使用量
            model=used_model,
            session_id=session_id,
            task_type=result.get("task_type", "task_execution")
        )

    async def _track_response_cost(self, response: Dict, session_id: str) -> None:
        """追踪响应成本"""
        used_model = self._get_model_from_result(response)
        
        await self.cost_tracker.track_usage(
            content=response["response"],
            token_usage=response["token_usage"],
            model=used_model or self.config.default_models["response_generation"],
            session_id=session_id,
            task_type="response_generation"
        )

    def _normalize_response(self, response: Any, context: WorkflowContext) -> Dict:
        """统一响应格式"""
        if isinstance(response, str):
            return {
                "response": response,
                "model": self._get_model_from_context(context),
                "token_usage": None
            }
        return response

    def _get_model_from_result(self, result: Dict) -> Optional[str]:
        """从结果中获取模型信息"""
        if isinstance(result.get("model"), dict):
            return result["model"].get("name")
        return result.get("model")

    def _get_model_from_context(self, context: WorkflowContext) -> str:
        """从上下文中获取模型信息"""
        # 尝试从最后一个任务结果中获取模型信息
        for result in reversed(context.task_results):
            model = self._get_model_from_result(result)
            if model:
                return model
        
        # 使用默认模型
        return self.config.default_models["response_generation"]

    def _create_response_prompt(self, query: str) -> str:
        """创建响应提示"""
        return f"""请基于以下要求详细回答问题：{query}

要求：
1. 答案必须超过500字，确保内容充实完整
2. 必须包含具体的数据、事实和专家观点作为支撑
3. 从多个维度进行分析（如现状、历史、趋势、影响等）
4. 结构要清晰，采用总-分-总的写作框架
5. 语言acee专业又要通俗易懂
6. 如有可能，请给出相关的建议或解决方案

请确保回答全面且有深度。"""

    def _create_error_message(self, error: str) -> str:
        """创建错误消息"""
        return f"处理查询时发生错误: {error}\n\n建议：\n1. 请检查输入内容是否合适\n2. 稍后重试\n3. 如果问题持续存在，请联系支持团队"

    def _create_success_result(self, context: WorkflowContext) -> WorkflowResult:
        """创建成功结果"""
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "used_fallback": context.evaluation.get("used_fallback", False),
            "web_sources": context.evaluation.get("web_sources", []),
            "quality_score": context.evaluation.get("quality_score", 0.0)
        }
        
        return WorkflowResult(
            final_answer=context.response["response"],
            subtasks=[task.dict() for task in context.tasks],
            evaluation_result=context.evaluation or {},
            conversation_id=context.session_id,
            metadata=metadata
        )

    def _create_error_result(self, context: WorkflowContext) -> WorkflowResult:
        """创建错误结果"""
        error_message = self._create_error_message(context.error or "未知错误")
        
        return WorkflowResult(
            final_answer=error_message,
            subtasks=[],
            evaluation_result={"quality_score": 0},
            conversation_id=context.session_id,
            metadata={
                "timestamp": datetime.now().isoformat(),
                "error": context.error
            }
        )