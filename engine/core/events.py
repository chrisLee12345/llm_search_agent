from typing import Dict, Any
from enum import Enum
from dataclasses import dataclass
from datetime import datetime

class EventType(Enum):
    QUERY_RECEIVED = "query_received"
    QUERY_VALIDATED = "query_validated"
    TASKS_GENERATED = "tasks_generated"
    TASKS_EXECUTED = "tasks_executed"
    RESPONSE_GENERATED = "response_generated"
    QUALITY_EVALUATED = "quality_evaluated"
    CONVERSATION_UPDATED = "conversation_updated"
    ERROR_OCCURRED = "error_occurred"

@dataclass
class WorkflowEvent:
    type: EventType
    data: Dict[str, Any]
    timestamp: datetime = datetime.now()
    metadata: Dict[str, Any] = None