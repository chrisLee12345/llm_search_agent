import pytest
from datetime import datetime
from engine.core.conversation_manager import ConversationManager, Message

@pytest.fixture
def conversation_manager():
    return ConversationManager()

def test_add_message(conversation_manager):
    session_id = "test_session"
    message = Message(
        role="user",
        content="测试消息",
        timestamp=datetime.now(),
        metadata={"test": "data"}
    )
    
    conversation_manager.add_message(session_id, message)
    history = conversation_manager.get_history(session_id)
    
    assert len(history) == 1
    assert history[0] == message

def test_get_history_empty(conversation_manager):
    history = conversation_manager.get_history("nonexistent_session")
    assert len(history) == 0

def test_multiple_messages(conversation_manager):
    session_id = "test_session"
    messages = [
        Message(
            role="user",
            content=f"消息{i}",
            timestamp=datetime.now(),
            metadata={}
        ) for i in range(3)
    ]
    
    for msg in messages:
        conversation_manager.add_message(session_id, msg)
    
    history = conversation_manager.get_history(session_id)
    assert len(history) == 3
    assert [msg.content for msg in history] == ["消息0", "消息1", "消息2"]

@pytest.fixture(autouse=True)
def cleanup_db(conversation_manager):
    conversation_manager.clear_all()
    yield