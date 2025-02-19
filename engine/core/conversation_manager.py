from typing import Dict, List, Optional
from datetime import datetime
from pydantic import BaseModel
import json
import sqlite3
from pathlib import Path
import tiktoken  # 添加 tiktoken 导入

class Message(BaseModel):
    role: str
    content: str
    timestamp: datetime
    metadata: Optional[Dict] = None
    token_count: Optional[int] = None  # 添加 token 计数字段

class ConversationManager:
    def __init__(self, db_path: str = "/Users/bojieli/pyproject/llm-search/data/conversations.db"):
        # 确保数据库目录存在
        db_dir = Path(db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)
        
        # 如果数据库文件存在，先删除它
        db_file = Path(db_path)
        if db_file.exists():
            db_file.unlink()
        
        # 连接数据库
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.create_tables()
        
        # 初始化 tokenizer
        try:
            self.tokenizer = tiktoken.encoding_for_model("gpt-4o")
        except Exception as e:
            print(f"初始化 tokenizer 失败: {e}")
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.clear_all()  # 初始化时清理数据库
    
    def clear_all(self):
        """清理所有对话历史"""
        self.conn.execute("DELETE FROM conversations")
        self.conn.commit()
    
    def clear_session(self, session_id: str):
        """清理指定会话的历史"""
        self.conn.execute("DELETE FROM conversations WHERE session_id = ?", (session_id,))
        self.conn.commit()

    def create_tables(self):
        """创建必要的数据表"""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY,
                session_id TEXT,
                message_type TEXT,
                content TEXT,
                timestamp DATETIME,
                metadata TEXT,
                token_count INTEGER
            )
        """)
        self.conn.commit()
    
    def add_message(self, 
                   session_id: str, 
                   message: Message):
        """添加新消息到对话历史"""
        self.conn.execute("""
            INSERT INTO conversations 
            (session_id, message_type, content, timestamp, metadata)
            VALUES (?, ?, ?, ?, ?)
        """, (
            session_id,
            message.role,
            message.content,
            message.timestamp.isoformat(),
            json.dumps(message.metadata)
        ))
        self.conn.commit()

    def count_tokens(self, text: str) -> int:
        """计算文本的 token 数量"""
        return len(self.tokenizer.encode(text))

    def add_message(self, session_id: str, message: Message):
        """添加新消息到对话历史"""
        # 计算 token 数量
        if message.token_count is None:
            message.token_count = self.count_tokens(message.content)
        
        self.conn.execute("""
            INSERT INTO conversations 
            (session_id, message_type, content, timestamp, metadata, token_count)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            session_id,
            message.role,
            message.content,
            message.timestamp.isoformat(),
            json.dumps(message.metadata),
            message.token_count
        ))
        self.conn.commit()

    def get_conversation_history(self, 
                               session_id: str,
                               limit: Optional[int] = None,
                               max_tokens: Optional[int] = None) -> List[Message]:
        """获取对话历史，支持 token 限制"""
        query = """
            SELECT message_type, content, timestamp, metadata, token_count
            FROM conversations
            WHERE session_id = ?
            ORDER BY timestamp DESC
        """
        params = [session_id]
        
        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)
            
        cursor = self.conn.execute(query, params)
        
        messages = []
        total_tokens = 0
        
        for row in cursor:
            token_count = row[4] or self.count_tokens(row[1])
            
            if max_tokens and total_tokens + token_count > max_tokens:
                break
                
            messages.append(Message(
                role=row[0],
                content=row[1],
                timestamp=datetime.fromisoformat(row[2]),
                metadata=json.loads(row[3]) if row[3] else {},
                token_count=token_count
            ))
            
            total_tokens += token_count
            
        return list(reversed(messages))  # 恢复正序

    def get_session_token_count(self, session_id: str) -> int:
        """获取会话的总 token 数量"""
        cursor = self.conn.execute("""
            SELECT SUM(token_count)
            FROM conversations
            WHERE session_id = ?
        """, (session_id,))
        
        result = cursor.fetchone()[0]
        return result or 0

    def trim_conversation(self, session_id: str, max_tokens: int) -> None:
        """修剪会话历史以适应 token 限制"""
        while self.get_session_token_count(session_id) > max_tokens:
            self.conn.execute("""
                DELETE FROM conversations
                WHERE session_id = ? AND id = (
                    SELECT id FROM conversations
                    WHERE session_id = ?
                    ORDER BY timestamp ASC
                    LIMIT 1
                )
            """, (session_id, session_id))
            self.conn.commit()

    def get_history(self, session_id: str) -> List[Message]:
        """获取指定会话的历史记录（兼容旧接口）"""
        return self.get_conversation_history(session_id)