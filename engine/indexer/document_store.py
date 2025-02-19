from pathlib import Path
from typing import List, Dict, Optional, Any
from langchain_chroma import Chroma
from langchain_openai import AzureOpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from .document_loader import DocumentLoader  # 确保这个导入正确
from ..web.apiconfig import config
import json
import tiktoken

class DocumentStore:
    def __init__(self, 
                 docs_dir: str = "/Users/bojieli/pyproject/llm-search/knowledge_base/docs",
                 index_dir: str = "/Users/bojieli/pyproject/llm-search/knowledge_base/indexes",
                 embedding_config: Optional[Dict[str, Any]] = None):
        # 基础路径配置
        self.docs_dir = Path(docs_dir)
        self.index_dir = Path(index_dir)
        self.loader = DocumentLoader(docs_dir)
        
        # 使用统一配置管理的 Embedding 配置
        self.embedding_config = embedding_config or config.api.embedding
        
        # 分块策略配置
        self.chunk_strategies = {
            'pdf': (1.5 * 1024 * 1024, 0.1),
            'docx': (768 * 1024, 0.1),
            'xlsx': (1024 * 1024, 0.1),
            'md': (384 * 1024, 0.1),
            'txt': (256 * 1024, 0.1),
            'default': (512 * 1024, 0.1)
        }
        
        # 初始化组件
        self._init_components()
    
    def _init_components(self):
        """初始化所有组件"""
        # 初始化 tokenizer
        try:
            self.tokenizer = tiktoken.encoding_for_model(self.embedding_config["model"])
        except Exception as e:
            print(f"初始化 tokenizer 失败: {e}")
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        # 初始化 embeddings
        self.embeddings = AzureOpenAIEmbeddings(**self.embedding_config)
        
        # 初始化向量存储
        persist_directory = str(self.index_dir / "chroma_db")
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        self.store = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embeddings,
            collection_name="documents"
        )
    
    def _process_documents(self, documents: List[Dict]) -> List[Document]:
        """处理文档"""
        # 这里需要实现文档处理逻辑
        # 返回处理后的文档列表
        return [Document(**doc) for doc in documents]
    
    async def _track_embedding_usage(self, text: str):
        """追踪 embedding 使用情况"""
        # 这里需要实现 token 使用统计逻辑
        pass
    
    def count_tokens(self, text: str) -> int:
        """计算文本的 token 数量"""
        return len(self.tokenizer.encode(text))
    
    async def search(self, 
                    query: str, 
                    k: int = 5,
                    filters: Optional[Dict] = None) -> List[Document]:
        """搜索文档"""
        # 计算 token 使用量
        try:
            if hasattr(self, 'cost_tracker'):
                await self._track_embedding_usage(query)
        except Exception as e:
            print(f"Token 统计错误: {e}")
            
        # 由于 similarity_search 是同步方法，使用 sync 调用
        return self.store.similarity_search(
            query, k=k, filter=filters
        )
    
    async def add_documents(self, 
                          documents: List[Document], 
                          metadata: Optional[Dict] = None):
        """添加文档"""
        if metadata:
            for doc in documents:
                doc.metadata.update(metadata)
        
        processed_docs = self._process_documents([
            {'content': doc.page_content, 'metadata': doc.metadata}
            for doc in documents
        ])
        
        # 追踪每个文档的 token 使用情况
        for doc in processed_docs:
            try:
                await self._track_embedding_usage(doc.page_content)
            except Exception as e:
                print(f"Token 统计错误: {e}")
        
        self.store.add_documents(processed_docs)
    
    def delete_documents(self, document_ids: List[str]):
        """删除文档"""
        self.store.delete(document_ids)