from ..indexer.document_store import DocumentStore
from ..utils.custom_llm import DeepSeekLLM
from langchain.chains import RetrievalQA

from ..utils.custom_llm import AzureGPT4LLM  # 更新导入

class SearchEngine:
    def __init__(self):
        self.document_store = DocumentStore()
        self.llm = AzureGPT4LLM()  # 更新类名
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.document_store.store.as_retriever()
        )
    
    def search(self, query):
        response = self.qa_chain.run(query)
        return response