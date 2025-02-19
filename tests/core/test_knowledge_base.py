from engine.indexer.document_store import DocumentStore

def test_knowledge_base():
    doc_store = DocumentStore()
    doc_store.index_documents()
    print("知识库索引完成")

if __name__ == "__main__":
    test_knowledge_base()