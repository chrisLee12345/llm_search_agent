from langchain_openai import AzureOpenAIEmbeddings
import numpy as np

def test_embedding():
    print("开始测试 Azure OpenAI Embedding 模型...")
    
    try:
        # 初始化 embedding 模型
        embeddings = AzureOpenAIEmbeddings(
            azure_endpoint="https://ai-search-gpt4.openai.azure.com",
            api_key="DuyZ0hHrPDcKN5FswiEyi43Reqqu2qLulqfQMglca8xDYA79URG3JQQJ99BAACYeBjFXJ3w3AAABACOGvlW6",
            #api_version="2024-08-01-preview",
            deployment="text-embedding-3-large"
            
        )
        
        # 测试文本
        test_texts = [
            "人工智能是计算机科学的一个重要分支",
            "机器学习是人工智能的核心技术之一"
        ]
        
        # 生成 embeddings
        print("\n正在生成文本向量...")
        vectors = embeddings.embed_documents(test_texts)
        
        # 验证结果
        print(f"\n生成的向量数量: {len(vectors)}")
        print(f"向量维度: {len(vectors[0])}")
        
        # 计算两个文本的相似度
        similarity = np.dot(vectors[0], vectors[1]) / (np.linalg.norm(vectors[0]) * np.linalg.norm(vectors[1]))
        print(f"\n两个测试文本的余弦相似度: {similarity:.4f}")
        
        print("\nEmbedding 模型测试成功！")
        
    except Exception as e:
        print(f"\n测试失败: {str(e)}")

if __name__ == "__main__":
    test_embedding()