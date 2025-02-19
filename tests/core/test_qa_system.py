import asyncio
from engine.core.result_evaluator import ResultEvaluator
from engine.indexer.document_store import DocumentStore
from engine.core.query_parser import QueryParser

async def test_qa():
    # 初始化组件
    query_parser = QueryParser()
    evaluator = ResultEvaluator()  # 添加这一行
    
    print("正在索引文档...")
    doc_store = DocumentStore()
    doc_store.index_documents()
    
    query = "灵碳智能成立于1985年"
    print(f"\n问题: {query}\n")
    
    print("搜索相关内容...")
    context = doc_store.search(query, k=3)
    
    print("生成初步答案...")
    initial_answer = query_parser.generate_answer(query, context)  # 修改这一行
    
    # 评估答案并可能使用备用搜索
    print("\n评估答案...")
    result = await evaluator.evaluate_with_fallback(
        answer=initial_answer,
        context=context,
        query=query
    )
    
    # 输出结果
    print("\n=== 最终结果 ===")
    print(f"答案: {result['answer']}")
    print(f"\n评估分数: {result['evaluation'].score}")
    print(f"幻觉风险: {result['evaluation'].hallucination_risk}")
    print(f"置信度: {result['evaluation'].confidence}")
    
    if result['used_fallback']:
        print("\n使用了网络搜索补充信息")
        print("参考来源:")
        for url in result['web_sources']:
            print(f"- {url}")

if __name__ == "__main__":
    asyncio.run(test_qa())