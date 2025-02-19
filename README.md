# LLM Search

基于大语言模型的智能搜索系统。

## 功能特点

- 支持多种文档格式（PDF、Word、Excel、Markdown、Text）
- 智能文档解析和向量化存储
- 基于 Azure OpenAI 的语义搜索
- 自适应上下文管理
- Token 使用追踪
- 结果质量评估

## 项目结构
llm-search/
├── engine/
│   ├── core/           # 核心功能模块
│   ├── indexer/        # 文档索引模块
│   └── web/           # Web API 配置
├── knowledge_base/
│   ├── docs/          # 文档存储
│   └── indexes/       # 索引存储
└── tests/            # 测试用例


## 环境要求

- Python 3.8+
- Azure OpenAI API 访问权限
- Bing Search API 访问权限

## 安装使用

1. 克隆仓库
```bash
git clone https://github.com/chrisLee12345/llm-search.git
cd llm-search

2. 安装依赖
```bash
pip install -r requirements.txt

3.配置环境变量
```bash
export AZURE_OPENAI_API_KEY="your_api_key"
export AZURE_OPENAI_ENDPOINT="your_endpoint"
export BING_API_KEY="your_bing_api_key"

## 使用示例

项目提供了快速启动脚本 quick_start.py，可以直接运行测试：

```bash
python quick_start.py

默认会执行一个示例查询："分析中国新能源汽车行业的发展趋势"
你也可以修改 quick_start.py 中的查询内容来测试其他场景：
if __name__ == "__main__":
    # 修改这里的查询内容
    query = "你的查询内容"
    asyncio.run(analyze_query(query))
