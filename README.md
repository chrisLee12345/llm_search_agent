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

### 1. 克隆仓库
```bash
git clone https://github.com/chrisLee12345/llm-search.git
cd llm-search
```

### 2. 安装依赖
```bash
# 推荐先创建虚拟环境
pip install -r requirements.txt
```

### 3. 配置 API 密钥
```bash
# 请替换为你的实际 API 信息（临时生效）
export AZURE_OPENAI_API_KEY="your_api_key"       # Azure OpenAI 密钥
export AZURE_OPENAI_ENDPOINT="your_endpoint"     # Azure 服务地址
export BING_API_KEY="your_bing_api_key"          # Bing 搜索 API 密钥

# 永久生效建议写入 ~/.bashrc 或 ~/.zshrc（Linux/Mac）
# Windows 可使用 setx 命令或系统环境变量面板
```

---

## 使用示例

### 快速启动
```bash
python quick_start.py
```
*默认执行示例查询：`"分析中国新能源汽车行业的发展趋势"`*

### 自定义查询
1. 编辑 `quick_start.py` 文件：
```python
if __name__ == "__main__":
    # 修改查询内容（支持自然语言）
    query = "你的查询内容"
    asyncio.run(analyze_query(query))
```

2. 重新运行脚本：
```bash
python quick_start.py
```

