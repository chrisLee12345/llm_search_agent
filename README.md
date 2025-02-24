# LLM Search

// This source code is licensed under the MIT license.
// LICENSE file in the root directory of this source tree.

An intelligent search system based on large language models.

Features
Supports multiple document formats (PDF, Word, Excel, Markdown, Text)
Intelligent document parsing and vectorized storage
Semantic search powered by Azure OpenAI
Adaptive context management
Token usage tracking
Result quality evaluation

## Project Structure
llm-search/
├── engine/
│   ├── core/           # Core functionality module
│   ├── indexer/        # Document indexing module
│   └── web/            # Web API configuration
├── knowledge_base/
│   ├── docs/           # Document storage
│   └── indexes/        # Index storage
└── tests/              # Test cases

## Requirements
Python 3.8+
Access to Azure OpenAI API
Access to Bing Search API

## Installation and Usage
### 1. Clone the Repository
git clone https://github.com/chrisLee12345/llm-search.git
cd llm-search

### 2. Install Dependencies
# It is recommended to create a virtual environment first
pip install -r requirements.txt

### 3. Configure API Keys
Configure the required API parameters in engine/web/apiconfig.py.

### 4. Knowledge Base Configuration
Knowledge base files are stored in the directory knowledge_base/docs/.
The system will automatically process documents and create indexes, which are stored in the knowledge_base/indexes/ directory.

### 5. Cost Calculation
The costs for each run will be automatically generated and stored in the following path:
logs/costs
---

## Usage Examples
### Quick Start
python quick_start.py
By default, this runs an example query:
"Analyze the development trends of China's new energy vehicle industry"

### Custom Queries
1. Edit the quick_start.py file:
if __name__ == "__main__":
    # Modify the query content (supports natural language)
    query = "Your query content"
    asyncio.run(analyze_query(query))

2. Re-run the script:
python quick_start.py

