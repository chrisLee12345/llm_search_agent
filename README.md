# LLM Search

**License Notice**  
This project is licensed under the [MIT License](./license.txt) (see the LICENSE file in the project's root directory for details).

**Overview**  
**LLM Search** is an intelligent search system powered by large language models. It efficiently parses and indexes various document formats, enabling semantic search for accurate results.

---

## Key Features

- **Multiple Document Format Support:**
  - Supports PDF, Word, Excel, Markdown, and plain text formats.

- **Intelligent Document Parsing and Vectorized Storage:**
  - Automatically parses document content and stores data in vectorized form for efficient retrieval.

- **Semantic Search:**
  - Leverages Azure OpenAI to deliver powerful semantic search capabilities.

- **Adaptive Context Management:**
  - Dynamically adjusts query context to enhance search precision.

- **Token Usage Tracking:**
  - Automatically tracks token usage during API calls.

- **Result Quality Evaluation:**
  - Evaluates and analyzes the quality of search results.

---

## Project Structure

```
llm-search/
├── engine/
│   ├── core/           # Core functionality module
│   ├── indexer/        # Document indexing module
│   └── web/            # Web API configuration
├── knowledge_base/
│   ├── docs/           # Document storage
│   └── indexes/        # Index storage
└── tests/              # Test cases
```

---

## Requirements

- Python 3.8+
- Access to Azure OpenAI API
- Access to Bing Search API

---

## Installation and Usage

### 1. Clone the Repository

```bash
git clone https://github.com/chrisLee12345/llm-search.git
cd llm-search
```

### 2. Install Dependencies

It is recommended to create a virtual environment first, then run:

```bash
pip install -r requirements.txt
```

### 3. Configure API Keys

Set up the required API parameters in the `engine/web/apiconfig.py` file for Azure OpenAI and Bing Search API.

### 4. Configure the Knowledge Base

Place your knowledge base documents in the `knowledge_base/docs/` directory. The system will automatically process these documents and create indexes, which will be stored in the `knowledge_base/indexes/` directory.

### 5. Cost Calculation

After each run, cost information is automatically generated and stored in the `logs/costs` directory.

---

## Usage Examples

### Quick Start

Run the example script to execute the default query:

```bash
python quick_start.py
```

The default query is:  
*"Analyze the development trends of China's new energy vehicle industry"*

### Custom Queries

1. Edit the `quick_start.py` file to modify the query content:

   ```python
   if __name__ == "__main__":
       # Modify the query content (natural language supported)
       query = "Your query content"
       asyncio.run(analyze_query(query))
   ```

2. Save your changes and run the script again:

   ```bash
   python quick_start.py
   ```

---

## License

This project is licensed under the MIT License. For more details, see the [LICENSE](./license.txt) file.
