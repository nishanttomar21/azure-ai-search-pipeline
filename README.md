# Azure AI Search Pipeline

A production-grade pipeline for document ingestion, processing, embedding, and search using Azure Cognitive Search, Azure Document Intelligence, and Azure OpenAI.

## Features
- Download and process documents from URLs
- Extract content and metadata using Azure Document Intelligence
- Generate vector embeddings with Azure OpenAI
- Index documents and embeddings in Azure Cognitive Search
- Interactive CLI for keyword, vector, and hybrid search

## Project Structure
```
azure-ai-search-pipeline/
├── src/
│   ├── __init__.py
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── azure_clients.py
│   │   ├── document_processor.py
│   │   └── embedding_generator.py
│   ├── search/
│   │   ├── __init__.py
│   │   ├── index_manager.py
│   │   └── interactive_search.py
│   └── utils/
│       ├── __init__.py
│       ├── file_utils.py
│       └── logger.py
├── main.py
├── .env
├── requirements.txt
└── README.md
```

## Setup
1. Clone the repo and install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
2. Create a `.env` file with your Azure and OpenAI credentials:
   ```env
   AZURE_SEARCH_ENDPOINT=...
   AZURE_SEARCH_API_KEY=...
   DOC_INTELLIGENCE_ENDPOINT=...
   DOC_INTELLIGENCE_KEY=...
   OPENAI_API_ENDPOINT=...
   OPENAI_API_KEY=...
   ```
3. Run the pipeline:
   ```sh
   python main.py
   ```

## License
MIT