"""Configuration settings for Azure RAG pipeline."""
import os
from dataclasses import dataclass
from typing import List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class AzureConfig:
    """Azure service configurations."""
    # Azure Search
    search_endpoint: str = os.getenv("AZURE_SEARCH_ENDPOINT")
    search_api_key: str = os.getenv("AZURE_SEARCH_API_KEY")
    search_index_name: str = "library"
    
    # Azure Document Intelligence
    doc_intelligence_endpoint: str = os.getenv("DOC_INTELLIGENCE_ENDPOINT")
    doc_intelligence_key: str = os.getenv("DOC_INTELLIGENCE_KEY")
    
    # Azure OpenAI
    openai_endpoint: str = os.getenv("OPENAI_API_ENDPOINT")
    openai_api_key: str = os.getenv("OPENAI_API_KEY")
    openai_api_version: str = "2024-10-21"
    openai_embedding_model: str = "text-embedding-3-large"
    
    def validate(self) -> bool:
        """Validate that all required configuration values are present."""
        required_fields = [
            self.search_endpoint, self.search_api_key,
            self.doc_intelligence_endpoint, self.doc_intelligence_key,
            self.openai_endpoint, self.openai_api_key
        ]
        return all(field is not None for field in required_fields)

@dataclass
class ProcessingConfig:
    """Document processing configurations."""
    vector_dimensions: int = 1536  # For text-embedding-3-large
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_retries: int = 3
    request_timeout: int = 30
    batch_size: int = 10

# Document URLs to process
DOCUMENT_URLS: List[str] = [
    "https://www.gehealthcare.com/support/manuals?search=eyJzZWFyY2hUZXJtIjoiMjA5MjcwMy0wMDEiLCJsYW5ndWFnZU5hbWUiOiJFbmdsaXNoIChFTikifQ%3D%3D",
    "https://www.gehealthcare.com/support/manuals?search=eyJzZWFyY2hUZXJtIjoiMjEwNTUxOC0wMDEiLCJsYW5ndWFnZU5hbWUiOiJFbmdsaXNoIChFTikifQ%3D%3D",
    "https://www.gehealthcare.com/support/manuals?search=eyJzZWFyY2hUZXJtIjoiMjA1MDgwMi0wMDEiLCJsYW5ndWFnZU5hbWUiOiJFbmdsaXNoIChFTikifQ%3D%3D",
    "https://www.gehealthcare.com/support/manuals?search=eyJzZWFyY2hUZXJtIjoiMjA4MTQ5OS0wMDIiLCJsYW5ndWFnZU5hbWUiOiJFbmdsaXNoIChFTikifQ%3D%3D"
    # "https://drive.google.com/uc?export=download&id=1oFOzMI_0B8lMqXePmBIRJ5DsYSDNR1sM",   # Sample 1
    # "https://drive.google.com/uc?export=download&id=14tog4mFflzasQhFpDxaeMCK-XoqeHLJ9"    # Sample 2

]

# Initialize global config instances
azure_config = AzureConfig()
processing_config = ProcessingConfig() 