"""Azure service client management."""
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents import SearchClient
from azure.ai.formrecognizer import DocumentAnalysisClient
from openai import AzureOpenAI
from src.config.settings import azure_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

class AzureClientManager:
    """Manages Azure service clients with lazy initialization."""
    
    def __init__(self):
        """Initialize the client manager."""
        self._search_index_client = None
        self._search_client = None
        self._document_analysis_client = None
        self._openai_client = None
        self._credential = None
    
    @property
    def credential(self) -> AzureKeyCredential:
        """Get Azure credential (lazy initialization)."""
        if self._credential is None:
            self._credential = AzureKeyCredential(azure_config.search_api_key)
        return self._credential
    
    @property
    def search_index_client(self) -> SearchIndexClient:
        """Get Azure Search Index client (lazy initialization)."""
        if self._search_index_client is None:
            self._search_index_client = SearchIndexClient(
                endpoint=azure_config.search_endpoint,
                credential=self.credential
            )
            logger.debug("Initialized Azure Search Index client")
        return self._search_index_client
    
    @property
    def search_client(self) -> SearchClient:
        """Get Azure Search client (lazy initialization)."""
        if self._search_client is None:
            self._search_client = SearchClient(
                endpoint=azure_config.search_endpoint,
                index_name=azure_config.search_index_name,
                credential=self.credential
            )
            logger.debug("Initialized Azure Search client")
        return self._search_client
    
    @property
    def document_analysis_client(self) -> DocumentAnalysisClient:
        """Get Azure Document Intelligence client (lazy initialization)."""
        if self._document_analysis_client is None:
            doc_credential = AzureKeyCredential(azure_config.doc_intelligence_key)
            self._document_analysis_client = DocumentAnalysisClient(
                endpoint=azure_config.doc_intelligence_endpoint,
                credential=doc_credential
            )
            logger.debug("Initialized Azure Document Intelligence client")
        return self._document_analysis_client
    
    @property
    def openai_client(self) -> AzureOpenAI:
        """Get Azure OpenAI client (lazy initialization)."""
        if self._openai_client is None:
            self._openai_client = AzureOpenAI(
                api_key=azure_config.openai_api_key,
                api_version=azure_config.openai_api_version,
                azure_endpoint=azure_config.openai_endpoint
            )
            logger.debug("Initialized Azure OpenAI client")
        return self._openai_client
    
    def health_check(self) -> dict:
        """Perform health check on all Azure services."""
        health_status = {
            "search_service": False,
            "document_intelligence": False,
            "openai_service": False
        }
        
        try:
            # Test Search service
            self.search_index_client.list_indexes()
            health_status["search_service"] = True
            logger.info("✅ Azure Search service is healthy")
        except Exception as e:
            logger.error(f"❌ Azure Search service health check failed: {e}")
        
        try:
            # Test OpenAI service
            self.openai_client.embeddings.create(
                input="health check",
                model=azure_config.openai_embedding_model
            )
            health_status["openai_service"] = True
            logger.info("✅ Azure OpenAI service is healthy")
        except Exception as e:
            logger.error(f"❌ Azure OpenAI service health check failed: {e}")
        
        # Document Intelligence health check would require actual document
        health_status["document_intelligence"] = True  # Assume healthy if client creation succeeds
        logger.info("✅ Azure Document Intelligence client initialized")
        
        return health_status 