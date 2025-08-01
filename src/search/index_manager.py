"""Azure Search index management functionality."""
from typing import List, Dict, Any, Optional
from azure.search.documents.indexes.models import (
    SearchIndex, SimpleField, SearchField, SearchFieldDataType,
    SearchableField, VectorSearch, VectorSearchProfile, HnswAlgorithmConfiguration
)
from azure.search.documents.models import VectorizedQuery
from src.core.azure_clients import AzureClientManager
from src.config.settings import azure_config, processing_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

class IndexManager:
    """Manages Azure Search index operations."""
    
    def __init__(self, client_manager: AzureClientManager):
        """Initialize with Azure client manager."""
        self.client_manager = client_manager
        self.index_name = azure_config.search_index_name
    
    def create_or_update_index(self) -> bool:
        """Create or update the search index with vector capabilities."""
        try:
            logger.info(f"🔧 Creating/updating index: {self.index_name}")
            
            fields = self._define_index_fields()
            vector_search = self._configure_vector_search()
            
            index = SearchIndex(
                name=self.index_name,
                fields=fields,
                vector_search=vector_search
            )
            
            result = self.client_manager.search_index_client.create_or_update_index(index=index)
            logger.info(f"✅ Index '{self.index_name}' created/updated successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to create/update index: {e}")
            return False
    
    def _define_index_fields(self) -> List:
        """Define the search index schema fields."""
        return [
            # Primary key field
            SimpleField(
                name="id", 
                type=SearchFieldDataType.String, 
                key=True, 
                filterable=True, 
                sortable=True
            ),
            
            # Searchable content field
            SearchableField(
                name="content", 
                type=SearchFieldDataType.String,
                searchable=True,
                filterable=False
            ),
            
            # Metadata fields
            SimpleField(
                name="product_name", 
                type=SearchFieldDataType.String, 
                filterable=True, 
                sortable=True,
                facetable=True
            ),
            SimpleField(
                name="filename", 
                type=SearchFieldDataType.String, 
                filterable=True, 
                sortable=True,
                facetable=True
            ),
            SimpleField(
                name="filepath", 
                type=SearchFieldDataType.String, 
                filterable=True
            ),
            SimpleField(
                name="document_url", 
                type=SearchFieldDataType.String, 
                filterable=True
            ),
            SimpleField(
                name="content_length", 
                type=SearchFieldDataType.Int32, 
                filterable=True, 
                sortable=True
            ),
            SimpleField(
                name="processed_at", 
                type=SearchFieldDataType.DateTimeOffset, 
                filterable=True, 
                sortable=True
            ),
            
            # Vector search field
            SearchField(
                name="content_vector",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=processing_config.vector_dimensions,
                vector_search_profile_name="vector-config"
            ),
        ]
    
    def _configure_vector_search(self) -> VectorSearch:
        """Configure vector search settings."""
        return VectorSearch(
            profiles=[
                VectorSearchProfile(
                    name="vector-config",
                    algorithm_configuration_name="hnsw-config"
                )
            ],
            algorithms=[
                HnswAlgorithmConfiguration(
                    name="hnsw-config",
                    parameters={
                        "m": 4,  # Number of bi-directional links for new nodes
                        "efConstruction": 400,  # Size of candidate list
                        "efSearch": 500,  # Size of candidate list for search
                        "metric": "cosine"  # Distance metric
                    }
                )
            ]
        )
    
    def upload_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """Upload documents to the search index."""
        try:
            if not documents:
                logger.warning("No documents to upload")
                return True
            
            logger.info(f"📤 Uploading {len(documents)} documents to index")
            
            # Validate documents before upload
            valid_docs = self._validate_documents(documents)
            if not valid_docs:
                logger.error("No valid documents to upload")
                return False
            
            results = self.client_manager.search_client.upload_documents(documents=valid_docs)
            
            # Analyze upload results
            success_count = sum(1 for r in results if r.succeeded)
            total_count = len(results)
            
            logger.info(f"📊 Upload complete: {success_count}/{total_count} successful")
            
            # Log failures
            for i, result in enumerate(results):
                if not result.succeeded:
                    logger.error(f"❌ Document {i+1} upload failed: {result.error_message}")
            
            return success_count == total_count
            
        except Exception as e:
            logger.error(f"❌ Failed to upload documents: {e}")
            return False
    
    def _validate_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate documents before upload."""
        valid_docs = []
        
        for doc in documents:
            if self._validate_single_document(doc):
                valid_docs.append(doc)
            else:
                logger.warning(f"⚠️ Invalid document skipped: {doc.get('id', 'unknown')}")
        
        return valid_docs
    
    def _validate_single_document(self, doc: Dict[str, Any]) -> bool:
        """Validate a single document structure."""
        required_fields = ['id', 'content']
        
        for field in required_fields:
            if field not in doc or not doc[field]:
                logger.warning(f"Missing required field '{field}' in document")
                return False
        
        # Validate vector embedding if present
        if 'content_vector' in doc:
            vector = doc['content_vector']
            if not isinstance(vector, list) or len(vector) != processing_config.vector_dimensions:
                logger.warning(f"Invalid vector dimension in document {doc['id']}")
                return False
        
        return True
    
    def delete_index(self) -> bool:
        """Delete the search index."""
        try:
            self.client_manager.search_index_client.delete_index(self.index_name)
            logger.info(f"✅ Index '{self.index_name}' deleted successfully")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to delete index: {e}")
            return False
    
    def get_index_statistics(self) -> Optional[Dict[str, Any]]:
        """Get index statistics and information."""
        try:
            index = self.client_manager.search_index_client.get_index(self.index_name)
            
            # Get document count
            results = self.client_manager.search_client.search(
                search_text="*",
                select=["id"],
                top=0,
                include_total_count=True
            )
            
            stats = {
                "index_name": index.name,
                "field_count": len(index.fields),
                "document_count": results.get_count(),
                "vector_search_enabled": index.vector_search is not None
            }
            
            logger.info(f"📊 Index statistics: {stats}")
            return stats
            
        except Exception as e:
            logger.error(f"❌ Failed to get index statistics: {e}")
            return None 