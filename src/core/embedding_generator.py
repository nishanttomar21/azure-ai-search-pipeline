"""Vector embedding generation functionality."""
from typing import List, Optional, Dict, Any
import time
from src.core.azure_clients import AzureClientManager
from src.config.settings import azure_config, processing_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

class EmbeddingGenerator:
    """Handles vector embedding generation using Azure OpenAI."""
    
    def __init__(self, client_manager: AzureClientManager):
        """Initialize with Azure client manager."""
        self.client_manager = client_manager
        self.model = azure_config.openai_embedding_model
    
    def generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generate vector embedding for given text."""
        try:
            if not text or not text.strip():
                logger.warning("Empty or whitespace-only text provided for embedding")
                return None
            
            # Truncate text if too long (model-specific limits)
            text = self._truncate_text(text)
            
            response = self.client_manager.openai_client.embeddings.create(
                input=text.strip(),
                model=self.model
            )
            
            embedding = response.data[0].embedding
            logger.debug(f"Generated embedding with {len(embedding)} dimensions")
            return embedding
            
        except Exception as e:
            logger.error(f"❌ Error generating embedding: {e}")
            return None
    
    def generate_batch_embeddings(self, texts: List[str], 
                                 batch_size: Optional[int] = None) -> List[Optional[List[float]]]:
        """Generate embeddings for multiple texts with batching and rate limiting."""
        if batch_size is None:
            batch_size = processing_config.batch_size
        
        all_embeddings = []
        
        logger.info(f"🧮 Generating embeddings for {len(texts)} texts (batch size: {batch_size})")
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = []
            
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
            
            for j, text in enumerate(batch):
                try:
                    embedding = self.generate_embedding(text)
                    batch_embeddings.append(embedding)
                    
                    # Rate limiting to avoid API throttling
                    if j < len(batch) - 1:  # Don't sleep after last item in batch
                        time.sleep(0.1)  # 100ms between requests
                        
                except Exception as e:
                    logger.error(f"Failed to generate embedding for item {i + j}: {e}")
                    batch_embeddings.append(None)
            
            all_embeddings.extend(batch_embeddings)
            
            # Longer pause between batches
            if i + batch_size < len(texts):
                time.sleep(1)
        
        successful_count = sum(1 for emb in all_embeddings if emb is not None)
        logger.info(f"📊 Embedding generation complete: {successful_count}/{len(texts)} successful")
        
        return all_embeddings
    
    def _truncate_text(self, text: str, max_tokens: int = 8191) -> str:
        """Truncate text to fit within token limits."""
        # Simple approximation: ~4 characters per token for English text
        max_chars = max_tokens * 4
        
        if len(text) <= max_chars:
            return text
        
        truncated = text[:max_chars]
        logger.warning(f"Truncated text from {len(text)} to {len(truncated)} characters")
        return truncated

    def add_embeddings_to_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Add vector embeddings to document objects."""
        logger.info(f"🔄 Adding embeddings to {len(documents)} documents...")

        texts = [doc.get('content', '') for doc in documents]
        embeddings = self.generate_batch_embeddings(texts)

        enhanced_docs = []
        for doc, embedding in zip(documents, embeddings):
            if embedding is not None:
                doc_copy = doc.copy()
                doc_copy['content_vector'] = embedding
                # doc_copy['embedding_dimensions'] = len(embedding)  # ← REMOVE THIS LINE
                enhanced_docs.append(doc_copy)
                logger.debug(f"✅ Added embedding to document {doc['id']}")
            else:
                logger.warning(f"⚠️ Skipping document {doc['id']} due to missing embedding")

        logger.info(f"📊 Enhanced {len(enhanced_docs)}/{len(documents)} documents with embeddings")
        return enhanced_docs

    def validate_embedding(self, embedding: List[float]) -> bool:
        """Validate that embedding meets expected criteria."""
        if not embedding:
            return False
        
        if len(embedding) != processing_config.vector_dimensions:
            logger.warning(f"Unexpected embedding dimension: {len(embedding)} (expected: {processing_config.vector_dimensions})")
            return False
        
        # Check for NaN or infinite values
        if any(not (-1e6 < val < 1e6) for val in embedding):
            logger.warning("Embedding contains invalid values")
            return False
        
        return True 