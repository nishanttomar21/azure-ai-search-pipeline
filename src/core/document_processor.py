"""Document processing functionality."""
import os
import requests
from typing import Tuple, Dict, Any, Optional, List
from pathlib import Path
from src.core.azure_clients import AzureClientManager
from src.config.settings import processing_config
from src.utils.logger import get_logger
from src.utils.file_utils import FileManager

logger = get_logger(__name__)

class DocumentProcessor:
    """Handles document download and content extraction."""
    
    def __init__(self, client_manager: AzureClientManager):
        """Initialize with Azure client manager."""
        self.client_manager = client_manager
        self.file_manager = FileManager()
    
    def download_document(self, url: str, destination: str) -> bool:
        """Download document from URL to local path."""
        try:
            logger.info(f"📥 Downloading: {url}")
            
            response = requests.get(
                url,
                timeout=processing_config.request_timeout,
                stream=True,
                headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            )
            response.raise_for_status()
            
            # Ensure destination directory exists
            dest_path = Path(destination)
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Download with progress
            total_size = int(response.headers.get('content-length', 0))
            downloaded_size = 0
            
            with open(destination, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded_size += len(chunk)
            
            file_size = self.file_manager.get_file_size(destination)
            logger.info(f"✅ Downloaded {file_size} bytes to {destination}")
            return True
            
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Download failed for {url}: {e}")
            return False
        except Exception as e:
            logger.error(f"❌ Unexpected error downloading {url}: {e}")
            return False
    
    def extract_content(self, file_path: str) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        """Extract text content and metadata from PDF."""
        try:
            logger.info(f"📄 Extracting content from: {file_path}")
            
            if not self.file_manager.file_exists(file_path):
                logger.error(f"File does not exist: {file_path}")
                return None, None
            
            # Use Document Intelligence to extract content
            with open(file_path, "rb") as f:
                poller = self.client_manager.document_analysis_client.begin_analyze_document(
                    "prebuilt-document", 
                    document=f
                )
                result = poller.result()
            
            # Extract text content
            text_lines = []
            for page in result.pages:
                for line in page.lines:
                    if line.content and line.content.strip():
                        text_lines.append(line.content)
            
            text = "\n".join(text_lines)
            
            # Extract metadata
            metadata = self._extract_metadata(result)
            
            logger.info(f"✅ Extracted {len(text)} characters from {file_path}")
            return text, metadata
            
        except Exception as e:
            logger.error(f"❌ Content extraction failed for {file_path}: {e}")
            return None, None
    
    def _extract_metadata(self, result) -> Dict[str, Any]:
        """Extract metadata from document analysis result."""
        metadata = {}
        
        try:
            if hasattr(result, "metadata") and result.metadata:
                metadata = {
                    "author": getattr(result.metadata, "author", None),
                    "title": getattr(result.metadata, "title", None),
                    "creation_date": getattr(result.metadata, "created_date", None),
                    "subject": getattr(result.metadata, "subject", None),
                    "keywords": getattr(result.metadata, "keywords", None)
                }
                
            # Add document statistics
            if hasattr(result, "pages") and result.pages:
                metadata.update({
                    "page_count": len(result.pages),
                    "language": result.pages[0].language if hasattr(result.pages[0], "language") else None
                })
                
        except Exception as e:
            logger.warning(f"Error extracting metadata: {e}")
        
        return metadata
    
    def process_documents_batch(self, urls: List[str]) -> List[Dict[str, Any]]:
        """Process multiple documents in batch."""
        processed_docs = []
        
        logger.info(f"🔄 Processing {len(urls)} documents...")
        
        for i, url in enumerate(urls, 1):
            logger.info(f"📋 Processing document {i}/{len(urls)}")
            
            # Create temporary file
            temp_file = self.file_manager.create_temp_file(
                suffix=".pdf",
                prefix=f"doc_{i}_"
            )
            
            try:
                # Download document
                if not self.download_document(url, temp_file):
                    logger.warning(f"⚠️ Skipping document {i} due to download failure")
                    continue
                
                # Extract content
                text, metadata = self.extract_content(temp_file)
                if not text:
                    logger.warning(f"⚠️ Skipping document {i} due to extraction failure")
                    continue
                
                # Create document structure
                doc_data = {
                    "id": f"doc_{i}",
                    "content": text,
                    "product_name": metadata.get("title", "Unknown Product") if metadata else "Unknown Product",
                    "filename": f"doc_{i}.pdf",
                    "filepath": os.path.abspath(temp_file),
                    "document_url": url,
                    "content_length": len(text),
                    "processed_at": self._get_current_timestamp()
                }
                
                processed_docs.append(doc_data)
                logger.info(f"✅ Successfully processed document {i}")
                
            except Exception as e:
                logger.error(f"❌ Failed to process document {i}: {e}")
                continue
            finally:
                # Keep temp file for now (cleanup handled by FileManager)
                pass
        
        logger.info(f"📊 Batch processing complete: {len(processed_docs)}/{len(urls)} documents processed")
        return processed_docs

    def _get_current_timestamp(self) -> str:
        """Get current timestamp in Azure-compatible DateTimeOffset format."""
        from datetime import datetime, timezone
        # Azure expects: 2025-08-01T18:29:49Z format
        return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
    
    def cleanup_temp_files(self) -> None:
        """Clean up temporary files created during processing."""
        self.file_manager.cleanup_temp_files(["doc_*.pdf"]) 