"""
Main entry point for the Azure AI Search Pipeline.
Orchestrates the complete document processing and search setup workflow.
"""
import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.config.settings import azure_config, DOCUMENT_URLS
from src.core.azure_clients import AzureClientManager
from src.core.document_processor import DocumentProcessor
from src.core.embedding_generator import EmbeddingGenerator
from src.search.index_manager import IndexManager
from src.search.interactive_search import start_interactive_search
from src.utils.logger import setup_logging, get_logger

# Setup logging
setup_logging(level="INFO")
logger = get_logger(__name__)


class AzureSearchPipeline:
    """Main pipeline orchestrator for document processing and search setup."""

    def __init__(self):
        """Initialize the pipeline with required components."""
        # Validate configuration
        if not azure_config.validate():
            raise ValueError("❌ Azure configuration validation failed. Check your .env file.")

        # Initialize core components
        self.client_manager = AzureClientManager()
        self.document_processor = DocumentProcessor(self.client_manager)
        self.embedding_generator = EmbeddingGenerator(self.client_manager)
        self.index_manager = IndexManager(self.client_manager)

    def run(self) -> bool:
        """Execute the complete pipeline workflow."""
        try:
            logger.info("🚀 STARTING AZURE AI SEARCH PIPELINE")
            logger.info("=" * 50)

            # Step 1: Health check
            if not self._health_check():
                return False

            # Step 2: Create/update search index
            if not self._setup_search_index():
                return False

            # Step 3: Process documents
            processed_docs = self._process_documents()
            if not processed_docs:
                logger.error("❌ No documents were processed successfully")
                return False

            # Step 4: Generate embeddings
            enhanced_docs = self._generate_embeddings(processed_docs)
            if not enhanced_docs:
                logger.error("❌ No documents have valid embeddings")
                return False

            # Step 5: Upload to search index
            if not self._upload_to_search(enhanced_docs):
                return False

            # Step 6: Display summary
            self._display_summary(enhanced_docs)

            # Step 7: Start interactive search
            self._start_interactive_search()

            return True

        except KeyboardInterrupt:
            logger.info("Pipeline interrupted by user")
            return False
        except Exception as e:
            logger.error(f"❌ Pipeline execution failed: {e}")
            return False

    def _health_check(self) -> bool:
        """Perform health check on Azure services."""
        logger.info("🔍 Performing Azure services health check...")

        try:
            health_status = self.client_manager.health_check()

            all_healthy = all(health_status.values())
            if all_healthy:
                logger.info("✅ All Azure services are healthy")
                return True
            else:
                failed_services = [service for service, status in health_status.items() if not status]
                logger.error(f"❌ Health check failed for: {', '.join(failed_services)}")
                return False

        except Exception as e:
            logger.error(f"❌ Health check failed: {e}")
            return False

    def _setup_search_index(self) -> bool:
        """Create or update the Azure Search index with error handling."""
        logger.info("🔧 Setting up Azure Search index...")

        try:
            success = self.index_manager.create_or_update_index()
            if success:
                logger.info("✅ Search index setup completed")
                return True
        except Exception as e:
            if "Algorithm name cannot be updated" in str(e):
                logger.warning("⚠️ Index exists with different algorithm. Trying with new name...")

                # Try with timestamped index name
                import time
                old_name = azure_config.search_index_name
                azure_config.search_index_name = f"{old_name}_{int(time.time())}"

                # Reinitialize index manager with new name
                self.index_manager = IndexManager(self.client_manager)
                success = self.index_manager.create_or_update_index()

                if success:
                    logger.info(f"✅ Created index with new name: {azure_config.search_index_name}")
                    return True

        return False

    def _process_documents(self) -> list:
        """Process documents from URLs."""
        logger.info("📄 Starting document processing...")
        logger.info(f"Documents to process: {len(DOCUMENT_URLS)}")

        processed_docs = self.document_processor.process_documents_batch(DOCUMENT_URLS)

        if processed_docs:
            logger.info(f"✅ Document processing completed: {len(processed_docs)} documents")
        else:
            logger.error("❌ Document processing failed")

        return processed_docs

    def _generate_embeddings(self, documents: list) -> list:
        """Generate vector embeddings for documents."""
        logger.info("🧮 Generating vector embeddings...")

        enhanced_docs = self.embedding_generator.add_embeddings_to_documents(documents)

        if enhanced_docs:
            logger.info(f"✅ Embedding generation completed: {len(enhanced_docs)} documents")
        else:
            logger.error("❌ Embedding generation failed")

        return enhanced_docs

    def _upload_to_search(self, documents: list) -> bool:
        """Upload documents to Azure Search index."""
        logger.info("📤 Uploading documents to Azure Search...")

        success = self.index_manager.upload_documents(documents)

        if success:
            logger.info("✅ Document upload completed")
        else:
            logger.error("❌ Document upload failed")

        return success

    def _display_summary(self, documents: list):
        """Display pipeline execution summary."""
        logger.info("\n" + "=" * 50)
        logger.info("📋 PIPELINE EXECUTION SUMMARY")
        logger.info("=" * 50)
        logger.info(f"📄 Total documents processed: {len(documents)}")
        logger.info(f"🧮 Documents with embeddings: {len(documents)}")
        logger.info(f"📤 Documents uploaded to search: {len(documents)}")

        # Display document details
        for doc in documents:
            logger.info(f"  • {doc['id']}: {doc['filename']} ({doc['content_length']} chars)")

        logger.info("✅ Pipeline completed successfully!")
        logger.info("=" * 50)

    def _start_interactive_search(self):
        """Launch the interactive search interface."""
        logger.info("🎯 Starting Interactive Search Interface...")
        logger.info("Press Ctrl+C to stop the pipeline and exit")

        try:
            start_interactive_search(
                search_endpoint=azure_config.search_endpoint,
                search_api_key=azure_config.search_api_key,
                index_name=azure_config.search_index_name
            )
        except KeyboardInterrupt:
            logger.info("Interactive search stopped by user")
        except Exception as e:
            logger.error(f"❌ Interactive search failed: {e}")


def main():
    """Main entry point."""
    try:
        print("=" * 50)
        print("🔍 Azure AI Search Pipeline (Documents)")
        print("=" * 50)
        print("\n")

        pipeline = AzureSearchPipeline()
        success = pipeline.run()

        return 0 if success else 1

    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())