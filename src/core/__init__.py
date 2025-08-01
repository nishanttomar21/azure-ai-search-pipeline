"""Core package initialization."""
from .azure_clients import AzureClientManager
from .document_processor import DocumentProcessor
from .embedding_generator import EmbeddingGenerator

__all__ = ['AzureClientManager', 'DocumentProcessor', 'EmbeddingGenerator'] 