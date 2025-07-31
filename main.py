"""
Azure AI Search Pipeline - Simple document processing and search
"""

import os
import requests
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime

from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex, SimpleField, SearchField, SearchFieldDataType,
    SearchableField, VectorSearch, VectorSearchProfile, HnswAlgorithmConfiguration
)
from azure.search.documents import SearchClient
from azure.ai.formrecognizer import DocumentAnalysisClient
from openai import AzureOpenAI
from dotenv import load_dotenv

from interactive_search import start_interactive_search

# Load environment variables
load_dotenv()

# Configuration
AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_API_KEY = os.getenv("AZURE_SEARCH_API_KEY")
AZURE_SEARCH_INDEX_NAME = "library"

DOC_INTELLIGENCE_ENDPOINT = os.getenv("DOC_INTELLIGENCE_ENDPOINT")
DOC_INTELLIGENCE_KEY = os.getenv("DOC_INTELLIGENCE_KEY")

OPENAI_API_ENDPOINT = os.getenv("OPENAI_API_ENDPOINT")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_VERSION = "2024-10-21"
OPENAI_EMBEDDING_MODEL = "text-embedding-ada-002"

DOCUMENT_URLS = [
    "https://drive.google.com/uc?export=download&id=1oFOzMI_0B8lMqXePmBIRJ5DsYSDNR1sM",
    "https://drive.google.com/uc?export=download&id=14tog4mFflzasQhFpDxaeMCK-XoqeHLJ9"
]


def create_search_index():
    """Create or update Azure AI Search index with vector capabilities"""
    credential = AzureKeyCredential(AZURE_SEARCH_API_KEY)
    index_client = SearchIndexClient(endpoint=AZURE_SEARCH_ENDPOINT, credential=credential)

    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True, filterable=True, sortable=True),
        SearchableField(name="content", type=SearchFieldDataType.String),
        SimpleField(name="product_name", type=SearchFieldDataType.String, filterable=True, sortable=True),
        SimpleField(name="filename", type=SearchFieldDataType.String, filterable=True, sortable=True),
        SimpleField(name="filepath", type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="document_url", type=SearchFieldDataType.String, filterable=True),
        SearchField(
            name="content_vector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=1536,
            vector_search_profile_name="vector-config"
        ),
    ]

    vector_search = VectorSearch(
        profiles=[VectorSearchProfile(name="vector-config", algorithm_configuration_name="algo-config")],
        algorithms=[HnswAlgorithmConfiguration(name="algo-config")]
    )

    index = SearchIndex(name=AZURE_SEARCH_INDEX_NAME, fields=fields, vector_search=vector_search)
    index_client.create_or_update_index(index=index)
    print(f"✅ Index '{AZURE_SEARCH_INDEX_NAME}' created/updated successfully")


def download_pdf(url: str, dest_path: str):
    """Download PDF file from URL to local path"""
    print(f"📥 Downloading {url} to {dest_path}...")
    response = requests.get(url, timeout=30)
    if response.status_code == 200:
        with open(dest_path, "wb") as f:
            f.write(response.content)

        # Check if it's a valid PDF
        with open(dest_path, "rb") as f:
            if f.read(4) == b'%PDF':
                print("✅ Download completed")
                return True
            else:
                print("❌ Downloaded file is not a valid PDF")
                return False
    else:
        print(f"❌ Failed to download {url}. Status code: {response.status_code}")
        return False


def extract_content_with_doc_intelligence(file_path: str):
    """Extract text content and metadata from PDF using Azure Document Intelligence"""
    print(f"🔍 Extracting content from {file_path}...")

    credential = AzureKeyCredential(DOC_INTELLIGENCE_KEY)
    client = DocumentAnalysisClient(endpoint=DOC_INTELLIGENCE_ENDPOINT, credential=credential)

    with open(file_path, "rb") as f_stream:
        poller = client.begin_analyze_document("prebuilt-document", document=f_stream)
        result = poller.result()

    # Extract text content
    text_lines = []
    for page in result.pages:
        for line in page.lines:
            text_lines.append(line.content)
    text = "\n".join(text_lines)

    # Extract metadata
    metadata = {}
    if hasattr(result, "metadata") and result.metadata:
        metadata = {
            "author": getattr(result.metadata, "author", None),
            "title": getattr(result.metadata, "title", None),
            "creation_date": getattr(result.metadata, "created_date", None)
        }

    # Filter out None values
    metadata = {k: v for k, v in metadata.items() if v is not None}

    print(f"✅ Extracted {len(text)} characters and {len(metadata)} metadata fields")
    return text, metadata


def get_embedding(text: str):
    """Generate embedding vector for text using Azure OpenAI"""
    # Truncate if too long
    if len(text) > 8000:
        text = text[:8000]
        print("⚠️ Text truncated for embedding")

    client = AzureOpenAI(
        api_key=OPENAI_API_KEY,
        api_version=OPENAI_API_VERSION,
        azure_endpoint=OPENAI_API_ENDPOINT
    )

    response = client.embeddings.create(
        input=text,
        model=OPENAI_EMBEDDING_MODEL
    )

    embedding = response.data[0].embedding
    print(f"✅ Generated {len(embedding)}-dimensional embedding")
    return embedding


def upload_documents_to_search(documents):
    """Upload processed documents to Azure AI Search index"""
    print(f"📤 Uploading {len(documents)} documents to search index...")

    credential = AzureKeyCredential(AZURE_SEARCH_API_KEY)
    search_client = SearchClient(
        endpoint=AZURE_SEARCH_ENDPOINT,
        index_name=AZURE_SEARCH_INDEX_NAME,
        credential=credential
    )

    results = search_client.upload_documents(documents=documents)

    success_count = 0
    for i, r in enumerate(results):
        if r.succeeded:
            success_count += 1
        print(f"Document {i + 1} upload success: {r.succeeded}")

    print(f"✅ Successfully uploaded {success_count}/{len(documents)} documents")
    return success_count == len(documents)


def process_single_document(url: str, doc_id: str):
    """Process a single document through the complete pipeline"""
    local_path = f"{doc_id}.pdf"

    try:
        # Download document
        if not download_pdf(url, local_path):
            return None

        # Extract content and metadata
        content, metadata = extract_content_with_doc_intelligence(local_path)

        # Generate embedding
        print(f"🧠 Generating embedding for {doc_id}...")
        embedding = get_embedding(content)

        # Prepare document for index
        document = {
            "id": doc_id,
            "content": content,
            "product_name": metadata.get("title", "Unknown Product"),
            "filename": os.path.basename(local_path),
            "filepath": os.path.abspath(local_path),
            "document_url": url,
            "content_vector": embedding
        }

        print(f"✅ Successfully processed {doc_id}")
        return document

    except Exception as e:
        print(f"❌ Failed to process {doc_id}: {e}")
        return None

    finally:
        # Clean up temporary file
        if os.path.exists(local_path):
            os.remove(local_path)


def run_pipeline():
    """Run the complete document processing pipeline"""
    print("🚀 Starting Azure AI Search Pipeline")

    # Step 1: Create search index
    create_search_index()

    # Step 2: Process all documents
    documents_to_upload = []
    for i, url in enumerate(DOCUMENT_URLS, 1):
        doc_id = f"doc_{i}"
        processed_doc = process_single_document(url, doc_id)

        if processed_doc:
            documents_to_upload.append(processed_doc)

    # Step 3: Upload documents
    if documents_to_upload:
        if upload_documents_to_search(documents_to_upload):
            print("🎉 Pipeline completed successfully!")
            return True
        else:
            print("❌ Some documents failed to upload")
            return False
    else:
        print("❌ No documents were processed successfully")
        return False


def main():
    """Main application entry point"""
    print("🚀 Azure AI Search Pipeline")
    print("=" * 50)

    if run_pipeline():
        print("\n🔍 Starting Interactive Search...")
        start_interactive_search(
            search_endpoint=AZURE_SEARCH_ENDPOINT,
            search_api_key=AZURE_SEARCH_API_KEY,
            index_name=AZURE_SEARCH_INDEX_NAME
        )
    else:
        print("❌ Pipeline failed. Check configuration and try again.")


if __name__ == "__main__":
    main()