import os
import requests
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

# Import the interactive search functionality
from interactive_search import start_interactive_search

# ================================================================================================
# STEP 1: ENVIRONMENT SETUP AND CONFIGURATION
# ================================================================================================

# Load environment variables from .env file
load_dotenv()

# Azure Search Service Configurations
AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_API_KEY = os.getenv("AZURE_SEARCH_API_KEY")
AZURE_SEARCH_INDEX_NAME = "library"

# Azure Document Intelligence Service Configurations
DOC_INTELLIGENCE_ENDPOINT = os.getenv("DOC_INTELLIGENCE_ENDPOINT")
DOC_INTELLIGENCE_KEY = os.getenv("DOC_INTELLIGENCE_KEY")

# Azure OpenAI Service Configurations
OPENAI_API_ENDPOINT = os.getenv("OPENAI_API_ENDPOINT")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_VERSION = "2024-10-21"
OPENAI_EMBEDDING_MODEL = "text-embedding-ada-002"

# URLs of PDFs to download and process
urls = [
    "https://www.gehealthcare.com/support/manuals?search=eyJzZWFyY2hUZXJtIjoiMjA5MjcwMy0wMDEiLCJsYW5ndWFnZU5hbWUiOiJFbmdsaXNoIChFTikifQ%3D%3D",
    "https://www.gehealthcare.com/support/manuals?search=eyJzZWFyY2hUZXJtIjoiMjEwNTUxOC0wMDEiLCJsYW5ndWFnZU5hbWUiOiJFbmdsaXNoIChFTikifQ%3D%3D",
    "https://www.gehealthcare.com/support/manuals?search=eyJzZWFyY2hUZXJtIjoiMjA1MDgwMi0wMDEiLCJsYW5ndWFnZU5hbWUiOiJFbmdsaXNoIChFTikifQ%3D%3D",
    "https://www.gehealthcare.com/support/manuals?search=eyJzZWFyY2hUZXJtIjoiMjA4MTQ5OS0wMDIiLCJsYW5ndWFnZU5hbWUiOiJFbmdsaXNoIChFTikifQ%3D%3D"
]


# ================================================================================================
# STEP 2: AZURE COGNITIVE SEARCH INDEX CREATION WITH VECTOR SEARCH CAPABILITIES
# ================================================================================================

def create_search_index():
    """Creates or updates Azure Cognitive Search index with vector search capabilities."""
    print("STEP 2: Creating Azure Cognitive Search Index...")

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
    print(f"Index '{AZURE_SEARCH_INDEX_NAME}' created/updated successfully.")


# ================================================================================================
# STEP 3: PDF DOCUMENT DOWNLOAD FROM URLS
# ================================================================================================

def download_pdf(url, dest_path):
    """Downloads a PDF file from URL and saves it to local path."""
    print(f"STEP 3: Downloading {url} to {dest_path}...")

    try:
        response = requests.get(url)
        if response.status_code == 200:
            with open(dest_path, "wb") as f:
                f.write(response.content)
            print("Download completed.")
            return True
        else:
            print(f"Failed to download {url}. Status code: {response.status_code}")
            return False
    except Exception as e:
        print(f"Error downloading {url}: {str(e)}")
        return False


# ================================================================================================
# STEP 4: CONTENT AND METADATA EXTRACTION USING AZURE DOCUMENT INTELLIGENCE
# ================================================================================================

def extract_content_with_doc_intelligence(file_path):
    """Extracts text content and metadata from PDF using Azure Document Intelligence."""
    print(f"STEP 4: Extracting content from {file_path} with Document Intelligence...")

    credential = AzureKeyCredential(DOC_INTELLIGENCE_KEY)
    client = DocumentAnalysisClient(endpoint=DOC_INTELLIGENCE_ENDPOINT, credential=credential)

    with open(file_path, "rb") as f_stream:
        poller = client.begin_analyze_document("prebuilt-document", document=f_stream)
        result = poller.result()

    text_lines = []
    for page in result.pages:
        for line in page.lines:
            text_lines.append(line.content)
    text = "\n".join(text_lines)

    metadata = {}
    if hasattr(result, "metadata") and result.metadata:
        metadata = {
            "author": result.metadata.author if hasattr(result.metadata, "author") else None,
            "title": result.metadata.title if hasattr(result.metadata, "title") else None,
            "creation_date": result.metadata.created_date if hasattr(result.metadata, "created_date") else None
        }

    print(f"Extracted {len(text)} characters of text content.")
    return text, metadata


# ================================================================================================
# STEP 5: VECTOR EMBEDDINGS GENERATION USING AZURE OPENAI
# ================================================================================================

def get_embedding(text):
    """Generates vector embeddings for text using Azure OpenAI embedding model."""
    print("STEP 5: Generating embeddings using Azure OpenAI...")

    client = AzureOpenAI(
        api_key=OPENAI_API_KEY,
        api_version=OPENAI_API_VERSION,
        azure_endpoint=OPENAI_API_ENDPOINT
    )

    try:
        response = client.embeddings.create(
            input=text,
            model=OPENAI_EMBEDDING_MODEL
        )
        embedding = response.data[0].embedding
        print(f"Generated embedding with {len(embedding)} dimensions.")
        return embedding
    except Exception as e:
        print(f"Error generating embedding: {e}")
        raise


# ================================================================================================
# STEP 6: DOCUMENT UPLOAD TO AZURE COGNITIVE SEARCH
# ================================================================================================

def upload_documents_to_search(documents):
    """Uploads processed documents to Azure Cognitive Search index."""
    print("STEP 6: Uploading documents to Azure Cognitive Search...")

    credential = AzureKeyCredential(AZURE_SEARCH_API_KEY)
    search_client = SearchClient(
        endpoint=AZURE_SEARCH_ENDPOINT,
        index_name=AZURE_SEARCH_INDEX_NAME,
        credential=credential
    )

    results = search_client.upload_documents(documents=documents)
    for i, result in enumerate(results):
        status = "Success" if result.succeeded else "Failed"
        print(f"Document {i + 1} upload: {status}")


# ================================================================================================
# STEP 7: EXAMPLE SEARCH OPERATIONS (KEYWORD, VECTOR, AND FILTERED SEARCH)
# ================================================================================================

def example_search(search_client):
    """Demonstrates keyword, vector, and filtered search operations on the index."""
    print("STEP 7: Running example search operations...")

    print("\n🔍 Keyword Search for 'anesthesia system':")
    results = search_client.search(search_text="anesthesia system")
    for r in results:
        print(f"- {r['filename']} - snippet: {r['content'][:100]}...")

    print("\nVector Search for 'training for device setup':")
    query_vector = get_embedding("training for device setup")

    from azure.search.documents.models import VectorizedQuery
    vector_query = VectorizedQuery(
        vector=query_vector,
        k_nearest_neighbors=3,
        fields="content_vector"
    )

    vector_results = search_client.search(
        search_text=None,
        vector_queries=[vector_query],
        select=["content", "filename", "document_url"]
    )

    for r in vector_results:
        print(f"- {r['filename']} - snippet: {r['content'][:100]}...")

    print("\nFiltered Search for documents with filename 'doc_1.pdf':")
    filtered_results = search_client.search(
        search_text="setup instructions",
        filter="filename eq 'doc_1.pdf'"
    )

    for r in filtered_results:
        print(f"- {r['filename']} - snippet: {r['content'][:100]}...")


# ================================================================================================
# STEP 8: MAIN PIPELINE - ORCHESTRATES THE ENTIRE DOCUMENT PROCESSING WORKFLOW
# ================================================================================================

def main():
    """Orchestrates the complete document processing pipeline from download to search interface."""
    create_search_index()
    print()

    documents_to_upload = []

    for i, url in enumerate(urls):
        print(f"\nProcessing Document {i + 1}/{len(urls)}")
        print("-" * 50)

        local_pdf_path = f"doc_{i + 1}.pdf"

        if download_pdf(url, local_pdf_path):
            text, metadata = extract_content_with_doc_intelligence(local_pdf_path)
            print(f'Content preview: {text[:200]}...\n')
            print(f'Metadata: {metadata}\n')

            embedding = get_embedding(text)
            print(f'Embedding preview: {embedding[:5]}... (showing first 5 dimensions)\n')

            doc = {
                "id": f"doc_{i + 1}",
                "content": text,
                "product_name": metadata.get("title", "Unknown Product"),
                "filename": os.path.basename(local_pdf_path),
                "filepath": os.path.abspath(local_pdf_path),
                "document_url": url,
                "content_vector": embedding
            }
            documents_to_upload.append(doc)
        else:
            print(f"Skipping document {i + 1} due to download failure.")

    print("\n" + "=" * 80)
    print("DOCUMENT PROCESSING SUMMARY")
    print("=" * 80)
    print(f"Total documents processed: {len(documents_to_upload)}")
    print("Document processing pipeline completed!")

    if documents_to_upload:
        print(f"\nUploading {len(documents_to_upload)} documents to Azure Cognitive Search...")
        upload_documents_to_search(documents_to_upload)
    else:
        print("No documents to upload.")

    start_interactive_search(
        search_endpoint=AZURE_SEARCH_ENDPOINT,
        search_api_key=AZURE_SEARCH_API_KEY,
        index_name=AZURE_SEARCH_INDEX_NAME
    )


# ================================================================================================
# ENTRY POINT
# ================================================================================================

if __name__ == "__main__":
    main()