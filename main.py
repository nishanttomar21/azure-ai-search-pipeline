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

# Load environment variables from .env file
load_dotenv()

# Azure Configurations
AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_API_KEY = os.getenv("AZURE_SEARCH_API_KEY")
AZURE_SEARCH_INDEX_NAME = "library"

DOC_INTELLIGENCE_ENDPOINT = os.getenv("DOC_INTELLIGENCE_ENDPOINT")
DOC_INTELLIGENCE_KEY = os.getenv("DOC_INTELLIGENCE_KEY")

OPENAI_API_ENDPOINT = os.getenv("OPENAI_API_ENDPOINT")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_VERSION = "2024-10-21"
#OPENAI_EMBEDDING_MODEL = "text-embedding-3-large"
OPENAI_EMBEDDING_MODEL = "text-embedding-ada-002"

# URLs of PDFs to download
urls = [
    "https://www.gehealthcare.com/support/manuals?search=eyJzZWFyY2hUZXJtIjoiMjA5MjcwMy0wMDEiLCJsYW5ndWFnZU5hbWUiOiJFbmdsaXNoIChFTikifQ%3D%3D",
    "https://www.gehealthcare.com/support/manuals?search=eyJzZWFyY2hUZXJtIjoiMjEwNTUxOC0wMDEiLCJsYW5ndWFnZU5hbWUiOiJFbmdsaXNoIChFTikifQ%3D%3D",
    "https://www.gehealthcare.com/support/manuals?search=eyJzZWFyY2hUZXJtIjoiMjA1MDgwMi0wMDEiLCJsYW5ndWFnZU5hbWUiOiJFbmdsaXNoIChFTikifQ%3D%3D",
    "https://www.gehealthcare.com/support/manuals?search=eyJzZWFyY2hUZXJtIjoiMjA4MTQ5OS0wMDIiLCJsYW5ndWFnZU5hbWUiOiJFbmdsaXNoIChFTikifQ%3D%3D"
]

# Dummy Documents
# urls = [
#     "https://drive.google.com/uc?export=download&id=1oFOzMI_0B8lMqXePmBIRJ5DsYSDNR1sM",
#     "https://drive.google.com/uc?export=download&id=14tog4mFflzasQhFpDxaeMCK-XoqeHLJ9"
# ]

# --- Step 1: Create Azure Cognitive Search Index with Vector Search ---

def create_search_index():
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

    # Create or update the index
    index_client.create_or_update_index(index=index)
    print(f"Index '{AZURE_SEARCH_INDEX_NAME}' created/updated successfully.")

# --- Step 2: Download PDFs from URLs ---

def download_pdf(url, dest_path):
    print(f"Downloading {url} to {dest_path}...")
    response = requests.get(url)
    if response.status_code == 200:
        with open(dest_path, "wb") as f:
            f.write(response.content)
        print("Download completed.")
        return True
    else:
        print(f"Failed to download {url}. Status code: {response.status_code}")
        return False

# --- Step 3: Use Azure Document Intelligence to extract content and metadata ---

def extract_content_with_doc_intelligence(file_path):
    credential = AzureKeyCredential(DOC_INTELLIGENCE_KEY)
    client = DocumentAnalysisClient(endpoint=DOC_INTELLIGENCE_ENDPOINT, credential=credential)

    with open(file_path, "rb") as f_stream:
        poller = client.begin_analyze_document("prebuilt-document", document=f_stream)
        result = poller.result()

    # Extract text content (concatenate lines from all pages)
    text_lines = []
    for page in result.pages:
        for line in page.lines:
            text_lines.append(line.content)
    text = "\n".join(text_lines)

    # Extract metadata fields (some may be None if not present)
    metadata = {}
    if hasattr(result, "metadata") and result.metadata:
        metadata = {
            "author": result.metadata.author if hasattr(result.metadata, "author") else None,
            "title": result.metadata.title if hasattr(result.metadata, "title") else None,
            "creation_date": result.metadata.created_date if hasattr(result.metadata, "created_date") else None
        }

    return text, metadata

# --- Step 4: Generate Embeddings for content using Azure OpenAI ---

def get_embedding(text):
    # Initialize Azure OpenAI client with new syntax
    client = AzureOpenAI(
        api_key=OPENAI_API_KEY,
        api_version=OPENAI_API_VERSION,
        azure_endpoint=OPENAI_API_ENDPOINT
    )

    try:
        # Use new client syntax for embeddings
        response = client.embeddings.create(
            input=text,
            model=OPENAI_EMBEDDING_MODEL  # Note: 'model' instead of 'engine'
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error generating embedding: {e}")
        raise

# --- Step 5: Upload documents to Azure Cognitive Search ---

def upload_documents_to_search(documents):
    credential = AzureKeyCredential(AZURE_SEARCH_API_KEY)
    search_client = SearchClient(
        endpoint=AZURE_SEARCH_ENDPOINT,
        index_name=AZURE_SEARCH_INDEX_NAME,
        credential=credential
    )
    results = search_client.upload_documents(documents=documents)
    for i, r in enumerate(results):
        print(f"Document {i+1} upload success: {r.succeeded}")

# --- Step 6: (Optional) Search examples ---

def example_search(search_client):
    print("\nKeyword Search for 'anesthesia system':")
    results = search_client.search(search_text="anesthesia system")
    for r in results:
        print(f"- {r['filename']} - snippet: {r['content'][:100]}")

    print("\nVector Search for 'training for device setup':")
    query_vector = get_embedding("training for device setup")
    from azure.search.documents.models import VectorizedQuery
    vector_query = VectorizedQuery(
        vector=query_vector, k_nearest_neighbors=3, fields="content_vector"
    )
    vector_results = search_client.search(
        search_text=None,
        vector_queries=[vector_query],
        select=["content", "filename", "document_url"]
    )
    for r in vector_results:
        print(f"- {r['filename']} - snippet: {r['content'][:100]}")

    print("\nFiltered Search for documents with filename 'doc_1.pdf':")
    filtered_results = search_client.search(
        search_text="setup instructions",
        filter="filename eq 'doc_1.pdf'"
    )
    for r in filtered_results:
        print(f"- {r['filename']} - snippet: {r['content'][:100]}")

# --- Main Pipeline ---

def main():
    # Step 1: Create Index (Schema)
    create_search_index()

    # Step 2 & 3 & 4 & 5: Process each document
    documents_to_upload = []
    for i, url in enumerate(urls):
        local_pdf_path = f"doc_{i+1}.pdf"
        if download_pdf(url, local_pdf_path):
            print(f"Extracting content from {local_pdf_path} with Document Intelligence...")
            text, metadata = extract_content_with_doc_intelligence(local_pdf_path)
            print(f'Content{i+1}:  {text}\n')
            print(f'Metadata{i+1}:  {metadata}\n')
            print("Generating embedding...")
            embedding = get_embedding(text)
            print(f'Embedding{i+1}:  {embedding}\n')

            doc = {
                "id": f"doc_{i+1}",
                "content": text,
                "product_name": metadata.get("title", "Unknown Product"),
                "filename": os.path.basename(local_pdf_path),
                "filepath": os.path.abspath(local_pdf_path),
                "document_url": url,
                "content_vector": embedding
            }
            documents_to_upload.append(doc)
        else:
            print(f"Skipping document {i+1} due to download failure.")

    print("Document processing complete!")
    print("documents_to_upload: ", documents_to_upload)

    if documents_to_upload:
        print(f"\nUploading {len(documents_to_upload)} documents to Azure Cognitive Search...")
        upload_documents_to_search(documents_to_upload)
    else:
        print("No documents to upload.")

    # Step 6: Start Interactive Search Chat
    print("\n🚀 Starting Interactive Search Chat...")
    start_interactive_search(
        search_endpoint=AZURE_SEARCH_ENDPOINT,
        search_api_key=AZURE_SEARCH_API_KEY,
        index_name=AZURE_SEARCH_INDEX_NAME
    )

if __name__ == "__main__":
    main()