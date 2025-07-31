import os
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from openai import AzureOpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

OPENAI_API_ENDPOINT = os.getenv("OPENAI_API_ENDPOINT")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_VERSION = "2024-10-21"
OPENAI_EMBEDDING_MODEL = "text-embedding-ada-002"


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
            model=OPENAI_EMBEDDING_MODEL
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error generating embedding: {e}")
        raise


def extract_contextual_snippet(content, search_term, context_length=200):
    """Extract snippet with context around the search term"""
    if not content or not search_term:
        return content[:150] + "..." if len(content) > 150 else content

    # Find the search term (case-insensitive)
    content_lower = content.lower()
    search_lower = search_term.lower()

    index = content_lower.find(search_lower)
    if index == -1:
        # Term not found, return beginning
        return content[:150] + "..." if len(content) > 150 else content

    # Calculate start and end positions for context
    start = max(0, index - context_length // 2)
    end = min(len(content), index + len(search_term) + context_length // 2)

    # Extract the snippet
    snippet = content[start:end]

    # Add ellipsis if we're not at the beginning/end
    if start > 0:
        snippet = "..." + snippet
    if end < len(content):
        snippet = snippet + "..."

    # Highlight the search term (case-insensitive)
    import re
    snippet = re.sub(
        re.escape(search_term),
        f"🔥{search_term}🔥",
        snippet,
        flags=re.IGNORECASE
    )

    return snippet


class InteractiveSearchChat:
    def __init__(self, search_endpoint, search_api_key, index_name):
        """Initialize the interactive search chat"""
        self.search_endpoint = search_endpoint
        self.search_api_key = search_api_key
        self.index_name = index_name
        self.current_search_query = ""  # Store current search query for context

        credential = AzureKeyCredential(search_api_key)
        self.search_client = SearchClient(
            endpoint=search_endpoint,
            index_name=index_name,
            credential=credential
        )

    def start_chat(self):
        """Start the interactive chat interface"""
        print("\n" + "=" * 70)
        print("🔍 AZURE AI SEARCH - INTERACTIVE CHAT (DOCUMENT PROCESSING PIPELINE)")
        print("=" * 70)
        print("Welcome! You can search through your uploaded documents.")
        print("Type 'help' for available commands or 'quit' to exit.")
        print("-" * 70)

        while True:
            print("\n📝 Search Options:")
            print("1. Keyword Search (traditional text search)")
            print("2. Vector Search (semantic similarity)")
            print("3. Hybrid Search (keyword + vector)")
            print("4. Filtered Search (with specific filters)")
            print("5. List All Documents")
            print("6. Get Document by ID")
            print("Type 'help' for details or 'quit' to exit")

            user_input = input("\n💬 Choose an option (1-6) or enter search query: ").strip()

            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye! Thanks for using Azure AI Search!")
                break

            elif user_input.lower() == 'help':
                self.show_help()
                continue

            elif user_input == '1':
                self.perform_keyword_search()

            elif user_input == '2':
                self.perform_vector_search()

            elif user_input == '3':
                self.perform_hybrid_search()

            elif user_input == '4':
                self.perform_filtered_search()

            elif user_input == '5':
                self.list_all_documents()

            elif user_input == '6':
                self.get_document_by_id()

            else:
                # Direct search query
                print(f"\n🔍 Performing keyword search for: '{user_input}'")
                self.perform_direct_search(user_input)

    def show_help(self):
        """Display help information"""
        help_text = """
        📚 HELP - SEARCH COMMANDS:

        1. Keyword Search: Traditional text-based search using exact words
           - Good for: Finding specific terms, product names, procedures
           - Example: "anesthesia system", "maintenance", "setup"

        2. Vector Search: AI-powered semantic similarity search
           - Good for: Finding conceptually similar content
           - Example: "device training" might find "equipment education"

        3. Hybrid Search: Combines keyword and vector search
           - Good for: Best of both worlds - exact matches + semantic similarity

        4. Filtered Search: Search with specific document filters
           - Good for: Limiting results to specific files or products

        5. List All Documents: Show all uploaded documents

        6. Get Document by ID: Retrieve a specific document

        💡 Tips:
        - Use specific terms for keyword search
        - Use natural language for vector search
        - Try both approaches if you're not finding what you need
        - 🔥 text 🔥 indicates highlighted search terms in results
        """
        print(help_text)

    def perform_keyword_search(self):
        """Interactive keyword search with hit highlighting"""
        query = input("🔤 Enter your keyword search query: ").strip()
        if not query:
            print("❌ Please enter a search query.")
            return

        self.current_search_query = query
        print(f"\n🔍 Searching for: '{query}'")
        try:
            results = self.search_client.search(
                search_text=query,
                top=5,
                highlight_fields="content",
                highlight_pre_tag="🔥",
                highlight_post_tag="🔥"
            )
            self.display_search_results_with_highlights(results, "Keyword Search")
        except Exception as e:
            print(f"❌ Search error: {e}")

    def perform_vector_search(self):
        """Interactive vector search"""
        query = input("🧠 Enter your semantic search query: ").strip()
        if not query:
            print("❌ Please enter a search query.")
            return

        self.current_search_query = query
        print(f"\n🔍 Performing semantic search for: '{query}'")
        try:
            query_vector = get_embedding(query)

            vector_query = VectorizedQuery(
                vector=query_vector,
                k_nearest_neighbors=5,
                fields="content_vector"
            )

            results = self.search_client.search(
                search_text=None,
                vector_queries=[vector_query],
                select=["content", "filename", "document_url", "product_name"]
            )
            self.display_search_results_with_context(results, "Vector Search", show_score=True)
        except Exception as e:
            print(f"❌ Search error: {e}")

    def perform_hybrid_search(self):
        """Interactive hybrid search (keyword + vector) with highlighting"""
        query = input("🔄 Enter your hybrid search query: ").strip()
        if not query:
            print("❌ Please enter a search query.")
            return

        self.current_search_query = query
        print(f"\n🔍 Performing hybrid search for: '{query}'")
        try:
            query_vector = get_embedding(query)

            vector_query = VectorizedQuery(
                vector=query_vector,
                k_nearest_neighbors=5,
                fields="content_vector"
            )

            results = self.search_client.search(
                search_text=query,  # Keyword component
                vector_queries=[vector_query],  # Vector component
                select=["content", "filename", "document_url", "product_name"],
                top=5,
                highlight_fields="content",
                highlight_pre_tag="🔥",
                highlight_post_tag="🔥"
            )
            self.display_search_results_with_highlights(results, "Hybrid Search", show_score=True)
        except Exception as e:
            print(f"❌ Search error: {e}")

    def perform_filtered_search(self):
        """Interactive filtered search with highlighting"""
        print("\n📋 Available filter fields:")
        print("- filename (e.g., 'doc_1.pdf')")
        print("- product_name (e.g., 'Medical Device')")
        print("- document_url")

        filter_field = input("\n🎯 Enter filter field: ").strip()
        filter_value = input(f"🎯 Enter filter value for {filter_field}: ").strip()
        search_query = input("🔍 Enter search query (optional): ").strip()

        if not filter_field or not filter_value:
            print("❌ Please provide both filter field and value.")
            return

        self.current_search_query = search_query
        # Construct OData filter
        filter_expression = f"{filter_field} eq '{filter_value}'"

        try:
            if search_query:
                results = self.search_client.search(
                    search_text=search_query,
                    filter=filter_expression,
                    top=5,
                    highlight_fields="content",
                    highlight_pre_tag="🔥",
                    highlight_post_tag="🔥"
                )
                print(f"\n🔍 Filtered search for '{search_query}' where {filter_expression}")
                self.display_search_results_with_highlights(results, "Filtered Search")
            else:
                results = self.search_client.search(
                    search_text="*",
                    filter=filter_expression,
                    top=5
                )
                print(f"\n🔍 All documents where {filter_expression}")
                self.display_search_results_basic(results, "Filtered Search")

        except Exception as e:
            print(f"❌ Search error: {e}")

    def list_all_documents(self):
        """List all documents in the index"""
        print("\n📄 All Documents in Index:")
        try:
            results = self.search_client.search(
                search_text="*",
                select=["id", "filename", "product_name", "document_url"],
                top=50
            )

            count = 0
            for result in results:
                count += 1
                print(
                    f"{count}. ID: {result['id']} | File: {result['filename']} | Product: {result.get('product_name', 'N/A')}")

            if count == 0:
                print("📭 No documents found in the index.")
            else:
                print(f"\n📊 Total documents: {count}")

        except Exception as e:
            print(f"❌ Error listing documents: {e}")

    def get_document_by_id(self):
        """Get a specific document by ID"""
        doc_id = input("🆔 Enter document ID: ").strip()
        if not doc_id:
            print("❌ Please enter a document ID.")
            return

        try:
            document = self.search_client.get_document(key=doc_id)
            print(f"\n📄 Document: {doc_id}")
            print(f"📁 Filename: {document.get('filename', 'N/A')}")
            print(f"🏷️ Product: {document.get('product_name', 'N/A')}")
            print(f"🔗 URL: {document.get('document_url', 'N/A')}")
            print(f"📝 Content Preview: {document.get('content', 'N/A')[:200]}...")

        except Exception as e:
            print(f"❌ Document not found or error: {e}")

    def perform_direct_search(self, query):
        """Perform a direct keyword search from user input with highlighting"""
        self.current_search_query = query
        try:
            results = self.search_client.search(
                search_text=query,
                top=5,
                highlight_fields="content",
                highlight_pre_tag="🔥",
                highlight_post_tag="🔥"
            )
            self.display_search_results_with_highlights(results, f"Search for '{query}'")
        except Exception as e:
            print(f"❌ Search error: {e}")

    def display_search_results_with_highlights(self, results, search_type, show_score=False):
        """Display search results with Azure AI Search highlighting - with null safety"""
        print(f"\n📋 {search_type} Results:")
        print("-" * 50)

        count = 0
        for result in results:
            count += 1
            score_text = f" (Score: {result['@search.score']:.3f})" if show_score and '@search.score' in result else ""

            print(f"\n{count}. 📄 {result.get('filename', 'Unknown file')}{score_text}")
            print(f"   🏷️ Product: {result.get('product_name', 'N/A')}")

            try:
                # Use highlighted content if available
                if ('@search.highlights' in result and
                        result['@search.highlights'] is not None and
                        'content' in result['@search.highlights'] and
                        result['@search.highlights']['content'] is not None):

                    highlights = result['@search.highlights']['content']
                    print(f"   🎯 Relevant snippets:")
                    for i, highlight in enumerate(highlights[:2]):  # Show top 2 highlights
                        if highlight:  # Check highlight is not None
                            print(f"      {i + 1}. {highlight}")
                else:
                    # Fallback to contextual snippet
                    content = result.get('content', '')
                    if content and self.current_search_query:
                        preview = extract_contextual_snippet(content, self.current_search_query)
                    elif content:
                        preview = content[:150] + "..." if len(content) > 150 else content
                    else:
                        preview = "No content available"
                    print(f"   📝 Preview: {preview}")

            except Exception as e:
                print(f"   ❌ Error processing result: {e}")
                # Show basic content as fallback
                content = result.get('content', 'No content available')
                if content and content != 'No content available':
                    preview = content[:150] + "..." if len(content) > 150 else content
                    print(f"   📝 Fallback Preview: {preview}")

            print(f"   🔗 URL: {result.get('document_url', 'N/A')}")

        if count == 0:
            print("🔍 No results found. Try a different search query or search type.")
        else:
            print(f"\n📊 Found {count} result(s)")

    def display_search_results_with_context(self, results, search_type, show_score=False):
        """Display search results with contextual snippets (for vector search)"""
        print(f"\n📋 {search_type} Results:")
        print("-" * 50)

        count = 0
        for result in results:
            count += 1
            score_text = f" (Score: {result['@search.score']:.3f})" if show_score and '@search.score' in result else ""

            print(f"\n{count}. 📄 {result.get('filename', 'Unknown file')}{score_text}")
            print(f"   🏷️ Product: {result.get('product_name', 'N/A')}")

            # For vector search, show contextual snippet based on search query
            content = result.get('content', '')
            if self.current_search_query:
                preview = extract_contextual_snippet(content, self.current_search_query)
            else:
                preview = content[:150] + "..." if len(content) > 150 else content
            print(f"   📝 Relevant content: {preview}")

            print(f"   🔗 URL: {result.get('document_url', 'N/A')}")

        if count == 0:
            print("🔍 No results found. Try a different search query or search type.")
        else:
            print(f"\n📊 Found {count} result(s)")

    def display_search_results_basic(self, results, search_type, show_score=False):
        """Basic display for results without search context"""
        print(f"\n📋 {search_type} Results:")
        print("-" * 50)

        count = 0
        for result in results:
            count += 1
            score_text = f" (Score: {result['@search.score']:.3f})" if show_score and '@search.score' in result else ""

            print(f"\n{count}. 📄 {result.get('filename', 'Unknown file')}{score_text}")
            print(f"   🏷️ Product: {result.get('product_name', 'N/A')}")

            # Show basic preview
            content = result.get('content', '')
            preview = content[:150] + "..." if len(content) > 150 else content
            print(f"   📝 Preview: {preview}")

            print(f"   🔗 URL: {result.get('document_url', 'N/A')}")

        if count == 0:
            print("🔍 No results found. Try a different search query or search type.")
        else:
            print(f"\n📊 Found {count} result(s)")


# Convenience function for easy import
def start_interactive_search(search_endpoint, search_api_key, index_name):
    """Start the interactive search chat"""
    chat = InteractiveSearchChat(search_endpoint, search_api_key, index_name)
    chat.start_chat()