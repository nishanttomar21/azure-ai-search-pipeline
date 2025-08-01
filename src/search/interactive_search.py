"""Interactive search interface functionality."""
from typing import List, Dict, Any, Optional
from azure.search.documents.models import VectorizedQuery
from src.core.azure_clients import AzureClientManager
from src.core.embedding_generator import EmbeddingGenerator
from src.utils.logger import get_logger

logger = get_logger(__name__)

class InteractiveSearchInterface:
    """Enhanced interactive search chat interface."""
    
    def __init__(self, client_manager: AzureClientManager, embedding_generator: EmbeddingGenerator):
        """Initialize with Azure client manager and embedding generator."""
        self.client_manager = client_manager
        self.embedding_generator = embedding_generator
        self.current_query = ""
    
    def start_chat(self):
        """Start the interactive chat interface."""
        self._print_welcome()
        
        while True:
            try:
                self._print_menu()
                choice = input("\n💬 Choose option (1-7) or enter query: ").strip()
                
                if self._should_exit(choice):
                    break
                elif choice == 'help':
                    self._show_help()
                elif choice.isdigit() and 1 <= int(choice) <= 7:
                    self._handle_menu_choice(int(choice))
                else:
                    self._perform_direct_search(choice)
                    
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                logger.error(f"Unexpected error in chat: {e}")
                print(f"❌ An error occurred: {e}")
    
    def _print_welcome(self):
        """Print welcome message."""
        print("\n" + "=" * 70)
        print("🔍 AZURE AI SEARCH - INTERACTIVE CHAT INTERFACE")
        print("=" * 70)
        print("Search through your documents using AI-powered capabilities.")
        print("Type 'help' for detailed commands or 'quit' to exit.")
        print("-" * 70)
    
    def _print_menu(self):
        """Print the main menu options."""
        print("\n📝 Search Options:")
        print("1. Keyword Search (traditional text-based)")
        print("2. Vector Search (semantic similarity)")
        print("3. Hybrid Search (keyword + vector combined)")
        print("4. Filtered Search (with metadata filters)")
        print("5. List All Documents")
        print("6. Get Document by ID")
        print("7. Index Statistics")
    
    def _should_exit(self, input_str: str) -> bool:
        """Check if user wants to exit."""
        return input_str.lower() in ['quit', 'exit', 'q', 'bye']
    
    def _handle_menu_choice(self, choice: int):
        """Handle numeric menu choices."""
        handlers = {
            1: self._keyword_search,
            2: self._vector_search,
            3: self._hybrid_search,
            4: self._filtered_search,
            5: self._list_documents,
            6: self._get_document_by_id,
            7: self._show_index_stats
        }
        
        handler = handlers.get(choice)
        if handler:
            handler()
        else:
            print("❌ Invalid option. Please choose 1-7.")
    
    def _keyword_search(self):
        """Perform keyword search with highlighting."""
        query = self._get_search_input("🔤 Enter keyword search query: ")
        if not query:
            return
        
        self.current_query = query
        print(f"\n🔍 Searching for: '{query}'")
        
        try:
            results = self.client_manager.search_client.search(
                search_text=query,
                top=5,
                highlight_fields="content",
                highlight_pre_tag="🔥",
                highlight_post_tag="🔥",
                select=["id", "content", "filename", "product_name", "document_url"]
            )
            
            self._display_results(list(results), "Keyword Search", show_highlights=True)
            
        except Exception as e:
            logger.error(f"Keyword search failed: {e}")
            print(f"❌ Search failed: {e}")
    
    def _vector_search(self):
        """Perform semantic vector search."""
        query = self._get_search_input("🧠 Enter semantic search query: ")
        if not query:
            return
        
        self.current_query = query
        print(f"\n🔍 Performing semantic search for: '{query}'")
        
        try:
            # Generate query embedding
            query_vector = self.embedding_generator.generate_embedding(query)
            if not query_vector:
                print("❌ Failed to generate embedding for query")
                return
            
            # Perform vector search
            vector_query = VectorizedQuery(
                vector=query_vector,
                k_nearest_neighbors=5,
                fields="content_vector"
            )
            
            results = self.client_manager.search_client.search(
                search_text=None,
                vector_queries=[vector_query],
                select=["id", "content", "filename", "product_name", "document_url"]
            )
            
            self._display_results(list(results), "Vector Search", show_score=True)
            
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            print(f"❌ Search failed: {e}")
    
    def _hybrid_search(self):
        """Perform hybrid search (keyword + vector)."""
        query = self._get_search_input("🔄 Enter hybrid search query: ")
        if not query:
            return
        
        self.current_query = query
        print(f"\n🔍 Performing hybrid search for: '{query}'")
        
        try:
            # Generate query embedding
            query_vector = self.embedding_generator.generate_embedding(query)
            if not query_vector:
                print("❌ Failed to generate embedding for query")
                return
            
            # Perform hybrid search
            vector_query = VectorizedQuery(
                vector=query_vector,
                k_nearest_neighbors=5,
                fields="content_vector"
            )
            
            results = self.client_manager.search_client.search(
                search_text=query,
                vector_queries=[vector_query],
                top=5,
                highlight_fields="content",
                highlight_pre_tag="🔥",
                highlight_post_tag="🔥",
                select=["id", "content", "filename", "product_name", "document_url"]
            )
            
            self._display_results(list(results), "Hybrid Search", show_highlights=True, show_score=True)
            
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            print(f"❌ Search failed: {e}")
    
    def _get_search_input(self, prompt: str) -> Optional[str]:
        """Get and validate search input."""
        query = input(prompt).strip()
        if not query:
            print("❌ Please enter a search query.")
            return None
        return query
    
    def _display_results(self, results: List[Dict], search_type: str,
                        show_highlights: bool = False, show_score: bool = False):
        """Display search results with proper formatting."""
        print(f"\n📋 {search_type} Results:")
        print("-" * 50)
        
        if not results:
            print("🔍 No results found. Try a different query or search type.")
            print("\n💡 Tips:")
            print("  • Use specific terms for keyword search")
            print("  • Use natural language for semantic search")
            print("  • Try different search types if needed")
            return
        
        for i, result in enumerate(results, 1):
            self._display_single_result(result, i, show_highlights, show_score)
        
        print(f"\n📊 Found {len(results)} result(s)")
    
    def _display_single_result(self, result: Dict, index: int,
                              show_highlights: bool, show_score: bool):
        """Display a single search result."""
        # Score display
        score_text = ""
        if show_score and '@search.score' in result:
            score_text = f" (Score: {result['@search.score']:.3f})"
        
        print(f"\n{index}. 📄 {result.get('filename', 'Unknown file')}{score_text}")
        print(f"   🏷️ Product: {result.get('product_name', 'N/A')}")
        print(f"   🆔 ID: {result.get('id', 'N/A')}")
        
        # Content display
        if show_highlights and self._has_highlights(result):
            self._display_highlights(result)
        else:
            self._display_content_preview(result)
        
        print(f"   🔗 URL: {result.get('document_url', 'N/A')}")
    
    def _has_highlights(self, result: Dict) -> bool:
        """Check if result has highlighting information."""
        return ('@search.highlights' in result and
                result['@search.highlights'] is not None and
                'content' in result['@search.highlights'] and
                result['@search.highlights']['content'])
    
    def _display_highlights(self, result: Dict):
        """Display highlighted content snippets."""
        highlights = result['@search.highlights']['content']
        print(f"   🎯 Relevant snippets:")
        for i, highlight in enumerate(highlights[:3], 1):  # Show top 3 highlights
            if highlight and highlight.strip():
                clean_highlight = highlight.strip()
                print(f"      {i}. {clean_highlight}")
    
    def _display_content_preview(self, result: Dict):
        """Display basic content preview."""
        content = result.get('content', '')
        if content:
            # Create contextual snippet if we have a current query
            if self.current_query:
                preview = self._extract_contextual_snippet(content, self.current_query)
            else:
                preview = content[:200] + "..." if len(content) > 200 else content
            print(f"   📝 Preview: {preview}")
        else:
            print(f"   📝 Preview: No content available")
    
    def _extract_contextual_snippet(self, content: str, query: str, context_length: int = 150) -> str:
        """Extract snippet with context around the search term."""
        if not content or not query:
            return content[:150] + "..." if len(content) > 150 else content
        
        # Find the query term (case-insensitive)
        content_lower = content.lower()
        query_lower = query.lower()
        
        index = content_lower.find(query_lower)
        if index == -1:
            return content[:150] + "..." if len(content) > 150 else content
        
        # Calculate context window
        start = max(0, index - context_length // 2)
        end = min(len(content), index + len(query) + context_length // 2)
        
        snippet = content[start:end]
        
        # Add ellipsis
        if start > 0:
            snippet = "..." + snippet
        if end < len(content):
            snippet = snippet + "..."
        
        return snippet
    
    def _show_help(self):
        """Display comprehensive help information."""
        help_text = """
        📚 SEARCH HELP & COMMANDS

        🔍 SEARCH TYPES:
        1. Keyword Search    - Traditional text matching
        2. Vector Search     - AI semantic similarity  
        3. Hybrid Search     - Best of both approaches
        4. Filtered Search   - Search with metadata filters
        
        💡 SEARCH TIPS:
        • Keyword: Use specific terms, product names, exact phrases
        • Vector: Use natural language, concepts, questions
        • Hybrid: Combines exact matches with semantic understanding
        
        🎯 EXAMPLES:
        • "maintenance procedure" (keyword)
        • "How do I troubleshoot the device?" (vector) 
        • "setup anesthesia equipment" (hybrid)
        
        📋 OTHER COMMANDS:
        • 'help' - Show this help message
        • 'quit' - Exit the application
        • Numbers 1-7 - Select menu options
        
        ⚡ SHORTCUTS:
        • Just type your query directly for keyword search
        """
        print(help_text)
        
    # Additional methods for other menu options...
    def _filtered_search(self):
        """Placeholder for filtered search."""
        print("🚧 Filtered search - Coming soon!")
    
    def _list_documents(self):
        """List all documents in the index."""
        print("\n📄 All Documents in Index:")
        try:
            results = self.client_manager.search_client.search(
                search_text="*",
                select=["id", "filename", "product_name", "content_length"],
                top=20
            )
            
            count = 0
            for result in results:
                count += 1
                content_len = result.get('content_length', 'N/A')
                print(f"{count:2d}. {result['id']} | {result['filename']} | "
                      f"Product: {result.get('product_name', 'N/A')} | "
                      f"Size: {content_len} chars")
            
            if count == 0:
                print("📭 No documents found in the index.")
            else:
                print(f"\n📊 Total documents: {count}")
                
        except Exception as e:
            logger.error(f"Failed to list documents: {e}")
            print(f"❌ Error listing documents: {e}")
    
    def _get_document_by_id(self):
        """Get a specific document by ID."""
        doc_id = input("🆔 Enter document ID: ").strip()
        if not doc_id:
            print("❌ Please enter a document ID.")
            return
        
        try:
            document = self.client_manager.search_client.get_document(key=doc_id)
            print(f"\n📄 Document Details: {doc_id}")
            print(f"📁 Filename: {document.get('filename', 'N/A')}")
            print(f"🏷️ Product: {document.get('product_name', 'N/A')}")
            print(f"📏 Content Length: {document.get('content_length', 'N/A')} characters")
            print(f"⏰ Processed: {document.get('processed_at', 'N/A')}")
            print(f"🔗 URL: {document.get('document_url', 'N/A')}")
            
            content = document.get('content', '')
            if content:
                preview = content[:300] + "..." if len(content) > 300 else content
                print(f"📝 Content Preview: {preview}")
            
        except Exception as e:
            logger.error(f"Failed to get document {doc_id}: {e}")
            print(f"❌ Document not found or error: {e}")
    
    def _show_index_stats(self):
        """Show index statistics."""
        print("📊 Index Statistics - Coming soon!")
    
    def _perform_direct_search(self, query: str):
        """Perform a direct keyword search from user input."""
        self.current_query = query
        print(f"\n🔍 Quick search for: '{query}'")
        
        try:
            results = self.client_manager.search_client.search(
                search_text=query,
                top=3,  # Fewer results for quick search
                highlight_fields="content",
                highlight_pre_tag="🔥",
                highlight_post_tag="🔥"
            )
            
            self._display_results(list(results), f"Quick Search", show_highlights=True)
            
        except Exception as e:
            logger.error(f"Direct search failed: {e}")
            print(f"❌ Search failed: {e}")

def start_interactive_search(search_endpoint: str, search_api_key: str, index_name: str):
    """Convenience function to start interactive search."""
    try:
        # Initialize components
        client_manager = AzureClientManager()
        embedding_generator = EmbeddingGenerator(client_manager)
        
        # Start interactive interface
        interface = InteractiveSearchInterface(client_manager, embedding_generator)
        interface.start_chat()
        
    except Exception as e:
        logger.error(f"Failed to start interactive search: {e}")
        print(f"❌ Failed to initialize search interface: {e}") 