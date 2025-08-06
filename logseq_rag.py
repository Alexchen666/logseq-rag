import logging
import pickle
import re

# Suppress warnings from sentence transformers
import warnings
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests
from langchain.schema import BaseMessage, HumanMessage, SystemMessage

# LangChain imports
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore", category=UserWarning, module="sentence_transformers")
# Suppress FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)


class LLMProvider(Enum):
    """Supported LLM providers"""

    ANTHROPIC = "anthropic"
    OPENAI = "openai"


@dataclass
class LLMConfig:
    """Configuration for LLM providers"""

    provider: LLMProvider
    api_key: Optional[str] = None
    model_name: Optional[str] = None
    base_url: Optional[str] = None  # For Ollama or custom endpoints
    temperature: float = 0.0
    max_tokens: int = 1000


@dataclass
class LogseqPage:
    """Represents a complete Logseq page with metadata"""

    content: str
    page_title: str
    file_path: str
    references: Optional[List[str]] = None
    word_count: int = 0
    created_date: Optional[str] = None

    def __post_init__(self):
        if self.references is None:
            self.references = self.extract_references()
        self.word_count = len(self.content.split())

    def extract_references(self) -> List[str]:
        """
        Extract [[references]] from content

        Returns:
            List[str]: List of references found in the content
        """
        pattern = r"\[\[([^\]]+)\]\]"
        return re.findall(pattern, self.content)

    def __hash__(self):
        """Make LogseqPage hashable based on page_title and file_path"""
        return hash((self.page_title, self.file_path))

    def __eq__(self, other):
        """Define equality based on page_title and file_path"""
        if not isinstance(other, LogseqPage):
            return False
        return self.page_title == other.page_title and self.file_path == other.file_path


class LogseqParser:
    """Parse Logseq markdown files into structured pages"""

    def __init__(self, logseq_path: str):
        self.logseq_path = Path(logseq_path)
        self.pages_path = self.logseq_path / "pages"
        self.journals_path = self.logseq_path / "journals"

    def parse_all_files(self) -> List[LogseqPage]:
        """
        Parse all markdown files in the Logseq directory as complete pages

        Returns:
            List[LogseqPage]: List of parsed LogseqPage objects"""
        pages = []

        # Parse pages
        if self.pages_path.exists():
            for md_file in self.pages_path.glob("*.md"):
                page = self.parse_file_as_page(md_file)
                if page:
                    pages.append(page)

        # Parse journals
        if self.journals_path.exists():
            for md_file in self.journals_path.glob("*.md"):
                page = self.parse_file_as_page(md_file)
                if page:
                    pages.append(page)

        return pages

    def parse_file_as_page(self, file_path: Path) -> Optional[LogseqPage]:
        """
        Parse a single markdown file as a complete page

        Args:
            file_path (Path): Path to the markdown file

        Returns:
            Optional[LogseqPage]: Parsed LogseqPage object or None if parsing fails
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return None

        page_title = file_path.stem

        # Extract creation date from journal files (format: YYYY_MM_DD)
        created_date = None
        if file_path.parent.name == "journals":
            try:
                date_str = file_path.stem.replace("_", "-")
                created_date = date_str
            except ValueError:
                pass

        return LogseqPage(
            content=content,
            page_title=page_title,
            file_path=str(file_path),
            created_date=created_date,
        )


class VectorStore:
    """Vector store for semantic search using sentence transformers"""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        use_chroma: bool = False,
    ):
        self.model = SentenceTransformer(model_name)
        self.pages: List[LogseqPage] = []
        self.embeddings: Optional[np.ndarray] = None
        self.use_chroma = use_chroma
        self.title_search_engine: Optional[TitleSearchEngine] = None

        if use_chroma:
            try:
                import chromadb

                self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
                collection_name = "logseq_pages"
                self.collection = self.chroma_client.get_or_create_collection(
                    collection_name
                )
            except ImportError:
                print("ChromaDB not installed. Install with: pip install chromadb")
                print("Falling back to NumPy approach...")
                self.use_chroma = False

    def add_pages(self, pages: List[LogseqPage]):
        """
        Add pages to the vector store

        Args:
            pages (List[LogseqPage]): List of LogseqPage objects to add
        """
        self.pages = pages
        print(f"Creating embeddings for {len(pages)} pages...")

        # Create embeddings for all pages
        texts = [f"{page.page_title}: {page.content}" for page in pages]
        embeddings = self.model.encode(texts, show_progress_bar=True)

        # Initialize title search engine
        if self.title_search_engine is None:
            print("Initializing title search engine...")
            self.title_search_engine = TitleSearchEngine(self.model)
            self.title_search_engine.build_index(pages)

        if self.use_chroma:
            # Store in ChromaDB
            documents = [page.content for page in pages]
            metadatas = [
                {
                    "page_title": page.page_title,
                    "file_path": page.file_path,
                    "references": ",".join(page.references) if page.references else "",
                    "word_count": page.word_count,
                    "created_date": page.created_date or "",
                }
                for page in pages
            ]
            ids = [str(i) for i in range(len(pages))]

            # Clear existing collection
            try:
                self.chroma_client.delete_collection("logseq_pages")
                self.collection = self.chroma_client.create_collection("logseq_pages")
            except Exception:
                pass

            # Add to ChromaDB in batches
            batch_size = 1000
            for i in range(0, len(documents), batch_size):
                batch_docs = documents[i : i + batch_size]
                batch_meta = metadatas[i : i + batch_size]
                batch_ids = ids[i : i + batch_size]
                batch_embeddings = embeddings[i : i + batch_size].tolist()

                self.collection.add(
                    documents=batch_docs,
                    metadatas=batch_meta,
                    ids=batch_ids,
                    embeddings=batch_embeddings,
                )
            print(f"Added {len(pages)} pages to ChromaDB")
        else:
            # Store as NumPy array
            self.embeddings = embeddings

    def search(self, query: str, top_k: int = 5) -> List[Tuple[LogseqPage, float]]:
        """
        Search for similar pages or blocks

        Args:
            query (str): Search query
            top_k (int): Maximum number of results to return

        Returns:
            List[Tuple[LogseqPage, float]]: List of (LogseqPage, similarity_score) tuples
        """
        if self.use_chroma:
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k,
            )

            # Convert ChromaDB results back to our format
            search_results = []
            for i, (doc, metadata, distance) in enumerate(
                zip(
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0],
                )
            ):
                # Find corresponding page or block
                item_idx = int(results["ids"][0][i])
                similarity = 1 - distance  # Convert distance to similarity

                search_results.append((self.pages[item_idx], similarity))

            return search_results
        else:
            # Use NumPy search
            if self.embeddings is None:
                return []

            query_embedding = self.model.encode([query])
            similarities = cosine_similarity(query_embedding, self.embeddings)[0]

            # Get top-k most similar items
            top_indices = np.argsort(similarities)[::-1][:top_k]

            results = []
            items = self.pages

            for idx in top_indices:
                results.append((items[idx], similarities[idx]))

            return results

    def search_titles(
        self, query: str, top_k: int = 5, min_similarity: float = 0.3
    ) -> List[Tuple[LogseqPage, float]]:
        """
        Search for pages by title similarity

        Args:
            query (str): Search query for titles
            top_k (int): Maximum number of results to return
            min_similarity (float): Minimum similarity threshold

        Returns:
            List[Tuple[LogseqPage, float]]: List of (page, similarity_score) tuples
        """
        if self.title_search_engine is None:
            print("Title search engine not initialized. Please add pages first.")
            return []

        return self.title_search_engine.search_titles(query, top_k, min_similarity)

    def search_exact_title(
        self, title: str, case_sensitive: bool = False
    ) -> Optional[LogseqPage]:
        """
        Find a page with exact title match

        Args:
            title: Exact title to search for
            case_sensitive: Whether to perform case-sensitive matching

        Returns:
            Optional[LogseqPage]: LogseqPage if found, None otherwise
        """
        if self.title_search_engine is None:
            print("Title search engine not initialized. Please add pages first.")
            return None

        return self.title_search_engine.search_exact_title(title, case_sensitive)

    def search_title_contains(
        self, substring: str, case_sensitive: bool = False
    ) -> List[LogseqPage]:
        """
        Find pages whose titles contain the given substring

        Args:
            substring (str): Substring to search for in titles
            case_sensitive (bool): Whether to perform case-sensitive matching

        Returns:
            List[LogseqPage]: List of LogseqPage objects with matching titles
        """
        if self.title_search_engine is None:
            print("Title search engine not initialized. Please add pages first.")
            return []

        return self.title_search_engine.search_title_contains(substring, case_sensitive)

    def save(self, path: str):
        """
        Save vector store to disk

        Args:
            path (str): Path to save the vector store
        """
        if not self.use_chroma:
            data = {"pages": self.pages, "embeddings": self.embeddings}
            with open(path, "wb") as f:
                pickle.dump(data, f)
            print(f"Vector store saved to {path}")
        else:
            print("ChromaDB data is automatically persisted")

    def load(self, path: str):
        """
        Load vector store from disk

        Args:
            path (str): Path to load the vector store from
        """
        if not self.use_chroma:
            with open(path, "rb") as f:
                data = pickle.load(f)
            self.pages = data["pages"]
            self.embeddings = data["embeddings"]
            # Initialize title search engine
            if self.title_search_engine is None:
                print("Initializing title search engine...")
                self.title_search_engine = TitleSearchEngine(self.model)
                self.title_search_engine.build_index(self.pages)
            print(f"Vector store loaded from {path}")
        else:
            print("ChromaDB data loaded from persistent storage")


class LLMManager:
    """Manages different LLM providers using LangChain"""

    def __init__(self, config: LLMConfig):
        self.config = config
        self.llm = self._initialize_llm()

    def _initialize_llm(self):
        """Initialize the appropriate LLM based on configuration"""
        if self.config.provider == LLMProvider.OPENAI:
            if not self.config.api_key:
                raise ValueError("OpenAI API key is required")

            return ChatOpenAI(
                api_key=self.config.api_key,
                model=self.config.model_name,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )

        elif self.config.provider == LLMProvider.ANTHROPIC:
            if not self.config.api_key:
                raise ValueError("Anthropic API key is required")

            return ChatAnthropic(
                anthropic_api_key=self.config.api_key,
                model=self.config.model_name,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )

        else:
            raise ValueError(f"Unsupported LLM provider: {self.config.provider}")

    def generate_response(self, messages: List[BaseMessage]) -> str:
        """
        Generate response using the configured LLM

        Args:
            messages (List[BaseMessage]): List of messages to send to the LLM

        Returns:
            str: Generated response content
        """
        try:
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            return f"Error generating response: {e}"

    def get_provider_info(self) -> Dict[str, str]:
        """Get information about the current provider"""
        return {
            "provider": self.config.provider.value,
            "model": self.config.model_name or "default",
            "temperature": str(self.config.temperature),
        }


class LogseqRAG:
    """Main RAG system class with multi-LLM support and title search"""

    def __init__(self, llm_config: LLMConfig, vector_store: VectorStore):
        self.llm_manager = LLMManager(llm_config)
        self.vector_store = vector_store

    # TODO: Ask LLM to define a topic title that is relevant to the question

    def query(
        self,
        question: str,
        title_query: str,
        top_k: int = 3,
        title_top_k: int = 5,
        url: Optional[str] = "",
        token: Optional[str] = "",
    ) -> Dict[str, Any]:
        """
        Enhanced query that combines content search, title-based page discovery, and Logseq API resource fetching

        Args:
            question (str): The main question to answer
            title_query (str): Query to find relevant pages by title
            top_k (int): Number of content-based results to retrieve
            title_top_k (int): Number of title-based results to retrieve
            url (Optional[str]): Logseq API URL for fetching resources
            token (Optional[str]): Authentication token for Logseq API

        Returns:
            Dict[str, Any]: Dictionary with answer and combined sources
        """
        # Get content-based results
        content_results = self.vector_store.search(question, top_k=top_k)

        # Get title-based results
        title_results = self.vector_store.search_titles(title_query, top_k=title_top_k)

        # Get resources from Logseq API if URL and token are provided
        journal_results = []
        page_results = []
        if url and token:
            top_title_results = title_results[0][0].page_title
            api_results = self.search_resources(f"[[{top_title_results}]]", url, token)
            journal_results = api_results.get("Journal", [])
            pages_queriable = api_results.get("Page", [])
            page_results = [
                self.vector_store.search_titles(page, top_k=1)
                for page in pages_queriable
            ]

            # Transform journal results into LogseqPage objects
            journal_results = [
                LogseqPage(
                    content=jr["Content"],
                    page_title=jr["Title"],
                    file_path="",
                    references=[],
                    created_date=None,
                )
                for jr in journal_results
            ]

        # Combine and deduplicate results
        all_results = {}

        # Add API results (This is the first priority)
        for page_result in page_results:
            for page, score in page_result:
                page_id = id(page)
                if page_id in all_results:
                    pass
                else:
                    all_results[page_id] = (page, 1.0, "API")

        for jr in journal_results:
            all_results[id(jr)] = (jr, 1.0, "API")

        # Add content results
        for page, score in content_results:
            page_id = id(page)
            if page_id in all_results:
                pass
            else:
                all_results[page_id] = (page, score, "content")

        # Add title results (with lower weight if already present)
        for page, score in title_results:
            page_id = id(page)
            if page_id in all_results:
                # Combine scores
                existing_score = all_results[page_id][1]
                combined_score = max(
                    existing_score, score * 0.7
                )  # Title results get 70% weight
                all_results[page_id] = (page, combined_score, "multiple")
            else:
                all_results[page_id] = (page, score * 0.7, "title")

        # Sort by combined score
        combined_results = [(page, score) for page, score, _ in all_results.values()]
        combined_results.sort(key=lambda x: x[1], reverse=True)

        if not combined_results:
            return {
                "answer": "No relevant information found in your knowledge base.",
                "sources": [],
                "title_query": title_query,
                "provider_info": self.llm_manager.get_provider_info(),
            }

        # Prepare context from combined results
        context = self._prepare_context(combined_results)

        # Generate response
        messages = self._create_messages(question, context)
        response = self.llm_manager.generate_response(messages)

        # Prepare sources with source type information
        sources = []
        for page, score in combined_results:
            page_id = id(page)
            source_type = next(
                source_type
                for p, s, source_type in all_results.values()
                if id(p) == page_id
            )

            sources.append(
                {
                    "page_title": page.page_title,
                    "content": page.content[:200] + "..."
                    if len(page.content) > 200
                    else page.content,
                    "similarity": score,
                    "references": page.references,
                    "source_type": source_type,  # "API", "content", "title", or "multiple"
                    "word_count": getattr(page, "word_count", 0),
                    "created_date": getattr(page, "created_date", None),
                }
            )

        return {
            "answer": response,
            "sources": sources,
            "title_query": title_query,
            "provider_info": self.llm_manager.get_provider_info(),
        }

    def search_titles(
        self, query: str, top_k: int = 5, min_similarity: float = 0.3
    ) -> Dict[str, Any]:
        """
        Search for pages by title similarity

        Args:
            query (str): Search query for titles
            top_k (int): Maximum number of results to return
            min_similarity (float): Minimum similarity threshold

        Returns:
            Dict[str, Any]: Dictionary with search results and metadata
        """
        results = self.vector_store.search_titles(query, top_k, min_similarity)

        if not results:
            return {
                "results": [],
                "query": query,
                "total_found": 0,
                "provider_info": self.llm_manager.get_provider_info(),
            }

        # Format results
        formatted_results = [
            {
                "page_title": page.page_title,
                "content": page.content[:300] + "..."
                if len(page.content) > 300
                else page.content,
                "similarity": score,
                "references": page.references,
                "word_count": page.word_count,
                "file_path": page.file_path,
                "created_date": page.created_date,
            }
            for page, score in results
        ]

        return {
            "results": formatted_results,
            "query": query,
            "total_found": len(results),
            "provider_info": self.llm_manager.get_provider_info(),
        }

    def search_resources(
        self,
        query: str,
        url: str,
        token: str,
    ) -> Dict[str, Any]:
        """
        Fetches resources from the Logseq API based on a query.
        It uses the Logseq API to query the knowledge base for resources that match the given query.
        If the results come from a journal page, it extracts the block tree and summarises the content.
        If the results are from a regular page, it returns the page name directly.

        Args:
            query (str): The query to search for resources.
            url (str): The URL of the Logseq API.
            token (str): The authentication token for the Logseq API.

        Returns:
            Dict[str, Any]: A dict of resources matching the query.
        """
        journal = []
        page_names = set()  # To avoid duplicates
        results = self._fetch_data(url, token, query)
        for result in results:
            if "journalDay" not in result["page"]:
                # If it is not from a journal page, return the page name directly
                page_names.add(result["page"]["name"])
            else:
                # If it is from a journal page, extract from the block tree
                tree = self._get_block_tree(url, token, result["page"]["name"])
                summary = self._get_summary_from_journal(tree, result["parent"]["id"])
                journal.append(summary)
        return {
            "Journal": journal,
            "Page": list(page_names),
        }

    def _fetch_data(self, url: str, token: str, query: str) -> List[Dict]:
        """
        Fetches data from the Logseq API based on the provided query.

        Args:
            url (str): The URL of the Logseq API.
            token (str): The authentication token for the Logseq API.
            query (str): The query to search for resources.
        Returns:
            List[Dict]: A list of results matching the query.
        """
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        data = {
            "method": "logseq.db.q",
            "args": [query],
        }
        response = requests.post(url, headers=headers, json=data)

        if response.status_code == 200:
            return response.json()
        else:
            response.raise_for_status()

    def _get_block_tree(self, url: str, token: str, page_name: str) -> List[Dict]:
        """
        Fetches the block tree for a given page from the Logseq API.

        Args:
            url (str): The URL of the Logseq API.
            token (str): The authentication token for the Logseq API.
            page_name (str): The name of the page to fetch the block tree for.

        Returns:
            List[Dict]: The block tree for the specified page.
        """
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        data = {
            "method": "logseq.Editor.getPageBlocksTree",
            "args": [page_name],
        }
        response = requests.post(url, headers=headers, json=data)

        if response.status_code == 200:
            return response.json()
        else:
            response.raise_for_status()

    def _get_summary_from_journal(self, block_tree: List[Dict], parent_id: int) -> Dict:
        """
        Extracts the summarised content from the block tree based on the given parent_id for a journal page.
        By parent_id, it can extract all the blocks that are children of the parent block, indicating they are the same topic.
        In this case, the parent block is the title of the journal page, and the children blocks are the content of the journal page.

        Args:
            block_tree (List[Dict]): The block tree.
            parent_id (int): The ID of the parent block to extract.

        Returns:
            Dict: A dictionary containing the title, source, and ordered content of the block.
        """
        selected_block = [
            block_item
            for block_item in block_tree
            if (block_item.get("id", -1) == parent_id)  # Find the parent block
            or (
                block_item.get("parent", {}).get("id", -1) == parent_id
            )  # Find the block belonging to the parent block
        ]
        # In most cases, there should be only one parent block
        if len(selected_block) != 1:
            raise ValueError(
                f"Expected one parent block with id {parent_id}, but found {len(selected_block)}."
            )
        else:
            title = selected_block[0].get("content", "")
            ordered = self._order_blocks(selected_block[0]["children"])

        return {
            "Title": title,
            "Source": ordered[0],
            "Content": ordered[1],
        }

    def _order_blocks(self, block_children: List[Dict]) -> List:
        """
        Orders the blocks based on their 'left' relationships.

        Args:
            block_children (List[Dict]): List of block children to order.

        Returns:
            ordered (List): Ordered list of block contents.
        """
        # Build a map from id to entry
        id_map = {item["id"]: item for item in block_children}

        # Start from the one that doesn't appear as 'left'
        left_ids = {item["left"]["id"] for item in block_children if "left" in item}
        all_ids = set(id_map.keys())

        start_ids = (
            all_ids - left_ids
        )  # Find the starting point, which is the one that doesn't appear as 'left' (it's the rightmost one)
        if start_ids:
            start_id = start_ids.pop()
        else:
            raise ValueError("No valid start node found")

        # Reconstruct the ordered list
        ordered = []
        current = id_map[start_id]
        while current:
            ordered.insert(
                0, current["content"]
            )  # Insert the 'left' content to the front
            next_id = current.get("left", {}).get("id")
            current = id_map.get(next_id)

        return ordered

    def _prepare_context(self, results: List[Tuple[LogseqPage, float]]) -> str:
        """
        Prepare context from search results

        Args:
            results (List[Tuple[LogseqPage, float]]): List of (LogseqPage, similarity_score) tuples

        Returns:
            str: Formatted context string for LLM input
        """
        context_parts = []

        for page, score in results:
            # Include page title and content
            context_part = f"From page '{page.page_title}':\n{page.content}"

            # Add references if any
            if page.references:
                refs = ", ".join(page.references)
                context_part += f"\n(References: {refs})"

            context_parts.append(context_part)

        return "\n\n---\n\n".join(context_parts)

    def _create_messages(self, question: str, context: str) -> List[BaseMessage]:
        """
        Create messages for the LLM

        Args:
            question (str): The user's question
            context (str): Context string from the knowledge base

        Returns:
            List[BaseMessage]: List of messages to send to the LLM
        """
        system_prompt = """You are an AI assistant helping to answer questions based on a personal knowledge base from Logseq.

Guidelines:
- Be specific and cite which pages the information comes from
- If you see [[references]] in the text, these are links to other pages in the knowledge base
- Provide direct quotes when relevant
- Be honest about the limitations of the available information
- If the context doesn't contain enough information to fully answer the question, say so and provide what information you can"""

        user_prompt = f"""Context from the knowledge base:
{context}

Question: {question}

Please provide a comprehensive answer based on the context above."""

        return [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]


class TitleSearchEngine:
    """
    Simple and effective title-based search engine using vector similarity
    """

    def __init__(self, embedding_model: SentenceTransformer):
        """
        Initialize the title search engine

        Args:
            embedding_model (SentenceTransformer): Pre-trained sentence transformer model for vectorizing titles
        """
        self.embedding_model = embedding_model
        self.pages: List[LogseqPage] = []
        self.title_embeddings: Optional[np.ndarray] = None

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def build_index(self, pages: List[LogseqPage]) -> None:
        """
        Build title embeddings for the given pages

        Args:
            pages (List[LogseqPage]): List of LogseqPage objects to index
        """
        self.pages = pages

        if not pages:
            self.logger.warning("No pages provided for indexing")
            return

        self.logger.info(f"Building title embeddings for {len(pages)} pages...")

        # Extract titles and create embeddings
        titles = [page.page_title for page in pages]
        self.title_embeddings = self.embedding_model.encode(
            titles, show_progress_bar=True
        )

        self.logger.info(f"Title index built successfully with {len(titles)} titles")

    def search_titles(
        self, query: str, top_k: int = 10, min_similarity: float = 0.3
    ) -> List[Tuple[LogseqPage, float]]:
        """
        Search for pages with titles similar to the query

        Args:
            query (str): Search query (can be keywords or phrases)
            top_k (int): Maximum number of results to return
            min_similarity (float): Minimum similarity threshold (0.0 to 1.0)

        Returns:
            List[Tuple[LogseqPage, float]]: List of (page, similarity_score) tuples sorted by similarity
        """
        if not query.strip():
            self.logger.warning("Empty query provided")
            return []

        if self.title_embeddings is None or not self.pages:
            self.logger.warning("Title index not built. Call build_index() first.")
            return []

        # Create query embedding
        query_embedding = self.embedding_model.encode([query])

        # Calculate similarities with all titles
        similarities = cosine_similarity(query_embedding, self.title_embeddings)[0]

        # Get indices sorted by similarity (descending)
        sorted_indices = np.argsort(similarities)[::-1]

        # Filter results by minimum similarity and top_k
        results = []
        for idx in sorted_indices:
            similarity = float(similarities[idx])
            if similarity >= min_similarity and len(results) < top_k:
                results.append((self.pages[idx], similarity))
            elif similarity < min_similarity:
                break  # Since sorted, no more results will meet threshold

        self.logger.debug(
            f"Title search for '{query}': found {len(results)} results above threshold {min_similarity}"
        )
        return results

    def search_exact_title(
        self, title: str, case_sensitive: bool = False
    ) -> Optional[LogseqPage]:
        """
        Find a page with exact title match

        Args:
            title (str): Exact title to search for
            case_sensitive (bool): Whether to perform case-sensitive matching

        Returns:
            Optional[LogseqPage]: LogseqPage if found, None otherwise
        """
        if not title.strip():
            return None

        search_title = title if case_sensitive else title.lower()

        for page in self.pages:
            page_title = page.page_title if case_sensitive else page.page_title.lower()
            if page_title == search_title:
                self.logger.debug(f"Exact title match found for '{title}'")
                return page

        self.logger.debug(f"No exact title match found for '{title}'")
        return None

    def search_title_contains(
        self, substring: str, case_sensitive: bool = False
    ) -> List[LogseqPage]:
        """
        Find pages whose titles contain the given substring

        Args:
            substring (str): Substring to search for in titles
            case_sensitive (bool): Whether to perform case-sensitive matching

        Returns:
            List[LogseqPage]: List of LogseqPage objects with matching titles
        """
        if not substring.strip():
            return []

        search_substring = substring if case_sensitive else substring.lower()
        results = []

        for page in self.pages:
            page_title = page.page_title if case_sensitive else page.page_title.lower()
            if search_substring in page_title:
                results.append(page)

        self.logger.debug(
            f"Title contains search for '{substring}': found {len(results)} results"
        )
        return results
