import argparse
import os
import pickle
import re

# Suppress warnings from sentence transformers
import warnings
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from dotenv import load_dotenv
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

    OPENAI = "openai"
    ANTHROPIC = "anthropic"


@dataclass
class LLMConfig:
    """Configuration for LLM providers"""

    provider: LLMProvider
    api_key: Optional[str] = None
    model_name: Optional[str] = None
    base_url: Optional[str] = None  # For Ollama or custom endpoints
    temperature: float = 0.1
    max_tokens: int = 1000


@dataclass
class LogseqBlock:
    """Represents a Logseq block with metadata"""

    content: str
    page_title: str
    file_path: str
    block_id: Optional[str] = None
    references: Optional[List[str]] = None

    def __post_init__(self):
        if self.references is None:
            self.references = self.extract_references()

    def extract_references(self) -> List[str]:
        """Extract [[references]] from content"""
        pattern = r"\[\[([^\]]+)\]\]"
        return re.findall(pattern, self.content)


class LogseqParser:
    """Parse Logseq markdown files into structured blocks"""

    def __init__(self, logseq_path: str):
        self.logseq_path = Path(logseq_path)
        self.pages_path = self.logseq_path / "pages"
        self.journals_path = self.logseq_path / "journals"

    def parse_all_files(self) -> List[LogseqBlock]:
        """Parse all markdown files in the Logseq directory"""
        blocks = []

        # Parse pages
        if self.pages_path.exists():
            for md_file in self.pages_path.glob("*.md"):
                blocks.extend(self.parse_file(md_file))

        # Parse journals
        if self.journals_path.exists():
            for md_file in self.journals_path.glob("*.md"):
                blocks.extend(self.parse_file(md_file))

        return blocks

    def parse_file(self, file_path: Path) -> List[LogseqBlock]:
        """Parse a single markdown file into blocks"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return []

        page_title = file_path.stem
        blocks = []

        # Split by lines and process blocks
        lines = content.split("\n")
        current_block = []

        for line in lines:
            line = line.strip()

            # Skip empty lines and metadata
            if not line or line.startswith("---") or line.startswith("title:"):
                if current_block:
                    block_content = "\n".join(current_block).strip()
                    if block_content:
                        blocks.append(
                            LogseqBlock(
                                content=block_content,
                                page_title=page_title,
                                file_path=str(file_path),
                            )
                        )
                    current_block = []
                continue

            # Handle bullet points and blocks
            if line.startswith("-") or line.startswith("*"):
                if current_block:
                    block_content = "\n".join(current_block).strip()
                    if block_content:
                        blocks.append(
                            LogseqBlock(
                                content=block_content,
                                page_title=page_title,
                                file_path=str(file_path),
                            )
                        )
                current_block = [line[1:].strip()]  # Remove bullet point
            else:
                current_block.append(line)

        # Handle last block
        if current_block:
            block_content = "\n".join(current_block).strip()
            if block_content:
                blocks.append(
                    LogseqBlock(
                        content=block_content,
                        page_title=page_title,
                        file_path=str(file_path),
                    )
                )

        return blocks


class VectorStore:
    """Vector store for semantic search using sentence transformers"""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", use_chroma: bool = False):
        self.model = SentenceTransformer(model_name)
        self.blocks: List[LogseqBlock] = []
        self.embeddings: Optional[np.ndarray] = None
        self.use_chroma = use_chroma

        if use_chroma:
            try:
                import chromadb

                self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
                self.collection = self.chroma_client.get_or_create_collection("logseq")
            except ImportError:
                print("ChromaDB not installed. Install with: pip install chromadb")
                print("Falling back to NumPy approach...")
                self.use_chroma = False

    def add_blocks(self, blocks: List[LogseqBlock]):
        """Add blocks to the vector store"""
        self.blocks = blocks
        print("Creating embeddings...")

        # Create embeddings for all blocks
        texts = [f"{block.page_title}: {block.content}" for block in blocks]
        embeddings = self.model.encode(texts, show_progress_bar=True)

        if self.use_chroma:
            # Store in ChromaDB
            documents = [block.content for block in blocks]
            metadatas = [
                {
                    "page_title": block.page_title,
                    "file_path": block.file_path,
                    "references": ",".join(block.references)
                    if block.references
                    else "",
                }
                for block in blocks
            ]
            ids = [str(i) for i in range(len(blocks))]

            # Clear existing collection
            try:
                self.chroma_client.delete_collection("logseq")
                self.collection = self.chroma_client.create_collection("logseq")
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
            print(f"Added {len(blocks)} blocks to ChromaDB")
        else:
            # Store as NumPy array
            self.embeddings = embeddings

    def search(
        self, query: str, top_k: int = 5, **filters
    ) -> List[Tuple[LogseqBlock, float]]:
        """Search for similar blocks"""
        if self.use_chroma:
            # Use ChromaDB search
            where_clause = {}
            if "page_title" in filters:
                where_clause["page_title"] = {"$eq": filters["page_title"]}
            if "references" in filters:
                where_clause["references"] = {"$contains": filters["references"]}

            results = self.collection.query(
                query_texts=[query],
                n_results=top_k,
                where=where_clause if where_clause else None,
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
                # Find corresponding block
                block_idx = int(results["ids"][0][i])
                similarity = 1 - distance  # Convert distance to similarity
                search_results.append((self.blocks[block_idx], similarity))

            return search_results
        else:
            # Use NumPy search
            if self.embeddings is None:
                return []

            query_embedding = self.model.encode([query])
            similarities = cosine_similarity(query_embedding, self.embeddings)[0]

            # Get top-k most similar blocks
            top_indices = np.argsort(similarities)[::-1][:top_k]

            results = []
            for idx in top_indices:
                # Apply filters manually for NumPy approach
                block = self.blocks[idx]
                if (
                    "page_title" in filters
                    and block.page_title != filters["page_title"]
                ):
                    continue
                if (
                    "references" in filters
                    and filters["references"] not in block.references
                ):
                    continue

                results.append((block, similarities[idx]))
                if len(results) >= top_k:
                    break

            return results

    def save(self, path: str):
        """Save vector store to disk"""
        if not self.use_chroma:
            data = {"blocks": self.blocks, "embeddings": self.embeddings}
            with open(path, "wb") as f:
                pickle.dump(data, f)
            print(f"Vector store saved to {path}")
        else:
            print("ChromaDB data is automatically persisted")

    def load(self, path: str):
        """Load vector store from disk"""
        if not self.use_chroma:
            with open(path, "rb") as f:
                data = pickle.load(f)
            self.blocks = data["blocks"]
            self.embeddings = data["embeddings"]
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
        """Generate response using the configured LLM"""
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
    """Main RAG system class with multi-LLM support"""

    def __init__(self, llm_config: LLMConfig, vector_store: VectorStore):
        self.llm_manager = LLMManager(llm_config)
        self.vector_store = vector_store

    def query(self, question: str, top_k: int = 5, **search_filters) -> Dict[str, Any]:
        """Query the RAG system"""
        # Retrieve relevant blocks
        results = self.vector_store.search(question, top_k=top_k, **search_filters)

        if not results:
            return {
                "answer": "No relevant information found in your knowledge base.",
                "sources": [],
                "provider_info": self.llm_manager.get_provider_info(),
            }

        # Prepare context from retrieved blocks
        context = self._prepare_context(results)

        # Generate response using configured LLM
        messages = self._create_messages(question, context)
        response = self.llm_manager.generate_response(messages)

        # Prepare sources
        sources = [
            {
                "page_title": block.page_title,
                "content": block.content[:200] + "..."
                if len(block.content) > 200
                else block.content,
                "similarity": score,
                "references": block.references,
            }
            for block, score in results
        ]

        return {
            "answer": response,
            "sources": sources,
            "provider_info": self.llm_manager.get_provider_info(),
        }

    def _prepare_context(self, results: List[Tuple[LogseqBlock, float]]) -> str:
        """Prepare context from search results"""
        context_parts = []

        for block, score in results:
            # Include page title and content
            context_part = f"From page '{block.page_title}':\n{block.content}"

            # Add references if any
            if block.references:
                refs = ", ".join(block.references)
                context_part += f"\n(References: {refs})"

            context_parts.append(context_part)

        return "\n\n---\n\n".join(context_parts)

    def _create_messages(self, question: str, context: str) -> List[BaseMessage]:
        """Create messages for the LLM"""
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


def create_llm_config_from_args(args) -> LLMConfig:
    """Create LLM configuration from command line arguments"""
    provider = LLMProvider(args.llm_provider)

    # Get API key from args or environment
    api_key = None
    if provider == LLMProvider.OPENAI:
        api_key = args.openai_key or os.getenv("OPENAI_API_KEY")
    elif provider == LLMProvider.ANTHROPIC:
        api_key = args.anthropic_key or os.getenv("ANTHROPIC_API_KEY")

    return LLMConfig(
        provider=provider,
        api_key=api_key,
        model_name=args.model_name,
        base_url=args.base_url,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Logseq RAG System with Multiple LLM Support"
    )

    # Core arguments
    parser.add_argument("--logseq-path", required=True, help="Path to Logseq directory")
    parser.add_argument(
        "--vector-store", default="vector_store.pkl", help="Path to vector store file"
    )
    parser.add_argument(
        "--rebuild", action="store_true", help="Rebuild vector store from scratch"
    )
    parser.add_argument(
        "--use-chroma", action="store_true", help="Use ChromaDB instead of NumPy"
    )
    parser.add_argument("--query", help="Query to ask")

    # LLM configuration
    parser.add_argument(
        "--llm-provider",
        choices=[p.value for p in LLMProvider],
        default="anthropic",
        help="LLM provider to use",
    )
    parser.add_argument("--model-name", help="Specific model name to use")
    parser.add_argument(
        "--temperature", type=float, default=0.1, help="Temperature for LLM"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=1000, help="Max tokens for LLM response"
    )

    # API keys
    parser.add_argument("--openai-key", help="OpenAI API key")
    parser.add_argument("--anthropic-key", help="Anthropic API key")
    parser.add_argument("--azure-key", help="Azure OpenAI API key")

    # Provider-specific options
    parser.add_argument("--base-url", help="Base URL for Ollama or custom endpoints")
    parser.add_argument("--azure-endpoint", help="Azure OpenAI endpoint")
    parser.add_argument("--azure-deployment", help="Azure OpenAI deployment name")
    parser.add_argument("--api-version", help="Azure OpenAI API version")

    args = parser.parse_args()

    # Create LLM configuration
    try:
        llm_config = create_llm_config_from_args(args)
    except ValueError as e:
        print(f"Configuration error: {e}")
        return

    # Initialize vector store
    vector_store = VectorStore(use_chroma=args.use_chroma)

    # Load or build vector store
    if not args.rebuild and os.path.exists(args.vector_store) and not args.use_chroma:
        print("Loading existing vector store...")
        vector_store.load(args.vector_store)
    else:
        print("Building vector store...")
        parser = LogseqParser(args.logseq_path)
        blocks = parser.parse_all_files()
        print(f"Parsed {len(blocks)} blocks from {args.logseq_path}")

        if blocks:
            vector_store.add_blocks(blocks)
            vector_store.save(args.vector_store)
        else:
            print("No blocks found. Please check your Logseq path.")
            return

    # Initialize RAG system
    try:
        rag = LogseqRAG(llm_config, vector_store)
        provider_info = rag.llm_manager.get_provider_info()
        print(f"Using {provider_info['provider']} with model {provider_info['model']}")
    except Exception as e:
        print(f"Error initializing LLM: {e}")
        return

    if args.query:
        # Single query mode
        print(f"\nQuery: {args.query}")
        print("=" * 50)
        result = rag.query(args.query)
        print(f"Answer ({result['provider_info']['provider']}):")
        print(result["answer"])

        print("\nSources:")
        for i, source in enumerate(result["sources"][:3]):
            print(
                f"{i + 1}. {source['page_title']} (similarity: {source['similarity']:.3f})"
            )
    else:
        # Interactive mode
        print("\nLogseq RAG System - Interactive Mode")
        print(
            f"Provider: {provider_info['provider']} | Model: {provider_info['model']}"
        )
        print("Type 'quit' to exit")
        print("=" * 50)

        while True:
            query = input("\nYour question: ").strip()
            if query.lower() in ["quit", "exit", "q"]:
                break

            if query:
                print("\nAnswer:")
                print("-" * 30)
                result = rag.query(query)
                print(result["answer"])

                print("\nSources (top 3):")
                for i, source in enumerate(result["sources"][:3]):
                    print(
                        f"{i + 1}. {source['page_title']} (similarity: {source['similarity']:.3f})"
                    )


if __name__ == "__main__":
    main()
