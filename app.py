import os
from typing import List

import streamlit as st

from logseq_rag import (
    LLMConfig,
    LLMProvider,
    LogseqPage,
    LogseqParser,
    LogseqRAG,
    VectorStore,
)


def initialize_session_state():
    """Initialize session state variables"""
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "rag_system" not in st.session_state:
        st.session_state.rag_system = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "current_provider" not in st.session_state:
        st.session_state.current_provider = None


def load_vector_store(logseq_path):
    """Load or build vector store with page-based approach"""
    vector_store = VectorStore()

    with st.spinner("Parsing Logseq files..."):
        parser = LogseqParser(logseq_path)
        pages: List[LogseqPage] = parser.parse_all_files()

    st.info(f"Parsed {len(pages)} pages from {logseq_path}")

    if pages:
        with st.spinner("Creating embeddings... This may take a few minutes."):
            vector_store.add_pages(pages)
        st.success("Vector store created and saved!")
    else:
        st.error("No pages found. Please check your Logseq path.")
        return None

    return vector_store


def create_llm_config(provider_str, api_keys, model_settings):
    """Create LLM configuration from UI inputs"""
    provider = LLMProvider(provider_str)

    # Get API key for the selected provider
    api_key = None
    if provider == LLMProvider.OPENAI:
        api_key = api_keys.get("openai")
    elif provider == LLMProvider.ANTHROPIC:
        api_key = api_keys.get("anthropic")

    return LLMConfig(
        provider=provider,
        api_key=api_key,
        model_name=model_settings["model_name"],
        temperature=model_settings["temperature"],
        max_tokens=model_settings["max_tokens"],
    )


def main():
    st.set_page_config(page_title="Logseq RAG System", page_icon="🧠", layout="wide")

    initialize_session_state()

    st.title("🧠 Logseq RAG System")
    st.markdown(
        "Ask questions about your personal knowledge base using different LLM providers!"
    )

    # Sidebar for configuration
    with st.sidebar:
        st.header("🔧 Configuration")

        # LLM Provider Selection
        st.subheader("LLM Provider")
        provider = st.selectbox(
            "Choose LLM Provider",
            options=[p.value for p in LLMProvider],
            index=0,
            help="Select which LLM provider to use for generating responses",
        )

        # API Keys Section
        st.subheader("🔑 Model API Keys")
        api_keys = {}

        if provider == LLMProvider.OPENAI.value:
            api_keys["openai"] = st.text_input(
                "OpenAI API Key",
                type="password",
                value=os.getenv("OPENAI_API_KEY", ""),
                help="Your OpenAI API key",
            )
        elif provider == LLMProvider.ANTHROPIC.value:
            api_keys["anthropic"] = st.text_input(
                "Anthropic API Key",
                type="password",
                value=os.getenv("ANTHROPIC_API_KEY", ""),
                help="Your Anthropic API key",
            )

        # Model Settings
        st.subheader("⚙️ Model Settings")
        model_settings = {}

        model_settings["model_name"] = st.text_input(
            "Model Name",
            help="Specific model to use (e.g., gpt-4, claude-3-opus-20240229, llama2:13b)",
        )

        model_settings["temperature"] = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=2.0,
            value=0.0,
            step=0.1,
            help="Controls randomness in responses (0 = deterministic, 2 = very creative)",
        )

        model_settings["max_tokens"] = st.number_input(
            "Max Tokens",
            min_value=100,
            max_value=4000,
            value=1000,
            step=100,
            help="Maximum number of tokens in the response",
        )

        st.subheader("🔑 Logseq API Keys")
        logseq_config = {}

        logseq_config["url"] = st.text_input(
            "Logseq API URL",
            help="URL for your Logseq API (e.g., http://127.0.0.1:12315/api)",
        )

        logseq_config["token"] = st.text_input(
            "Logseq API Token",
            type="password",
            help="Your Logseq API token",
        )

        st.divider()

        # Retrieval Configuration
        st.subheader("🚀 Retrieval Config")

        st.markdown("**Search Limits**")
        vector_top_k = st.slider(
            "Vector Search Results",
            min_value=1,
            max_value=20,
            value=5,
            help="More pages provide more context but may include less relevant information",
        )

        title_max_k = st.slider(
            "LLM Maximum Number of Title Retrievals",
            min_value=1,
            max_value=10,
            value=5,
            help="More topics provide more context but may include less relevant information",
        )

        max_attempts = st.slider(
            "Max Retrieval Attempts",
            min_value=1,
            max_value=5,
            value=3,
            help="Number of attempts to retrieve relevant information if initial attempts fail",
        )

        st.divider()

        # Vector Store Configuration
        st.subheader("📚 Knowledge Base")

        # Logseq Path
        logseq_path = st.text_input(
            "Logseq Directory Path",
            placeholder="/path/to/your/logseq/directory",
            help="Path to your Logseq directory containing pages/ folders",
        )

        st.divider()

        # Initialise button
        if st.button("🚀 Initialise RAG System", type="primary"):
            # Validate inputs
            validation_errors = []

            if not logseq_path or not os.path.exists(logseq_path):
                validation_errors.append("Please provide a valid Logseq directory path")

            # Check API key requirements
            if provider == LLMProvider.OPENAI.value and not api_keys.get("openai"):
                validation_errors.append("OpenAI API key is required")
            elif provider == LLMProvider.ANTHROPIC.value and not api_keys.get(
                "anthropic"
            ):
                validation_errors.append("Anthropic API key is required")

            if validation_errors:
                for error in validation_errors:
                    st.error(error)
            else:
                try:
                    # Load vector store
                    vector_store = load_vector_store(
                        logseq_path,
                    )

                    if vector_store:
                        # Create LLM configuration
                        llm_config = create_llm_config(
                            provider,
                            api_keys,
                            model_settings,
                        )

                        # Initialize RAG system
                        rag_system = LogseqRAG(llm_config, vector_store)

                        # Store in session state
                        st.session_state.vector_store = vector_store
                        st.session_state.rag_system = rag_system
                        st.session_state.current_provider = (
                            rag_system.llm_manager.get_provider_info()
                        )

                        success_msg = "✅ RAG system initialized successfully!"
                        st.success(success_msg)
                        st.rerun()

                except Exception as e:
                    st.error(f"❌ Error initialising RAG system: {e}")

        st.divider()

        # System Status
        st.subheader("📊 System Status")

        if st.session_state.vector_store:
            if st.session_state.vector_store.pages:
                st.success(
                    f"✅ Vector store: {len(st.session_state.vector_store.pages):,} pages loaded"
                )

        else:
            st.warning("❌ Vector store not loaded")

        if st.session_state.current_provider:
            provider_info = st.session_state.current_provider
            st.success(
                f"✅ LLM: {provider_info['provider']} ({provider_info['model']})"
            )
            st.caption(f"Temperature: {provider_info['temperature']}")
        else:
            st.warning("❌ LLM not configured")

        # Quick actions
        if st.session_state.rag_system:
            st.subheader("⚡ Quick Actions")
            if st.button("🗑️ Clear Chat History"):
                st.session_state.chat_history = []
                st.rerun()

    # Main content area
    if st.session_state.rag_system:
        # Chat interface
        st.header("💬 Ask Questions")

        # Display provider info
        if st.session_state.current_provider:
            provider_info = st.session_state.current_provider
            st.info(
                f"🤖 Using **{provider_info['provider']}** with model **{provider_info['model']}**"
            )

        # Query input
        with st.form("query_form", clear_on_submit=True):
            query = st.text_area(
                "Your Question",
                placeholder="Ask anything about your knowledge base...",
                height=100,
                key="query_input",
            )

            col1, col2, col3 = st.columns(3)
            with col1:
                submit_button = st.form_submit_button("🔍 Ask", type="primary")
            with col2:
                preview_button = st.form_submit_button("👀 Preview Vector Search")
            with col3:
                st.form_submit_button("🔄 Clear Form")

        # Handle form submissions
        if submit_button and query.strip():
            with st.spinner("🔍 Searching knowledge base and generating answer..."):
                try:
                    result = st.session_state.rag_system.query(
                        query,
                        top_k=vector_top_k,
                        title_max_k=title_max_k,
                        max_attempts=max_attempts,
                        url=logseq_config.get("url"),
                        token=logseq_config.get("token"),
                    )

                    # Add to chat history
                    st.session_state.chat_history.append((query, result))
                    st.rerun()

                except Exception as e:
                    st.error(f"❌ Error processing query: {e}")

        elif preview_button and query.strip():
            # Preview search results without generating answer
            st.subheader("🔍 Vector Search Preview")

            results = st.session_state.vector_store.search(query, top_k=vector_top_k)

            if results:
                st.success(f"Found {len(results)} relevant pages:")
                for i, (item, score) in enumerate(results):
                    with st.expander(
                        f"Page {i + 1}: '{item.page_title}' (Relevance: {score:.3f})"
                    ):
                        st.markdown(f"**Page:** {item.page_title}")
                        st.markdown(
                            f"**Content:** {item.content[:1000]}{'...' if len(item.content) > 1000 else ''}"
                        )
                        if item.references:
                            st.markdown(f"**References:** {', '.join(item.references)}")
                        if hasattr(item, "word_count"):
                            st.markdown(f"**Word Count:** {item.word_count}")
                        if hasattr(item, "created_date") and item.created_date:
                            st.markdown(f"**Created:** {item.created_date}")
                        st.markdown(f"**File:** {item.file_path}")
            else:
                st.warning("No relevant pages found for your query.")

        # Display chat history
        for i, (question, result) in enumerate(
            st.session_state.chat_history[::-1]
        ):  # Reverse order for latest first
            with st.container():
                st.markdown(
                    f"**Q{len(st.session_state.chat_history) - i}:** {question}"
                )

                with st.expander(
                    f"Answer (via {result['provider_info']['provider']})", expanded=True
                ):
                    st.markdown(result["answer"])

                    # Show sources
                    if result["sources"]:
                        st.markdown("**📚 Sources:**")
                        for j, source in enumerate(result["sources"][:10]):
                            with st.expander(
                                f"Source {j + 1}: {source['page_title']} | Source Type - {source['source_type']} | Relevance: {source['relevance']:.3f}"
                            ):
                                st.markdown(f"**Content:** {source['content']}")
                                if source.get("references"):
                                    st.markdown(
                                        f"**References:** {', '.join(source['references'])}"
                                    )
                                if source.get("word_count"):
                                    st.markdown(
                                        f"**Word Count:** {source['word_count']}"
                                    )
                                if source.get("created_date"):
                                    st.markdown(
                                        f"**Created:** {source['created_date']}"
                                    )

                st.divider()

        # Knowledge base statistics
        if st.session_state.vector_store:
            with st.expander("📊 Knowledge Base Statistics"):
                if st.session_state.vector_store.pages:
                    pages = st.session_state.vector_store.pages

                    # Basic stats
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Pages", len(pages))
                    with col2:
                        total_words = sum(page.word_count for page in pages)
                        st.metric("Total Words", f"{total_words:,}")
                    with col3:
                        total_refs = sum(len(page.references) for page in pages)
                        st.metric("Total References", total_refs)

    else:
        # Welcome message
        st.markdown("""
        ## Welcome to Logseq RAG System with Multi-LLM Support! 🚀
        
        This enhanced system allows you to ask questions about your personal knowledge base using various LLM providers.
        
        ### 🤖 Supported LLM Providers
        - OpenAI: GPT-3.5, GPT-4, and other OpenAI models
        - Anthropic: Claude-3 Sonnet, Opus, and Haiku

        ### ✨ Enhanced Features
        - Page-based embeddings: Use complete pages instead of fine-grained blocks for better context
        - Multi-LLM support: Switch between different providers
        - Hybrid search modes: Combine content and title search for better results
        - Search preview: See what pages are retrieved before generating answers
        - ChromaDB support: Advanced vector database backen
        
        ### 🎯 Getting Started
        1. Choose your LLM provider in the sidebar
        2. Configure API keys and model settings
        3. Set your Logseq path and vector store options
        4. Configure database retrieval settings
        5. Initialize the system to build embeddings
        6. Start asking questions with advanced search filters

d
        
        """)


if __name__ == "__main__":
    main()
