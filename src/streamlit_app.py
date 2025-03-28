import os
import streamlit as st
import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any, List, Optional
import io
import torch
from enum import Enum

# Import our modules
from cag_generator import CAGGenerator, ModelBackend
from ollama_client import OllamaGenerationOptions
from document_processor import DocumentProcessor

# Optional imports for HuggingFace
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

# Set page configuration
st.set_page_config(
    page_title="Cache Augmented Generation",
    page_icon="⚡",
    layout="wide",
)

# Initialize session state for backend selection
if "backend" not in st.session_state:
    st.session_state.backend = "ollama"  # Default to Ollama

if "hf_model_loaded" not in st.session_state:
    st.session_state.hf_model_loaded = False

if "hf_model" not in st.session_state:
    st.session_state.hf_model = None

if "hf_tokenizer" not in st.session_state:
    st.session_state.hf_tokenizer = None

# Initialize the generator based on the selected backend
def initialize_generator():
    if st.session_state.backend == "huggingface":
        if not st.session_state.hf_model_loaded:
            st.warning("No HuggingFace model loaded. Please load a model in the settings.")
            return None
        
        return CAGGenerator(
            backend=ModelBackend.HUGGINGFACE,
            hf_model=st.session_state.hf_model,
            hf_tokenizer=st.session_state.hf_tokenizer,
            cache_dir=os.path.abspath("../cache"),
        )
    else:  # ollama
        return CAGGenerator(
            backend=ModelBackend.OLLAMA,
            ollama_url="http://localhost:11434",
            cache_dir=os.path.abspath("../cache"),
        )

# Initialize other components
if "generator" not in st.session_state:
    st.session_state.generator = initialize_generator()

if "doc_processor" not in st.session_state:
    st.session_state.doc_processor = DocumentProcessor(
        cache_dir=os.path.abspath("../cache/docs")
    )

if "generation_history" not in st.session_state:
    st.session_state.generation_history = []

if "benchmark_results" not in st.session_state:
    st.session_state.benchmark_results = None

if "selected_documents" not in st.session_state:
    st.session_state.selected_documents = []

if "knowledge_text" not in st.session_state:
    st.session_state.knowledge_text = None

if "preloaded_knowledge_id" not in st.session_state:
    st.session_state.preloaded_knowledge_id = None

# Load HuggingFace model function
def load_huggingface_model(model_name, use_4bit=True):
    """Load a HuggingFace model."""
    if not HF_AVAILABLE:
        st.error("HuggingFace transformers library not available. Please install it with pip install transformers.")
        return None, None
    
    try:
        with st.spinner(f"Loading {model_name}... This might take a while."):
            # Configure quantization if requested
            if use_4bit:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
                )
            else:
                quantization_config = None
            
            # Load tokenizer and model
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto"
            )
            
            return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

# App title and description
st.title("⚡ Cache Augmented Generation (CAG)")
st.markdown(
    """
    This application demonstrates Cache-Augmented Generation using local LLMs.
    It supports both:
    - **Ollama models** running locally via API
    - **HuggingFace models** loaded directly in Python (requires GPU)
    
    CAG preloads knowledge into the model's KV cache, resulting in significant speedups for queries about that knowledge.
    """
)

# Sidebar for configurations
with st.sidebar:
    st.header("Backend Settings")
    # Backend selection
    backend_options = ["ollama"]
    if HF_AVAILABLE:
        backend_options.append("huggingface")
    
    selected_backend = st.selectbox(
        "Backend", 
        options=backend_options,
        index=0 if st.session_state.backend == "ollama" else 1
    )
    
    # Update backend if changed
    if selected_backend != st.session_state.backend:
        st.session_state.backend = selected_backend
        st.session_state.generator = initialize_generator()
        st.experimental_rerun()
    
    # HuggingFace-specific settings
    if selected_backend == "huggingface":
        st.subheader("HuggingFace Settings")
        
        hf_model_name = st.text_input(
            "Model Name", 
            value="meta-llama/Meta-Llama-3.1-8B-Instruct" if not st.session_state.hf_model_loaded else "",
            placeholder="E.g., meta-llama/Meta-Llama-3.1-8B-Instruct"
        )
        
        use_4bit = st.checkbox("Load in 4-bit", value=True)
        
        if st.button("Load Model") and hf_model_name:
            model, tokenizer = load_huggingface_model(hf_model_name, use_4bit)
            
            if model is not None and tokenizer is not None:
                st.session_state.hf_model = model
                st.session_state.hf_tokenizer = tokenizer
                st.session_state.hf_model_loaded = True
                st.session_state.generator = initialize_generator()
                st.success(f"Model {hf_model_name} loaded successfully!")
            else:
                st.error("Failed to load model.")
    
    # Ollama-specific settings
    if selected_backend == "ollama":
        st.subheader("Ollama Settings")
        ollama_url = st.text_input("Ollama URL", value="http://localhost:11434")
        
        if ollama_url != "http://localhost:11434" and st.button("Update Ollama URL"):
            st.session_state.generator = CAGGenerator(
                backend=ModelBackend.OLLAMA,
                ollama_url=ollama_url,
                cache_dir=os.path.abspath("../cache"),
            )
            st.success(f"Ollama URL updated to {ollama_url}")
        # Cache settings
    st.divider()
    st.header("Cache Settings")
    use_cache = st.checkbox("Use KV Cache", value=True)
    save_cache = st.checkbox("Save KV Cache", value=True)
    
    if st.button("Clear Cache"):
        if selected_backend == "ollama":
            st.session_state.generator.kv_cache_manager.clear_cache()
        st.success("Cache cleared successfully!")
    # Model selection (for both backends)
    st.divider()
    st.header("Model Settings")
    
    # Get available models
    try:
        if st.session_state.generator:
            available_models = st.session_state.generator.list_available_models()
            if not available_models:
                if selected_backend == "ollama":
                    available_models = ["mistral", "deepseek-r1"]  # Fallback options for Ollama
                else:
                    available_models = ["No models available"]
        else:
            available_models = ["No backend initialized"]
    except Exception as e:
        st.error(f"Error connecting to backend: {str(e)}")
        if selected_backend == "ollama":
            st.error("Make sure Ollama is running on the specified URL")
            available_models = ["mistral", "deepseek-r1"]  # Fallback options
        else:
            available_models = ["No models available"]
    
    selected_model = st.selectbox("Model", available_models)
    
    st.divider()
    st.header("Generation Parameters")
    
    # Generation parameters
    temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.1)
    top_p = st.slider("Top P", 0.0, 1.0, 0.9, 0.01)
    max_tokens = st.slider("Max Tokens", 32, 4096, 1024, 32)
    


# Main content area - tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Text Generation", "Knowledge Base", "CAG Preloading", "Benchmark", "About CAG"
])

# Tab 1: Text Generation
with tab1:
    # Add context from documents or preloaded knowledge
    use_knowledge_base = st.checkbox("Use Knowledge Base", value=False)
    use_preloaded_knowledge = st.checkbox("Use Preloaded Knowledge", value=False)
    # Text input area
    prompt = st.text_area("Enter your prompt", height=150)
    
    # Cannot use both
    if use_knowledge_base and use_preloaded_knowledge:
        st.warning("Please select only one knowledge source (either Knowledge Base or Preloaded Knowledge).")
    
    knowledge_text = None
    
    if use_knowledge_base and not use_preloaded_knowledge:
        if not st.session_state.selected_documents:
            st.warning("No documents selected. Please add documents in the Knowledge Base tab.")
        else:
            # Search box for knowledge base
            kb_query = st.text_input("Search in Knowledge Base (optional)")
            
            if kb_query:
                # Search in documents
                search_results = st.session_state.doc_processor.search_documents(kb_query)
                
                if search_results:
                    st.subheader("Relevant Context")
                    for chunk_id, chunk_text, score in search_results:
                        with st.expander(f"Document Chunk (Relevance: {score:.2f})"):
                            st.text(chunk_text)
                            if st.button(f"Use This Context", key=f"use_{chunk_id}"):
                                knowledge_text = chunk_text
                                st.success("Context selected!")
                else:
                    st.info("No relevant information found in knowledge base.")
    
    elif use_preloaded_knowledge and not use_knowledge_base:
        if st.session_state.knowledge_text is None:
            st.warning("No knowledge has been preloaded. Please preload knowledge in the CAG Preloading tab.")
        else:
            knowledge_text = st.session_state.knowledge_text
            st.success(f"Using preloaded knowledge ({len(knowledge_text)} characters)")
    
    # Create two columns for buttons
    col1, col2 = st.columns(2)
    
    with col1:
        generate_clicked = st.button("Generate Text", type="primary")
    
    with col2:
        clear_clicked = st.button("Clear History")
    
    # Handle generation
    if generate_clicked and prompt and st.session_state.generator:
        with st.spinner("Generating..."):
            # Generate with CAG
            result = st.session_state.generator.generate(
                prompt=prompt,
                model_name=selected_model,
                use_cache=use_cache,
                save_cache=save_cache,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                knowledge_text=knowledge_text
            )
            
            # Add to history
            st.session_state.generation_history.append(result)
    
    # Handle clearing history
    if clear_clicked:
        st.session_state.generation_history = []
    
    # Display generation history
    if st.session_state.generation_history:
        st.divider()
        st.subheader("Generation History")
        
        for i, result in enumerate(reversed(st.session_state.generation_history)):
            with st.expander(f"Generation {len(st.session_state.generation_history) - i}", expanded=(i == 0)):
                # Layout with columns
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown("**Generated Text:**")
                    st.markdown(result["text"])
                    
                    st.markdown("**Prompt:**")
                    st.text(result["prompt"])
                
                with col2:
                    st.markdown("**Generation Stats:**")
                    
                    cache_status = "No cache used"
                    if result.get("cache_hit", False):
                        cache_status = "Full cache hit ✅"
                    elif result.get("cache_prefix_hit", False):
                        prefix_length = result.get("prefix_length", 0)
                        prefix_ratio = prefix_length / len(result["prompt"]) if len(result["prompt"]) > 0 else 0
                        cache_status = f"Prefix cache hit ({prefix_ratio:.1%}) ✅"
                    
                    knowledge_status = "No knowledge used"
                    if result.get("knowledge_augmented", False):
                        knowledge_status = "Knowledge augmented ✅"
                    
                    st.markdown(f"- **Cache Status:** {cache_status}")
                    st.markdown(f"- **Knowledge:** {knowledge_status}")
                    st.markdown(f"- **Tokens Generated:** {result.get('tokens_generated', 0)}")
                    st.markdown(f"- **Generation Time:** {result.get('generation_time', 0):.2f}s")
                    st.markdown(f"- **Speed:** {result.get('tokens_per_second', 0):.2f} tokens/sec")
                    st.markdown(f"- **Model:** {result.get('model', 'unknown')}")
    else:
        st.info("Enter a prompt and click 'Generate Text' to start!")

# Tab 2: Knowledge Base
with tab2:
    st.header("Knowledge Base")
    st.markdown(
        """
        Upload documents to create a knowledge base for context-aware text generation.
        These documents will be processed, chunked, and made available for search.
        """
    )
    
    # File upload section
    st.subheader("Upload Documents")
    uploaded_file = st.file_uploader("Choose a text file", type=["txt", "md", "csv", "json", "pdf", "doc", "docx"], accept_multiple_files=False)
    
    col1, col2 = st.columns(2)
    
    with col1:
        chunk_size = st.number_input("Chunk Size (characters)", min_value=200, max_value=4000, value=1000)
    
    with col2:
        chunk_overlap = st.number_input("Chunk Overlap (characters)", min_value=0, max_value=chunk_size-1, value=200)
    
    if uploaded_file and st.button("Process Document"):
        with st.spinner("Processing document..."):
            # Read the file
            try:
                content = uploaded_file.read().decode("utf-8")
                
                # Process the document
                doc_id = st.session_state.doc_processor.process_document(
                    content=content, 
                    filename=uploaded_file.name,
                    chunk_size=chunk_size,
                    overlap=chunk_overlap
                )
                
                st.success(f"Document processed successfully! Document ID: {doc_id}")
                st.session_state.selected_documents = st.session_state.doc_processor.get_all_documents()
            except Exception as e:
                st.error(f"Error processing document: {str(e)}")
    
    # Display existing documents
    st.subheader("Existing Documents")
    
    existing_docs = st.session_state.doc_processor.get_all_documents()
    st.session_state.selected_documents = existing_docs
    
    if existing_docs:
        for doc in existing_docs:
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.write(f"**{doc['filename']}** ({doc['chunks']} chunks)")
            
            with col2:
                if st.button("Remove", key=f"remove_{doc['id']}"):
                    st.session_state.doc_processor.remove_document(doc['id'])
                    st.rerun()
                    
                if st.button("View Chunks", key=f"view_{doc['id']}"):
                    chunks = st.session_state.doc_processor.get_document_chunks(doc['id'])
                    for i, chunk in enumerate(chunks):
                        with st.expander(f"Chunk {i+1}"):
                            st.text(chunk)
    else:
        st.info("No documents in the knowledge base. Upload a document to get started.")

# Tab 3: CAG Preloading
with tab3:
    st.header("CAG Knowledge Preloading")
    st.markdown(
        """
        Preload knowledge directly into the model's KV cache for faster inference.
        This is the core of Cache-Augmented Generation (CAG).
        
        Enter or paste knowledge text below, then preload it into the model's KV cache.
        This knowledge will be efficiently reused for all subsequent queries.
        """
    )
    # Information about loading from files
    st.subheader("Load Knowledge from Knowledge Base")
    st.markdown(
        """
        You can also load knowledge from your knowledge base.
        """
    )
    
    if not st.session_state.selected_documents:
        st.warning("No documents in knowledge base. Please add documents in the Knowledge Base tab.")
    else:
        # Create a dropdown of available documents
        doc_options = [f"{doc['filename']} ({doc['chunks']} chunks)" for doc in st.session_state.selected_documents]
        selected_doc_idx = st.selectbox("Select Document", range(len(doc_options)), format_func=lambda x: doc_options[x])
        
        if st.button("Load Document as Knowledge"):
            selected_doc = st.session_state.selected_documents[selected_doc_idx]
            doc_chunks = st.session_state.doc_processor.get_document_chunks(selected_doc["id"])
            
            # Combine all chunks into one text
            combined_text = "\n\n".join(doc_chunks)
            
            # Update the knowledge text area
            st.session_state.knowledge_text = combined_text
            st.success(f"Loaded document '{selected_doc['filename']}' as knowledge text")
            st.rerun()
    # Knowledge text input
    cag_knowledge = st.text_area(
        "Knowledge Text", 
        height=300,
        value=st.session_state.knowledge_text if st.session_state.knowledge_text else "",
        placeholder="Enter the knowledge text to preload into the model's KV cache..."
    )
    
    # System prompt (for HuggingFace models)
    if st.session_state.backend == "huggingface":
        system_prompt = st.text_area(
            "System Prompt (optional)",
            height=100,
            placeholder="Optional system instructions for the model"
        )
    else:
        system_prompt = None
    
    # Preload button
    if st.button("Preload Knowledge", type="primary") and cag_knowledge:
        if not st.session_state.generator:
            st.error("No generator initialized. Please check your backend settings.")
        else:
            with st.spinner("Preloading knowledge into KV cache..."):
                try:
                    knowledge_id = st.session_state.generator.preload_knowledge(
                        knowledge=cag_knowledge,
                        model_name=selected_model,
                        system_prompt=system_prompt
                    )
                    
                    # Save the knowledge text for later use
                    st.session_state.knowledge_text = cag_knowledge
                    st.session_state.preloaded_knowledge_id = knowledge_id
                    
                    st.success("Knowledge successfully preloaded into KV cache!")
                    if knowledge_id:
                        st.info(f"Knowledge ID: {knowledge_id}")
                except Exception as e:
                    st.error(f"Error preloading knowledge: {str(e)}")
    
    # Show preloaded knowledge if available
    if st.session_state.knowledge_text:
        st.subheader("Currently Preloaded Knowledge")
        with st.expander("View preloaded knowledge"):
            st.text(st.session_state.knowledge_text)
        
        if st.button("Clear Preloaded Knowledge"):
            st.session_state.knowledge_text = None
            st.session_state.preloaded_knowledge_id = None
            st.success("Preloaded knowledge cleared")
            st.rerun()
    
    

# Tab 4: Benchmark
with tab4:
    st.header("Benchmark CAG Performance")
    st.markdown(
        """
        Compare the performance of standard generation vs. CAG (with preloaded knowledge).
        """
    )
    
    # Knowledge text for benchmark
    benchmark_knowledge = st.text_area(
        "Knowledge Text for Benchmark", 
        height=200,
        placeholder="Enter knowledge text for the benchmark..."
    )
    
    # Query text
    benchmark_query = st.text_area(
        "Query", 
        height=100,
        placeholder="Enter a query about the knowledge..."
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        num_runs = st.slider("Number of benchmark runs", 1, 5, 3)
    
    with col2:
        pass  # Empty column for layout
    
    if st.button("Run CAG Benchmark", type="primary") and benchmark_knowledge and benchmark_query:
        if not st.session_state.generator:
            st.error("No generator initialized. Please check your backend settings.")
        else:
            with st.spinner("Running benchmark..."):
                results = st.session_state.generator.benchmark_cag(
                    query=benchmark_query,
                    model_name=selected_model,
                    knowledge_text=benchmark_knowledge,
                    runs=num_runs,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens
                )
                
                st.session_state.benchmark_results = results
    
    # Display benchmark results
    if st.session_state.benchmark_results:
        results = st.session_state.benchmark_results
        
        st.divider()
        st.subheader("Benchmark Results")
        
        # Create comparison table
        data = {
            "Method": ["Standard (Knowledge in Prompt)", "CAG (Preloaded Knowledge)"],
            "Avg. Time (s)": [
                results.get("standard_avg_time", 0),
                results.get("cag_avg_time", 0)
            ],
            "Speedup": [
                "1.00x (baseline)",
                f"{results.get('speedup', 1):.2f}x"
            ]
        }
        
        # Convert to DataFrame and display
        df = pd.DataFrame(data)
        st.table(df)
        
        # Create bar chart for speedup
        fig, ax = plt.subplots(figsize=(10, 6))
        
        methods = ["Standard", "CAG"]
        speedups = [1.0, results.get("speedup", 1)]
        
        ax.bar(methods, speedups, color=['blue', 'green'])
        ax.set_ylabel('Speedup Factor')
        ax.set_title('Generation Speedup with CAG')
        ax.axhline(y=1, color='r', linestyle='-', alpha=0.3)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        for i, v in enumerate(speedups):
            ax.text(i, v + 0.1, f'{v:.2f}x', ha='center')
        
        st.pyplot(fig)
        
        # Display example output
        st.subheader("Example Outputs")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Standard Approach Output:**")
            if results["standard_approach"]:
                st.text(results["standard_approach"][0]["text"])
        
        with col2:
            st.markdown("**CAG Approach Output:**")
            if results["cag_approach"]:
                st.text(results["cag_approach"][0]["text"])
    else:
        st.info("Enter knowledge and a query, then click 'Run CAG Benchmark' to start!")

# Tab 5: About CAG
with tab5:
    st.header("About Cache-Augmented Generation (CAG)")
    
    st.markdown(
        """
        ### What is Cache-Augmented Generation?

        Cache-Augmented Generation (CAG) is a technique for enhancing language model inference by preloading
        knowledge directly into the model's key-value (KV) cache, rather than including it in the prompt.
        
        ### CAG vs RAG (Retrieval-Augmented Generation)
        
        **Retrieval-Augmented Generation (RAG)** works by:
        1. Storing knowledge as vectors in a database
        2. Converting the query to a vector and finding similar knowledge vectors
        3. Including retrieved knowledge in the prompt
        
        **Cache-Augmented Generation (CAG)** works by:
        1. Preloading all knowledge directly into the model's KV cache
        2. Keeping this processed knowledge in the cache between queries
        3. Processing only the query during inference
        
        ### Key Benefits of CAG
        
        - **Speed**: Much faster inference by avoiding reprocessing the knowledge
        - **Simplicity**: No vector database or embedding model needed
        - **Efficiency**: Directly leverages the model's internal mechanism
        
        ### How This Implementation Works
        
        This application supports two approaches to CAG:
        
        **HuggingFace Implementation**:
        - Directly manipulates the transformer model's KV cache
        - Efficiently preloads knowledge text and saves the resulting KV cache
        - Truncates the KV cache after each generation to maintain consistency
        
        **Ollama Implementation**:
        - Uses Ollama's API for local model inference
        - Implements CAG by intelligently caching full prompts and prefix matches
        - Provides similar benefits with models running through Ollama
        
        ### When to Use CAG
        
        CAG is most effective when:
        - Working with a stable knowledge base
        - Running multiple queries against the same knowledge
        - Optimizing for response time
        - Working with knowledge that fits within the model's context window
        
        For very large knowledge bases that exceed the model's context window, a traditional RAG approach may be more appropriate.
        """
    )

# Footer
st.divider()
st.caption("Cache-Augmented Generation | KV Cache-Based Knowledge Integration for LLMs") 