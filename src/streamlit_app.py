import os
import streamlit as st
import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any, List, Optional
import io

# Import our modules (assuming they're in the same directory)
from kv_cache_generator import KVCacheGenerator
from ollama_client import OllamaGenerationOptions
from document_processor import DocumentProcessor

# Set page configuration
st.set_page_config(
    page_title="KV Cache Augmented Generation",
    page_icon="⚡",
    layout="wide",
)

# Initialize session state
if "kv_generator" not in st.session_state:
    st.session_state.kv_generator = KVCacheGenerator(
        cache_dir=os.path.abspath("../cache"),
        ollama_url="http://localhost:11434",
    )

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

# App title and description
st.title("⚡ KV Cache Augmented Generation")
st.markdown(
    """
    This application demonstrates Key-Value Cache Augmented Generation using Ollama models.
    It allows you to see the performance benefits of reusing KV caches between generations.
    
    **Models available:** Mistral and DeepSeek-R1 (loaded locally via Ollama)
    """
)

# Sidebar for configurations
with st.sidebar:
    # Cache settings
    st.header("Cache Settings")
    use_cache = st.checkbox("Use KV Cache", value=True)
    save_cache = st.checkbox("Save KV Cache", value=True)
        
    if st.button("Clear Cache"):
        st.session_state.kv_generator.cache_manager.clear_cache()
        st.success("Cache cleared successfully!")
        
    # Model settings
    st.divider()
    st.header("Model Settings")
    
    # Get available models
    try:
        available_models = st.session_state.kv_generator.list_available_models()
        if not available_models:
            available_models = ["mistral", "deepseek-r1"]  # Fallback options
    except Exception as e:
        st.error(f"Error connecting to Ollama: {str(e)}")
        st.error("Make sure Ollama is running on http://localhost:11434")
        available_models = ["mistral", "deepseek-r1"]  # Fallback options
    
    selected_model = st.selectbox("Model", available_models)
    
    st.divider()
    st.header("Generation Parameters")
    
    # Generation parameters
    temperature = st.slider("Temperature", 0.0, 1.0, 0.5, 0.1)
    top_p = st.slider("Top P", 0.0, 1.0, 0.9, 0.01)
    top_k = st.slider("Top K", 1, 100, 50, 1)
    max_tokens = st.slider("Max Tokens", 32, 4096, 2048, 16)
    

# Main content area - tabs
tab1, tab2, tab3, tab4 = st.tabs(["Text Generation", "Knowledge Base", "Benchmark", "About KV Cache"])

# Tab 1: Text Generation
with tab1:
    # Text input area
    prompt = st.text_area("Enter your prompt", height=150)
    
    # Add context from documents
    use_knowledge_base = st.checkbox("Use Knowledge Base", value=False)
    
    if use_knowledge_base:
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
                                current_prompt = prompt if prompt else ""
                                prompt = f"{current_prompt}\n\nContext:\n{chunk_text}\n\n"
                                st.experimental_rerun()
                else:
                    st.info("No relevant information found in knowledge base.")
    
    # Create two columns for buttons
    col1, col2 = st.columns(2)
    
    with col1:
        generate_clicked = st.button("Generate Text", type="primary")
    
    with col2:
        clear_clicked = st.button("Clear History")
    
    # Handle generation
    if generate_clicked and prompt:
        with st.spinner("Generating..."):
            # Create options
            options = OllamaGenerationOptions(
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_tokens=max_tokens,
            )
            
            # Generate with KV cache
            result = st.session_state.kv_generator.generate(
                prompt=prompt,
                model=selected_model,
                options=options,
                use_cache=use_cache,
                save_cache=save_cache,
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
                    
                    st.markdown(f"- **Cache Status:** {cache_status}")
                    st.markdown(f"- **Tokens Generated:** {result.get('tokens_generated', 0)}")
                    st.markdown(f"- **Generation Time:** {result.get('total_duration', 0):.2f}s")
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
                    st.experimental_rerun()
                    
                if st.button("View Chunks", key=f"view_{doc['id']}"):
                    chunks = st.session_state.doc_processor.get_document_chunks(doc['id'])
                    for i, chunk in enumerate(chunks):
                        with st.expander(f"Chunk {i+1}"):
                            st.text(chunk)
    else:
        st.info("No documents in the knowledge base. Upload a document to get started.")

# Tab 3: Benchmark
with tab3:
    st.header("Benchmark KV Cache Performance")
    st.markdown(
        """
        This benchmark compares generation times with and without KV cache.
        It runs multiple generations with the same prompt and reports the speedup.
        """
    )
    
    benchmark_prompt = st.text_area("Benchmark prompt", height=150)
    
    col1, col2 = st.columns(2)
    
    with col1:
        num_runs = st.slider("Number of runs per method", 1, 5, 2)
    
    with col2:
        warm_up = st.checkbox("Do warm-up run", value=True)
    
    if st.button("Run Benchmark", type="primary") and benchmark_prompt:
        with st.spinner("Running benchmark..."):
            # Create options
            options = OllamaGenerationOptions(
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_tokens=max_tokens,
            )
            
            # Run benchmark
            results = st.session_state.kv_generator.benchmark(
                prompt=benchmark_prompt,
                model=selected_model,
                options=options,
                runs=num_runs,
                warm_up=warm_up
            )
            
            # Store results
            st.session_state.benchmark_results = results
    
    # Display benchmark results
    if st.session_state.benchmark_results:
        results = st.session_state.benchmark_results
        
        st.divider()
        st.subheader("Benchmark Results")
        
        # Create comparison table
        data = {
            "Method": ["No Cache", "With Full Cache", "With Prefix Cache*"],
            "Avg. Time (s)": [
                results.get("no_cache_avg_time", 0),
                results.get("cache_avg_time", 0),
                results.get("prefix_cache_avg_time", 0) if "prefix_cache_avg_time" in results else "-"
            ],
            "Avg. Speed (tokens/s)": [
                results.get("no_cache_avg_tps", 0),
                results.get("cache_avg_tps", 0),
                results.get("prefix_cache_avg_tps", 0) if "prefix_cache_avg_tps" in results else "-"
            ],
            "Speedup": [
                "1.00x (baseline)",
                f"{results.get('cache_speedup', 1):.2f}x",
                f"{results.get('prefix_cache_speedup', 1):.2f}x" if "prefix_cache_speedup" in results else "-"
            ]
        }
        
        # Convert to DataFrame and display
        df = pd.DataFrame(data)
        st.table(df)
        
        # Create bar chart for speedup
        fig, ax = plt.subplots(figsize=(10, 6))
        
        methods = ["No Cache", "Full Cache"]
        speedups = [1.0, results.get("cache_speedup", 1)]
        
        if "prefix_cache_speedup" in results:
            methods.append("Prefix Cache")
            speedups.append(results.get("prefix_cache_speedup", 1))
        
        ax.bar(methods, speedups, color=['blue', 'green', 'orange'][:len(methods)])
        ax.set_ylabel('Speedup Factor')
        ax.set_title('Generation Speedup with KV Cache')
        ax.axhline(y=1, color='r', linestyle='-', alpha=0.3)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        for i, v in enumerate(speedups):
            ax.text(i, v + 0.1, f'{v:.2f}x', ha='center')
        
        st.pyplot(fig)
        
        st.caption("* Prefix Cache only shown if benchmark included prefix testing")
    else:
        st.info("Enter a prompt and click 'Run Benchmark' to start!")

# Tab 4: Info about KV Cache
with tab4:
    st.header("About KV Cache Augmented Generation")
    
    st.markdown(
        """
        ### What is KV Cache?

        In transformer-based language models, the **Key-Value (KV) Cache** is a performance optimization technique. 
        When generating text, the model needs to compute attention over all tokens it has processed. 
        Without caching, the model would need to recompute the key and value projections for all previous tokens at each generation step.

        The KV cache stores these key and value projections for all processed tokens, so they don't need to be recomputed, making generation much faster.

        ### What is KV Cache Augmentation?

        KV Cache Augmentation extends this concept between separate generation requests:

        1. When you generate text with a prompt, the system saves the KV cache
        2. If you generate text with the same prompt again, the saved KV cache is reused
        3. If you generate text with a prompt that has a common prefix with a cached prompt, the system can reuse part of the cached KV state
        
        This approach can provide significant speedups, especially for applications that:
        - Process similar prompts repeatedly
        - Have standard prefixes (like system prompts) that are reused across different queries
        - Need to optimize for low latency
        
        ### Knowledge Base Augmentation

        This application also demonstrates knowledge base augmentation:

        1. Upload text documents to create a knowledge base
        2. Documents are split into chunks for easier processing
        3. When generating text, you can search the knowledge base for relevant context
        4. The context is added to your prompt, providing the model with additional information
        
        This approach allows the model to generate more accurate and informed responses based on your documents.
        
        ### Benefits Demonstrated in this App
        
        This application shows:
        
        1. **Full Cache Match**: When the exact prompt is reused, showing the maximum possible speedup
        2. **Prefix Cache Match**: When a prompt starts with a cached prompt, demonstrating partial KV reuse
        3. **No Cache**: Baseline performance without KV cache reuse
        4. **Knowledge Base Integration**: Using document context to enhance generation quality
        
        The benchmark tab quantifies these benefits precisely with your specific hardware and chosen model.
        """
    )

# Footer
st.divider()
st.caption("KV Cache Augmented Generation | Using Ollama and local models") 