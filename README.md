# Cache-Augmented Generation (CAG)

A complete implementation of Cache-Augmented Generation, a technique for faster LLM inference by preloading knowledge into KV cache. This project provides a Streamlit-based UI that supports both HuggingFace and Ollama models.

## Features

- **Cache-Augmented Generation**: Preload knowledge into KV cache for faster inference
- **HuggingFace Integration**: Direct KV cache manipulation with HuggingFace transformers models
- **Ollama Support**: Use locally running Ollama models for inference
- **Knowledge Base**: Upload and process documents for knowledge augmentation
- **Document Search**: Find relevant document chunks to augment prompts
- **CAG Benchmarking**: Compare standard generation vs. CAG for performance metrics
- **Streamlit UI**: User-friendly interface for all features

## What is CAG?

Cache-Augmented Generation (CAG) is a technique that enhances language model inference by preloading knowledge directly into the model's key-value (KV) cache, rather than including it in each prompt. Unlike RAG (Retrieval-Augmented Generation), which embeds and retrieves knowledge through vector similarity, CAG directly leverages the model's internal caching mechanism.

### CAG vs. RAG

**Retrieval-Augmented Generation (RAG)**:
1. Stores knowledge as vectors in a database
2. Converts queries to vectors to find similar knowledge
3. Includes retrieved knowledge in the prompt

**Cache-Augmented Generation (CAG)**:
1. Preloads all knowledge directly into the model's KV cache
2. Keeps processed knowledge in cache between queries
3. Processes only the query during inference

CAG provides significant speedups by avoiding reprocessing the same knowledge for each query.

## Prerequisites

- Python 3.8 or higher
- For Ollama backend:
  - [Ollama](https://ollama.ai/) installed and running locally
  - Local LLM models (e.g., Mistral and DeepSeek-R1)
- For HuggingFace backend:
  - CUDA-compatible GPU recommended
  - Access to HuggingFace models (e.g., Llama-3.1)

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/cache-augmented-generation.git
   cd cache-augmented-generation
   ```

2. Install required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

3. For Ollama backend, ensure Ollama is installed and running:
   - Download from [ollama.ai](https://ollama.ai/)
   - Start the Ollama service
   - Pull the models you want to use, e.g.:
     ```bash
     ollama pull mistral
     ollama pull deepseek-r1
     ```

## Usage

1. Run the application:
   ```bash
   python run.py
   ```

2. The application will open in your default web browser at http://localhost:8501

3. Using the interface:

   **Backend Selection**:
   - Choose between Ollama and HuggingFace
   - For HuggingFace, load a model by entering its name (e.g., `meta-llama/Meta-Llama-3.1-8B-Instruct`)

   **Knowledge Base Tab**:
   - Upload documents to create a knowledge base
   - Documents are automatically split into manageable chunks

   **CAG Preloading Tab**:
   - Enter or paste knowledge text to preload into the model's KV cache
   - You can also load documents from your knowledge base

   **Text Generation Tab**:
   - Enter your prompt
   - Optionally use knowledge from the knowledge base or preloaded knowledge
   - Generate text with the selected model

   **Benchmark Tab**:
   - Compare standard generation with CAG
   - See speed improvements and output quality

## How It Works

The implementation includes two approaches to CAG:

**HuggingFace Implementation**:
- Directly manipulates the transformer model's KV cache
- Efficiently preloads knowledge text and saves the resulting KV cache
- Truncates the KV cache after each generation to maintain consistency

**Ollama Implementation**:
- Uses Ollama's API for local model inference
- Implements CAG by intelligently caching full prompts and prefix matches
- Provides similar benefits with models running through Ollama

## Project Structure

```
kv_cache_project/
├── cache/                   # Storage for various caches
│   ├── docs/                # Processed document chunks
│   ├── hf_cache/            # HuggingFace KV caches
│   └── ollama_cache/        # Ollama KV caches
├── src/
│   ├── __init__.py          # Package initialization
│   ├── kv_cache_manager.py  # Manages KV cache for Ollama
│   ├── hf_cache_manager.py  # Manages KV cache for HuggingFace
│   ├── ollama_client.py     # Client for Ollama API
│   ├── cag_generator.py     # High-level CAG implementation
│   ├── document_processor.py # Processes documents
│   └── streamlit_app.py     # Streamlit UI application
├── requirements.txt         # Python dependencies
├── run.py                   # Main entry point
└── README.md                # This file
```

## When to Use CAG

CAG is most effective when:
- Working with a stable knowledge base
- Running multiple queries against the same knowledge
- Optimizing for response time
- Working with knowledge that fits within the model's context window

For very large knowledge bases that exceed the model's context window, a traditional RAG approach may be more appropriate.

## License

This project is released under the MIT License. See the LICENSE file for details.

## Acknowledgments

- [Ollama](https://ollama.ai/) for providing easy access to local LLMs
- [Streamlit](https://streamlit.io/) for the interactive UI framework
- The original CAG research paper for the conceptual framework 