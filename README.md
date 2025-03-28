# KV Cache Augmented Generation

This project demonstrates Key-Value (KV) Cache Augmented Generation for improving the performance of text generation with large language models. It provides a Streamlit-based UI to interact with local LLMs through Ollama, with the ability to reuse KV caches between generations and augment prompts with knowledge from uploaded documents.

## Features

- **KV Cache Reuse**: Reuse key-value pairs from previous generations to speed up inference
- **Prefix Matching**: Leverage partial caches when the new prompt starts with a cached prompt
- **Knowledge Base**: Upload documents to provide context for text generation
- **Document Search**: Find relevant document chunks to augment prompts
- **Benchmarking**: Compare generation performance with and without KV cache
- **Streamlit UI**: User-friendly interface for interacting with models
- **Ollama Integration**: Works with locally-run models via Ollama

## Prerequisites

- Python 3.8 or higher
- [Ollama](https://ollama.ai/) installed and running locally
- Local LLM models (this project was tested with Mistral and DeepSeek-R1)

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/kv-cache-project.git
   cd kv-cache-project
   ```

2. Install required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure Ollama is installed and running:
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
   - Select a model from the dropdown in the sidebar
   - Adjust generation parameters as needed
   - Upload documents in the Knowledge Base tab
   - Search and use document contexts in the Text Generation tab
   - Enter a prompt and generate text
   - Use the Benchmark tab to measure speedup with KV cache

## How It Works

### KV Cache in Transformer Models

In transformer models, generation is done token by token. For each token:
1. The model computes attention over all previously processed tokens
2. For each token, the model calculates "key" and "value" vectors for each attention layer
3. Without caching, these calculations are repeated for every token at each generation step

KV caching saves these key-value vectors from previous calculations, avoiding redundant computation and significantly speeding up generation.

### Cross-Request KV Cache Augmentation

This project extends KV caching beyond a single generation request:

1. When generating text with a prompt, we save the KV cache to disk
2. For subsequent requests with the same prompt, we load and reuse this cache
3. For new prompts that share a prefix with cached prompts, we reuse the relevant portion of the cache

This approach can provide substantial speedups, especially for applications with:
- Repeated prompts or system instructions
- Common prefixes in prompts
- Need for low-latency generation

### Knowledge Base Augmentation

The application also supports knowledge base augmentation:

1. Upload text documents in the Knowledge Base tab
2. Documents are automatically split into manageable chunks
3. When generating text, search for relevant information in your documents
4. Add selected context to your prompt to provide the model with additional information

This allows the model to generate more accurate and contextually relevant responses by incorporating information from your documents.

## Project Structure

```
kv_cache_project/
├── cache/               # Storage for KV caches
│   └── docs/            # Processed document chunks
├── src/
│   ├── __init__.py      # Package initialization
│   ├── kv_cache_manager.py    # Manages KV cache storage and retrieval
│   ├── ollama_client.py       # Client for interacting with Ollama API
│   ├── kv_cache_generator.py  # Combines KV cache with generation
│   ├── document_processor.py  # Processes and manages document chunks
│   └── streamlit_app.py       # Streamlit UI application
├── requirements.txt     # Python dependencies
├── run.py               # Main entry point
└── README.md            # This file
```

## Limitations

- KV cache format is model-specific and may need adjustments for different models
- Large caches can consume significant disk space
- Prefix matching works best with substantial common prefixes
- Performance gains depend on model architecture and hardware
- Document search uses simple keyword matching (not semantic search)

## License

This project is released under the MIT License. See the LICENSE file for details.

## Acknowledgments

- [Ollama](https://ollama.ai/) for providing easy access to local LLMs
- [Streamlit](https://streamlit.io/) for the interactive UI framework 