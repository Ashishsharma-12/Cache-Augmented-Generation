#!/usr/bin/env python3
"""
Run script for the KV Cache Augmented Generation application.
"""

import os
import subprocess
import sys
import argparse

def main():
    """Main entry point for the application"""
    parser = argparse.ArgumentParser(description="Run KV Cache Augmented Generation application")
    parser.add_argument("--port", type=int, default=8501, help="Port to run Streamlit on")
    args = parser.parse_args()
    
    # Change to script's directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Check if Ollama is running
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code != 200:
            print("⚠️ Warning: Ollama API is not responding. Make sure Ollama is running.")
    except Exception:
        print("⚠️ Warning: Ollama API is not responding. Make sure Ollama is running.")
        print("   You can download Ollama from: https://ollama.ai/")
        print("   After installing, run: ollama pull mistral")
    
    # Run Streamlit (let Streamlit handle browser opening)
    cmd = [
        sys.executable, "-m", "streamlit", "run", 
        os.path.join("src", "streamlit_app.py"),
        "--server.port", str(args.port),
    ]
    
    print(f"Starting KV Cache Augmented Generation on port {args.port}...")
    subprocess.run(cmd)

if __name__ == "__main__":
    main() 