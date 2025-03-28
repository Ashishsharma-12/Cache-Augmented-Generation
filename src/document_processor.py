import os
import re
import json
from typing import List, Dict, Any, Optional, Tuple
import hashlib

class DocumentProcessor:
    """
    Process documents for knowledge caching.
    """
    
    def __init__(self, cache_dir: str = "../cache/docs"):
        """
        Initialize the document processor.
        
        Args:
            cache_dir: Directory to store document caches
        """
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.doc_index = self._load_doc_index()
    
    def _load_doc_index(self) -> Dict[str, Any]:
        """Load document index from disk or create a new one."""
        index_path = os.path.join(self.cache_dir, "doc_index.json")
        if os.path.exists(index_path):
            with open(index_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {"documents": {}, "chunks": {}}
    
    def _save_doc_index(self):
        """Save document index to disk."""
        index_path = os.path.join(self.cache_dir, "doc_index.json")
        with open(index_path, 'w', encoding='utf-8') as f:
            json.dump(self.doc_index, f, indent=2)
    
    def _generate_file_id(self, file_content: str, filename: str) -> str:
        """Generate a unique ID for a file."""
        content_hash = hashlib.md5(file_content.encode()).hexdigest()
        return f"{os.path.basename(filename).split('.')[0]}_{content_hash[:8]}"
    
    def _chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Text to split
            chunk_size: Maximum chunk size in characters
            overlap: Overlap between chunks in characters
            
        Returns:
            List of text chunks
        """
        chunks = []
        start = 0
        text_len = len(text)
        
        while start < text_len:
            end = min(start + chunk_size, text_len)
            
            # Try to end at a sentence boundary if possible
            if end < text_len:
                # Find the last sentence boundary within the chunk
                last_period = text.rfind('. ', start, end)
                if last_period > start + chunk_size * 0.7:  # Only use if at least 70% into the chunk
                    end = last_period + 1  # Include the period
            
            # Extract the chunk
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start position, with overlap
            start = end - overlap if end < text_len else text_len
        
        return chunks
    
    def process_document(self, content: str, filename: str, chunk_size: int = 1000, overlap: int = 200) -> str:
        """
        Process a document and add it to the index.
        
        Args:
            content: Document content
            filename: Original file name
            chunk_size: Maximum chunk size in characters
            overlap: Overlap between chunks in characters
            
        Returns:
            Document ID
        """
        # Generate document ID
        doc_id = self._generate_file_id(content, filename)
        
        # Create chunks
        chunks = self._chunk_text(content, chunk_size, overlap)
        
        # Store document info
        self.doc_index["documents"][doc_id] = {
            "filename": os.path.basename(filename),
            "content_length": len(content),
            "chunks": [],
            "timestamp": os.path.getmtime(filename) if os.path.exists(filename) else None,
        }
        
        # Process and store chunks
        for i, chunk_text in enumerate(chunks):
            chunk_id = f"{doc_id}_chunk_{i}"
            
            # Store chunk
            chunk_file = os.path.join(self.cache_dir, f"{chunk_id}.txt")
            with open(chunk_file, 'w', encoding='utf-8') as f:
                f.write(chunk_text)
            
            # Add to index
            self.doc_index["chunks"][chunk_id] = {
                "document_id": doc_id,
                "chunk_index": i,
                "content_length": len(chunk_text),
                "path": chunk_file,
            }
            
            self.doc_index["documents"][doc_id]["chunks"].append(chunk_id)
        
        # Save the updated index
        self._save_doc_index()
        
        return doc_id
    
    def get_all_documents(self) -> List[Dict[str, Any]]:
        """
        Get all documents in the index.
        
        Returns:
            List of document info dictionaries
        """
        documents = []
        for doc_id, doc_info in self.doc_index["documents"].items():
            documents.append({
                "id": doc_id,
                "filename": doc_info["filename"],
                "chunks": len(doc_info["chunks"]),
            })
        return documents
    
    def get_document_chunks(self, doc_id: str) -> List[str]:
        """
        Get all chunks for a document.
        
        Args:
            doc_id: Document ID
            
        Returns:
            List of chunk texts
        """
        if doc_id not in self.doc_index["documents"]:
            return []
        
        chunks = []
        for chunk_id in self.doc_index["documents"][doc_id]["chunks"]:
            if chunk_id in self.doc_index["chunks"]:
                chunk_info = self.doc_index["chunks"][chunk_id]
                with open(chunk_info["path"], 'r', encoding='utf-8') as f:
                    chunks.append(f.read())
        
        return chunks
    
    def search_documents(self, query: str) -> List[Tuple[str, str, float]]:
        """
        Very simple text search across document chunks.
        
        Args:
            query: Search query
            
        Returns:
            List of (chunk_id, chunk_text, score) tuples sorted by relevance
        """
        results = []
        query_terms = set(re.findall(r'\w+', query.lower()))
        
        if not query_terms:
            return []
        
        for chunk_id, chunk_info in self.doc_index["chunks"].items():
            try:
                with open(chunk_info["path"], 'r', encoding='utf-8') as f:
                    chunk_text = f.read()
                    
                # Simple term matching score
                chunk_terms = set(re.findall(r'\w+', chunk_text.lower()))
                matching_terms = query_terms.intersection(chunk_terms)
                
                if matching_terms:
                    score = len(matching_terms) / len(query_terms)
                    results.append((chunk_id, chunk_text, score))
            except Exception as e:
                print(f"Error reading chunk {chunk_id}: {e}")
        
        # Sort by score descending
        results.sort(key=lambda x: x[2], reverse=True)
        return results[:5]  # Return top 5 results
    
    def remove_document(self, doc_id: str) -> bool:
        """
        Remove a document and its chunks from the index.
        
        Args:
            doc_id: Document ID
            
        Returns:
            True if successful, False otherwise
        """
        if doc_id not in self.doc_index["documents"]:
            return False
        
        # Remove chunk files and index entries
        for chunk_id in self.doc_index["documents"][doc_id]["chunks"]:
            if chunk_id in self.doc_index["chunks"]:
                chunk_path = self.doc_index["chunks"][chunk_id]["path"]
                if os.path.exists(chunk_path):
                    os.remove(chunk_path)
                del self.doc_index["chunks"][chunk_id]
        
        # Remove document from index
        del self.doc_index["documents"][doc_id]
        self._save_doc_index()
        
        return True 