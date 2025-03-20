from typing import List, Dict
import numpy as np
import os
import time
import pickle
import hashlib

# Import our classes
# Assuming all classes are in the same file or properly imported
from document_class import Document
from vector_store_class import VectorStore
from mock_llm_class import MockLLM
from query_processor import process_query

class Cache:
    def __init__(self, cache_dir: str = "cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
    def _get_cache_key(self, data: str) -> str:
        """Generate a unique cache key for the given data."""
        return hashlib.md5(data.encode()).hexdigest()
        
    def get(self, key: str) -> any:
        """Retrieve data from cache."""
        cache_file = os.path.join(self.cache_dir, f"{key}.pkl")
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        return None
        
    def set(self, key: str, data: any):
        """Store data in cache."""
        cache_file = os.path.join(self.cache_dir, f"{key}.pkl")
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)

def main():
    """
    Main function to demonstrate the document retrieval system.
    """
    print("Document Retrieval System Demo")
    print("-" * 50)
    
    # Initialize cache
    cache = Cache()
    
    # Set file path - adjust as needed
    file_path = "data/paper1.pdf"
    
    # Check if file exists, if not try a text file
    if not os.path.exists(file_path):
        file_path = "data/paper1.txt"
        # If still doesn't exist, create a sample text file
        if not os.path.exists(file_path):
            print(f"Error: Required file {file_path} not found.")
            return
    
    try:
        # Initialize system
        print(f"Loading document: {file_path}")
        doc_processor = Document(file_path)
        vector_store = VectorStore()
        mock_llm = MockLLM()
        
        # Process document
        print("Processing document...")
        content = doc_processor.load_document()
        print(f"Document loaded: {len(content)} characters")
        
        # Check cache for embeddings
        content_hash = cache._get_cache_key(content)
        cached_embeddings = cache.get(f"embeddings_{content_hash}")
        
        if cached_embeddings is not None:
            print("Using cached embeddings...")
            chunks, embeddings = cached_embeddings
        else:
            chunks = doc_processor.chunk_text(chunk_size=200, overlap=50)
            print(f"Document split into {len(chunks)} chunks")
            
            # Use Word2Vec for embeddings
            embedding_method = "word2vec"
            print(f"Generating embeddings using {embedding_method}...")
            embeddings = doc_processor.generate_embeddings(chunks, method=embedding_method)
            print(f"Created {len(embeddings)} embeddings")
            
            # Cache the embeddings
            cache.set(f"embeddings_{content_hash}", (chunks, embeddings))
        
        # Add embeddings to vector store with metadata
        print("Adding embeddings to vector store...")
        metadata = []
        for i in range(len(chunks)):
            metadata.append({
                "source": os.path.basename(file_path),
                "chunk_index": i,
                "char_count": len(chunks[i])
            })
        vector_store.add_embeddings(embeddings, metadata)
        
        # Process queries
        sample_queries = [
            "What are the two primary solar energy technologies?",
            "What is the typical power range of commercial onshore wind turbines?",
            "What percentage of global electricity generation is provided by hydroelectric power?"
        ]
        
        for query in sample_queries:
            print("\n" + "=" * 50)
            print(f"Processing query: '{query}'")
            
            # Check cache for query results
            query_hash = cache._get_cache_key(query)
            cached_response = cache.get(f"query_{query_hash}")
            
            if cached_response is not None:
                print("Using cached query result...")
                response = cached_response
                processing_time = 0.0  # No processing time for cached results
            else:
                # Process query
                start_time = time.time()
                response = process_query(query, vector_store, mock_llm, chunks, embedding_method)
                processing_time = time.time() - start_time
                
                # Cache the response
                cache.set(f"query_{query_hash}", response)
            
            # Display results
            print(f"Query processed in {processing_time:.2f} seconds")
            
            if response["success"]:
                print("\nResponse:")
                print(response["data"]["text"])
                print(f"\nTokens used: {response['data']['tokens_used']}")
            else:
                print(f"\nError: {response['error']}")
                
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
