from typing import List, Dict
import numpy as np
from helper.document import Document
from helper.vector_store import VectorStore
from helper.mock_llm import MockLLM

def process_query(query: str, vector_store: VectorStore, llm: MockLLM, chunks: List[str], embedding_method: str = 'word2vec') -> Dict:
    """
    Process a query using the RAG pipeline.
    
    Args:
        query (str): User's question
        vector_store (VectorStore): Vector store containing document embeddings
        llm (MockLLM): Language model for generating responses
        chunks (List[str]): List of document chunks
        embedding_method (str): Method used for embeddings ('tfidf' or 'word2vec')
        
    Returns:
        Dict: Response containing generated text and metadata
    """
    # Create a temporary document processor for the query
    doc_processor = Document("")  # Empty file path since we're just processing text
    
    # Generate query embedding using the same method as the document
    if embedding_method == 'tfidf':
        # For TF-IDF, we need to use the same vectorizer as the document
        # This is a simplified version - in practice, you'd want to save and reuse the vectorizer
        query_embedding = doc_processor.generate_embeddings([query], method='tfidf')[0]
    else:  # word2vec
        query_embedding = doc_processor.generate_embeddings([query], method='word2vec')[0]
    
    # Search for relevant chunks
    results = vector_store.search(query_embedding, k=3)
    
    # Extract context from the most relevant chunks
    context = []
    for result in results:
        chunk_idx = result['metadata']['chunk_index']
        context.append(chunks[chunk_idx])
    
    # Generate response using the LLM
    response = llm.generate_response(query, context)
    
    return response 