from typing import List, Dict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re

def preprocess_text(text: str) -> List[str]:
    """
    Preprocess text for embedding generation.
    
    Args:
        text (str): Input text
        
    Returns:
        List[str]: List of preprocessed tokens
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    # Stemming
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(token) for token in tokens]
    
    return tokens

def embed_query(query: str, chunks: List[str], method: str = 'tfidf') -> np.ndarray:
    """
    Create embedding for a query using the same method as the documents.
    
    Args:
        query (str): Query text
        chunks (List[str]): Document chunks used to train the embedding model
        method (str): Embedding method ('tfidf' or 'word2vec')
        
    Returns:
        np.ndarray: Query embedding vector
    """
    if method == 'tfidf':
        # Add query to the chunks to ensure same vocabulary
        all_texts = chunks + [query]
        vectorizer = TfidfVectorizer(max_features=100)
        tfidf_matrix = vectorizer.fit_transform(all_texts)
        # Get the last vector (query)
        query_embedding = tfidf_matrix[-1].toarray()[0]
        
    elif method == 'word2vec':
        # Preprocess query and chunks
        preprocessed_query = preprocess_text(query)
        preprocessed_chunks = [preprocess_text(chunk) for chunk in chunks]
        
        # Train Word2Vec model on chunks
        model = Word2Vec(sentences=preprocessed_chunks, vector_size=100, window=5, min_count=1, workers=4)
        
        # Get word vectors for query tokens
        vectors = [model.wv[token] for token in preprocessed_query if token in model.wv]
        
        # If no vectors are found, use a zero vector
        if not vectors:
            query_embedding = np.zeros(100)
        else:
            # Average the word vectors
            query_embedding = np.mean(vectors, axis=0)
    else:
        raise ValueError(f"Unsupported embedding method: {method}")
        
    return query_embedding

def process_query(query: str, vector_store, mock_llm, chunks: List[str] = None, embedding_method: str = 'tfidf') -> dict:
    """
    Process a user query and generate a response.
    
    Args:
        query (str): User's question
        vector_store: VectorStore instance
        mock_llm: MockLLM instance
        chunks (List[str], optional): Document chunks for embedding generation
        embedding_method (str): Method for embedding generation
        
    Returns:
        dict: Response from the mock LLM
    """
    try:
        # Embed the query
        query_embedding = embed_query(query, chunks, method=embedding_method)
        
        # Search for relevant chunks
        search_results = vector_store.search(query_embedding, k=3)
        
        # Extract chunks from metadata
        # This assumes the chunks are stored in metadata or can be retrieved from somewhere
        context_chunks = []
        for result in search_results:
            # Get original chunk text based on metadata
            chunk_index = result["metadata"].get("chunk_index", 0)
            if chunks and chunk_index < len(chunks):
                context_chunks.append(chunks[chunk_index])
            
        # Generate response
        response = mock_llm.generate_response(query, context_chunks)
        
        return response
    except Exception as e:
        # Handle errors
        return {
            "success": False,
            "error": str(e),
            "data": None
        }
