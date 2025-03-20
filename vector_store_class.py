from typing import List, Dict
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class VectorStore:
    def __init__(self):
        self.embeddings = []
        self.metadata = []
        
    def add_embeddings(self, embeddings: List[np.ndarray], metadata: List[dict]):
        """
        Store embeddings with their metadata.
        
        Args:
            embeddings (List[np.ndarray]): List of embedding vectors
            metadata (List[dict]): List of metadata dictionaries corresponding to embeddings
        """
        if len(embeddings) != len(metadata):
            raise ValueError("Number of embeddings must match number of metadata entries")
            
        self.embeddings.extend(embeddings)
        self.metadata.extend(metadata)
        
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[dict]:
        """
        Find most similar chunks using cosine similarity.
        Return chunks with their metadata.
        
        Args:
            query_embedding (np.ndarray): Embedding vector of the query
            k (int): Number of results to return
            
        Returns:
            List[dict]: List of dictionaries containing results and metadata
        """
        if not self.embeddings:
            return []
            
        # Ensure query embedding has the right shape
        query_embedding = query_embedding.reshape(1, -1)
        
        # Calculate cosine similarity between query and all embeddings
        similarities = []
        for embedding in self.embeddings:
            # Reshape embedding to match the query
            embedding_reshaped = embedding.reshape(1, -1)
            # Calculate similarity
            similarity = cosine_similarity(query_embedding, embedding_reshaped)[0][0]
            similarities.append(similarity)
            
        # Get indices of top k results
        if k > len(similarities):
            k = len(similarities)
            
        top_indices = np.argsort(similarities)[-k:][::-1]  # Sort in descending order
        
        # Prepare results
        results = []
        for idx in top_indices:
            result = {
                "similarity": similarities[idx],
                "metadata": self.metadata[idx].copy()  # Return a copy of metadata
            }
            results.append(result)
            
        return results
