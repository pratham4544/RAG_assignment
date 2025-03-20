from typing import List, Dict
import numpy as np
import os
import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
import PyPDF2

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

class Document:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.content = None
        self.chunks = None
        self.embeddings = None
        
    def load_document(self) -> str:
        """
        Load document content from file.
        Handle both PDF and TXT formats.
        
        Returns:
            str: The text content of the document
        """
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"File not found: {self.file_path}")
            
        file_extension = os.path.splitext(self.file_path)[1].lower()
        
        if file_extension == '.txt':
            # Process text file
            with open(self.file_path, 'r', encoding='utf-8') as file:
                self.content = file.read()
        elif file_extension == '.pdf':
            # Process PDF file
            text = ""
            with open(self.file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text += page.extract_text()
            self.content = text
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
            
        return self.content
    
    def chunk_text(self, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """
        Split document into overlapping chunks.
        
        Args:
            chunk_size (int): Number of characters per chunk
            overlap (int): Number of characters to overlap between chunks
            
        Returns:
            List[str]: List of text chunks
        """
        if self.content is None:
            raise ValueError("Document content is not loaded. Call load_document() first.")
            
        # Split text into sentences
        sentences = sent_tokenize(self.content)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            # If adding this sentence would exceed chunk size, store current chunk and start a new one
            if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
                chunks.append(current_chunk)
                # Start new chunk with overlap from the previous chunk
                overlap_size = min(overlap, len(current_chunk))
                current_chunk = current_chunk[-overlap_size:] + sentence
            else:
                current_chunk += sentence
        
        # Add the last chunk if it's not empty
        if current_chunk:
            chunks.append(current_chunk)
            
        self.chunks = chunks
        return chunks
    
    def preprocess_text(self, text: str) -> List[str]:
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
    
    def generate_embeddings(self, chunks: List[str] = None, method: str = 'tfidf') -> List[np.ndarray]:
        """
        Create embeddings for text chunks.
        Use either TF-IDF or word2vec approach.
        
        Args:
            chunks (List[str], optional): List of text chunks. If None, use self.chunks
            method (str): Embedding method ('tfidf' or 'word2vec')
            
        Returns:
            List[np.ndarray]: List of embedding vectors
        """
        if chunks is None:
            if self.chunks is None:
                raise ValueError("No chunks available. Call chunk_text() first.")
            chunks = self.chunks
            
        if method == 'tfidf':
            # TF-IDF embeddings
            vectorizer = TfidfVectorizer(max_features=100)
            tfidf_matrix = vectorizer.fit_transform(chunks)
            self.embeddings = [tfidf_matrix[i].toarray()[0] for i in range(len(chunks))]
            
        elif method == 'word2vec':
            # Word2Vec embeddings
            # Preprocess text for word2vec
            preprocessed_chunks = [self.preprocess_text(chunk) for chunk in chunks]
            
            # Train Word2Vec model
            model = Word2Vec(sentences=preprocessed_chunks, vector_size=100, window=5, min_count=1, workers=4)
            
            # Create document embeddings by averaging word vectors
            embeddings = []
            for chunk_tokens in preprocessed_chunks:
                # Get word vectors for all tokens in the chunk
                vectors = [model.wv[token] for token in chunk_tokens if token in model.wv]
                
                # If no vectors are found, use a zero vector
                if not vectors:
                    embeddings.append(np.zeros(100))
                else:
                    # Average the word vectors
                    embeddings.append(np.mean(vectors, axis=0))
                    
            self.embeddings = embeddings
        else:
            raise ValueError(f"Unsupported embedding method: {method}")
            
        return self.embeddings
