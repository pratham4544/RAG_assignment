# RAG (Retrieval-Augmented Generation) System

A Python implementation of a Retrieval-Augmented Generation (RAG) system that combines document retrieval with language model generation.

## Features

- Document processing (PDF and TXT support)
- Text chunking with overlap
- Multiple embedding methods (TF-IDF and Word2Vec)
- Vector similarity search
- Caching system for embeddings and query results
- Mock LLM with rate limiting and error simulation
- Clean and modular code structure

## Project Structure

```
.
├── data/               # Directory for input documents
├── cache/             # Directory for cached embeddings and results
├── helper/            # Helper modules
│   ├── cache.py      # Caching system
│   ├── document.py   # Document processing
│   ├── vector_store.py # Vector storage and search
│   ├── mock_llm.py   # Mock language model
│   └── query_processor.py # Query processing
├── main-script.py     # Main application script
└── requirements.txt   # Project dependencies
```

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Place your documents in the `data` directory

## Usage

1. Run the main script:
   ```bash
   python main-script.py
   ```

2. The script will:
   - Load and process the document
   - Generate embeddings (using Word2Vec by default)
   - Process sample queries
   - Cache results for future use

## Configuration

- Document chunking: Adjust `chunk_size` and `overlap` in `main-script.py`
- Embedding method: Change `embedding_method` in `main-script.py` ('tfidf' or 'word2vec')
- Cache location: Modify `cache_dir` in the `Cache` class initialization

## Dependencies

- numpy>=1.21.0
- scikit-learn>=0.24.2
- nltk>=3.6.3
- gensim>=4.1.2
- PyPDF2>=3.0.0

## Notes

- The system uses a mock LLM for demonstration purposes
- Caching is implemented for both embeddings and query results
- Word2Vec is used as the default embedding method
- The system supports both PDF and TXT input formats
