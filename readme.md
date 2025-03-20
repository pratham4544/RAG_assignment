# Document Retrieval System

A Python-based document retrieval system that helps researchers find relevant information from academic papers. This system implements the core components of a retrieval-augmented architecture.

## Features

- Document processing (PDF and TXT formats)
- Text chunking with configurable overlap
- Embedding generation (TF-IDF and Word2Vec methods)
- In-memory vector storage
- Semantic search using cosine similarity
- Query processing and results ranking
- Mock LLM integration with error handling and rate limiting

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- Required Python packages (see requirements.txt)

### Installation

1. Clone this repository or download the source code
2. Create a virtual environment:
   ```
   python -m venv venv
   ```
3. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - macOS/Linux: `source venv/bin/activate`
4. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
5. Download NLTK resources:
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   ```

### Project Structure

```
document-retrieval-system/
├── data/                   # Folder for document files
│   └── paper1.txt          # Sample document (automatically created if missing)
├── document_class.py       # Document processing implementation
├── vector_store_class.py   # Vector storage implementation
├── mock_llm_class.py       # Mock LLM implementation
├── query_processor.py      # Query processing functions
├── main.py                 # Main script to run the system
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## Usage

1. Place your PDF or TXT files in the `data/` directory
2. Run the main script:
   ```
   python main.py
   ```
3. The script will process the document, generate embeddings, and run sample queries

### Example Code

```python
# Initialize system
doc_processor = Document("data/paper1.pdf")
vector_store = VectorStore()
mock_llm = MockLLM()

# Process document
content = doc_processor.load_document()
chunks = doc_processor.chunk_text()
embeddings = doc_processor.generate_embeddings(chunks)

# Add embeddings to vector store
metadata = [{"source": "paper1.pdf", "chunk_index": i} for i in range(len(chunks))]
vector_store.add_embeddings(embeddings, metadata)

# Process query
query = "What are the main impacts of climate change on agriculture?"
response = process_query(query, vector_store, mock_llm, chunks)
print(f"Response: {response}")
```

## Design Decisions

### Document Processing

- **Chunking Strategy**: Documents are split into chunks by sentences rather than arbitrary character limits to maintain semantic coherence. This helps preserve the meaning of text segments.
- **Overlap Implementation**: Chunks overlap by keeping a portion of the previous chunk when creating a new one, ensuring context is maintained across chunk boundaries.
- **File Type Support**: The system handles both PDF and TXT files with appropriate error handling for unsupported formats.

### Embedding Generation

- **Multiple Methods**: Support for both TF-IDF and Word2Vec embeddings:
  - TF-IDF: Simple but effective for keyword-based retrieval
  - Word2Vec: Better semantic understanding but more computationally intensive
- **Text Preprocessing**: Includes lowercase conversion, stopword removal, and stemming to improve embedding quality.

### Vector Storage & Search

- **In-Memory Storage**: Simple array-based storage for embeddings and metadata for quick access.
- **Cosine Similarity**: Used for measuring similarity between query and document embeddings.
- **Result Ranking**: Results are sorted by similarity score in descending order.

### Error Handling

- **Robust Error Management**: Comprehensive error handling for file operations, rate limiting, and processing failures.
- **Rate Limiting**: Simulation of API rate limits to mimic real-world constraints.
- **Error Simulation**: Random errors are simulated to test system resilience.

## Limitations and Potential Improvements

### Current Limitations

1. **Memory Usage**: The in-memory storage is not suitable for very large document collections.
2. **Basic Embedding Methods**: The implemented embedding methods are simple compared to state-of-the-art techniques.
3. **No Persistence**: Data is not saved between program runs.
4. **Limited File Format Support**: Only PDF and TXT formats are supported.

### Potential Improvements

1. **Database Integration**: Add support for persistent storage using SQLite or another database.
2. **Advanced Embeddings**: Implement more sophisticated embedding methods like BERT or sentence transformers.
3. **Caching**: Implement caching for frequent queries to improve performance.
4. **API Interface**: Create a REST API for the system to enable client applications to interact with it.
5. **Additional File Formats**: Add support for DOCX, HTML, and other common formats.
6. **Parallel Processing**: Implement multithreading for handling multiple documents simultaneously.

## Performance Metrics

The system's performance will vary based on:
- Document size and complexity
- Number of chunks and embedding dimensions
- Query complexity and frequency
- Hardware capabilities

On average, you can expect:
- Document processing: 1-5 seconds per document
- Query processing: 0.1-1 second per query
- Typical accuracy: Depends on embedding method and document quality

## Error Handling Examples

The system handles several error scenarios:
- File not found or invalid format
- Rate limit exceeded
- Timeout or server errors
- Invalid input format

Example response for rate limit error:
```json
{
  "success": false,
  "error": "Rate limit exceeded. Try again in 45.23 seconds",
  "data": null
}
```
