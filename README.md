# RAG System with Ollama and LangChain

This project implements a Retrieval Augmented Generation (RAG) system using Ollama, LangChain, and Chroma DB.

## Setup

1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Make sure Ollama is installed and running on your system.

3. Pull the required models:
   ```
   ollama pull llama3.2
   ollama pull mxbai-embed-large
   ```

4. Create necessary directories

- data
- db

## Usage

### Document Ingestion

To ingest documents into the vector database:

```
python ingest.py
```

This will process documents from the `data/` directory and store them in the Chroma DB.

### Querying Documents

There are three different implementations for querying the documents:

1. **Standard Implementation (with tqdm progress bar)**:
   ```
   python chat.py
   ```
   This implementation uses the `tqdm` library to display a progress bar while Ollama is processing your query.

2. **Simple Implementation (no dependencies)**:
   ```
   python chat_simple.py
   ```
   This implementation uses a simple spinner animation without requiring additional dependencies.

3. **Streaming Implementation**:
   ```
   python chat_streaming.py
   ```
   This implementation uses Ollama's streaming capabilities to show real-time token generation, providing the most detailed feedback during processing.

## How It Works

1. **Document Ingestion**: Documents are split into chunks, embedded, and stored in a Chroma vector database.
2. **Querying**: When you ask a question, the system:
   - Retrieves the most relevant document chunks using vector similarity search
   - Sends these chunks along with your question to the LLM
   - Returns the LLM's response based on the provided context

## Progress Indicators

All three implementations provide visual feedback while Ollama is processing your query:

- **chat.py**: Uses a simple spinning animation

Choose the implementation that best suits your needs and preferences. 

## Credits

This code wasn't possible if I didn't see the following video first: https://youtu.be/kw7qwgfVeLA?si=xWK3g1Bsyc-IXqbT
