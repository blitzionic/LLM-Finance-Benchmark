import os
import sys
from pathlib import Path
from llama_index.core import (
    GPTVectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    Settings,
    load_index_from_storage,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from dotenv import load_dotenv
from tqdm import tqdm

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

def build_index(documents_dir: str, persist_dir: str = "storage"):
    """
    Build and persist the RAG index from documents.
    
    Args:
        documents_dir: Directory containing the documents to index
        persist_dir: Directory to persist the index to
    """
    # skips if index_store.json exists 
    index_path = os.path.join(persist_dir, "index_store.json")
    if os.path.exists(index_path):
        print("⚠️ Index already exists. Skipping re-build.")
        return
    
    print(f"Loading documents from {documents_dir}")
    
    # Load and clean documents
    documents = SimpleDirectoryReader(documents_dir).load_data()
    print(f"Loaded {len(documents)} documents")
    
    # Set up embeddings
    embed_model = HuggingFaceEmbedding(model_name="all-MiniLM-L6-v2")
    print(f"Using embedding model: {embed_model.model_name}")
    
    # Build index with progress bar
    print("Building index...")
    print("This may take a few minutes depending on the size of your documents.")
    
    # Create a progress bar for document processing
    with tqdm(total=len(documents), desc="Processing documents") as pbar:
        def progress_callback(doc_num):
            pbar.update(1)
        
        index = GPTVectorStoreIndex.from_documents(
            documents, 
            embed_model=embed_model,
            show_progress=True,
        )
    
    # Persist index
    print(f"Persisting index to {persist_dir}")
    index.storage_context.persist(persist_dir=persist_dir)
    print("✅ Index built and persisted successfully!")

def get_query_engine(persist_dir: str = "storage"):
    """
    Get or create a query engine from the persisted index.
    
    Args:
        persist_dir: Directory where the index is persisted
    
    Returns:
        QueryEngine: The configured query engine for retrieval only
    """
    # Load index from storage
    storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
    index = load_index_from_storage(storage_context)
    
    # Create query engine for retrieval only
    query_engine = index.as_query_engine(
        embed_model = HuggingFaceEmbedding(model_name="all-MiniLM-L6-v2"), 
        similarity_top_k = 3
    )
    
    return query_engine

def retrieve_chunks(query: str, persist_dir: str = "storage"):
    """
    Retrieve relevant chunks for a query without generating an answer.
    
    Args:
        query: The question to find relevant chunks for
        persist_dir: Directory where the index is persisted
    
    Returns:
        List of retrieved chunks with their relevance scores
    """
    query_engine = get_query_engine(persist_dir)
    chunks = query_engine.retrieve(query)
    
    # Print the retrieved chunks
    print(f"\nRetrieved chunks for query: '{query}'")
    print("-" * 80)
    for i, chunk in enumerate(chunks, 1):
        print(f"\nChunk {i}:")
        print(f"Score: {chunk.score:.4f}")
        print(f"Text: {chunk.text}")
        print("-" * 80)
    
    return chunks

def main():
    # Load environment variables
    load_dotenv()
    
    # Get documents directory from environment or use default
    documents_dir = os.path.join(project_root, "data/financial_docs")
    
    # Get persist directory from environment or use default
    persist_dir = os.path.join(project_root, "src/scripts/storage")
    
    # Create persist directory if it doesn't exist
    os.makedirs(persist_dir, exist_ok=True)
    
    # Build and persist index if needed
    build_index(documents_dir, persist_dir)
    
    # Example of retrieving chunks
    query = "What is capital gains tax?"
    retrieve_chunks(query, persist_dir)

if __name__ == "__main__":
    main()
