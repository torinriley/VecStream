"""
Example script demonstrating in-memory vector storage and search with VecStream.
"""

from vecstream.vector_store import VectorStore
import numpy as np

def main():
    # Initialize the in-memory vector store
    vector_store = VectorStore()
    
    # Sample data to store (simulated document embeddings)
    documents = [
        "Machine learning is a subset of artificial intelligence",
        "Python is a popular programming language",
        "Neural networks are inspired by biological brains",
        "Data science involves statistical analysis",
        "Deep learning is revolutionizing AI"
    ]
    
    # Dictionary to store document texts
    doc_texts = {}
    
    print("🔄 Adding documents to vector store...")
    # Add documents with unique IDs
    for i, doc in enumerate(documents):
        # Simulate document embedding (normally you'd use a proper embedding model)
        vector = np.random.random(384).astype(np.float32).tolist()
        doc_id = f"doc_{i}"
        vector_store.add_vector(id=doc_id, vector=vector)
        doc_texts[doc_id] = doc
    
    print("\n📚 Current documents in store:")
    for i in range(len(documents)):
        doc_id = f"doc_{i}"
        vector = vector_store.get_vector(doc_id)
        print(f"ID: {doc_id} | Text: {doc_texts[doc_id]}")
    
    # Perform a similarity search
    print("\n🔍 Performing similarity search...")
    query_vector = np.random.random(384).astype(np.float32).tolist()
    results = vector_store.search_similar(query_vector, k=3)
    
    print("\nTop 3 similar documents:")
    for doc_id, score in results:
        print(f"Score: {score:.4f} | Text: {doc_texts[doc_id]}")
    
    # Demonstrate removing a document
    print("\n🗑️ Removing document 'doc_0'...")
    vector_store.remove_vector("doc_0")
    doc_texts.pop("doc_0")
    
    # Verify removal
    print("\n📚 Remaining documents:")
    for i in range(1, len(documents)):
        doc_id = f"doc_{i}"
        try:
            vector = vector_store.get_vector(doc_id)
            print(f"ID: {doc_id} | Text: {doc_texts[doc_id]}")
        except KeyError:
            continue

if __name__ == "__main__":
    main() 