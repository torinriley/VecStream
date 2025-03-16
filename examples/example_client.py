#!/usr/bin/env python3
from sentence_transformers import SentenceTransformer
from vecstream.client import VectorDBClient
import time

def main():
    # Initialize the client
    client = VectorDBClient()
    
    # Initialize the sentence transformer model
    print("Loading sentence transformer model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Example sentences
    sentences = {
        'sent1': 'The quick brown fox jumps over the lazy dog',
        'sent2': 'A lazy dog sleeps in the sun',
        'sent3': 'The cat chases a mouse in the garden',
    }
    
    # Store sentences as vectors
    print("\nStoring sentences as vectors...")
    for id, text in sentences.items():
        vector = model.encode(text)
        response = client.add_vector(id, vector)
        print(f"Added {id}: {response}")
        
    # Retrieve a vector
    print("\nRetrieving a vector...")
    vector = client.get_vector('sent1')
    print(f"Retrieved vector shape: {vector.shape}")
    
    # Search for similar sentences
    print("\nSearching for similar sentences...")
    query = "A dog resting in the sunshine"
    query_vector = model.encode(query)
    
    response = client.search_similar(query_vector, k=3)
    print(f"\nResults for query: '{query}'")
    
    if response['status'] == 'success':
        for id, similarity in response['results']:
            print(f"- '{sentences[id]}' (Similarity: {similarity:.4f})")
            
    # Clean up
    print("\nClearing database...")
    client.clear_database()

if __name__ == '__main__':
    main() 