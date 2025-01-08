import os
import time
import json
import uuid
import numpy as np
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec, Index

# Configuration
PINECONE_API_KEY = "pcsk_22reMi_CoW2s5jpVBSzePsuNTNPNbLgVzEb6ZzUynpCpGkECDH2D22NAz4eFdJd94RPmgj"
ENVIRONMENT = "us-east-1"
INDEX_NAME = "regulations"
JSON_FILE_PATH = r"C:\Users\gallo\OneDrive\Desktop\Regulations\extracted_ECB_TRIM2017clean.json"
INDEX_HOST = "https://regulations-bf89341.svc.aped-4627-b74a.pinecone.io"
MODEL_NAME = "all-mpnet-base-v2"
NAMESPACE = "EECB_TRIM2017"
TARGET_DIMENSION = 768
MAX_RETRIES = 3
RETRY_DELAY = 2

def get_or_create_index(pc: Pinecone, index_name: str) -> None:
    """Get existing index or create a new one if it doesn't exist."""
    try:
        # First try to describe the index
        try:
            index_info = pc.describe_index(index_name)
            print(f"Found existing index '{index_name}' with dimension {index_info.dimension}")
            return
        except Exception as e:
            if "not found" not in str(e).lower():
                raise
            
            # Index doesn't exist, create it
            print(f"Creating new index '{index_name}'...")
            pc.create_index(
                name=index_name,
                dimension=TARGET_DIMENSION,
                metric='cosine',
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-east-1'
                )
            )
            
            # Wait for index to be ready
            print("Waiting for index to be ready...")
            while True:
                try:
                    status = pc.describe_index(index_name).status
                    if status.get('ready'):
                        break
                    time.sleep(1)
                except Exception:
                    time.sleep(1)
            
            print(f"Index '{index_name}' created and ready")
            
    except Exception as e:
        print(f"Error with index operation: {e}")
        raise

def initialize_pinecone(api_key: str, index_host: str, index_name: str) -> Index:
    """Initialize Pinecone client and return index."""
    try:
        # Initialize Pinecone
        pc = Pinecone(api_key=api_key)
        
        # Ensure index exists
        get_or_create_index(pc, index_name)
        
        # Connect to the index
        index = pc.Index(host=index_host)
        
        # Get current stats
        try:
            stats = index.describe_index_stats()
            print("\nCurrent index statistics:")
            print(f"Total vectors: {stats.total_vector_count}")
            for ns, count in stats.namespaces.items():
                print(f"  Namespace '{ns}': {count.vector_count} vectors")
        except Exception as e:
            print(f"Warning: Could not get index stats: {e}")
        
        return index
        
    except Exception as e:
        print(f"Error initializing Pinecone: {e}")
        raise

def generate_unique_id(section_name: str, document_name: str) -> str:
    """Generate a unique ID incorporating section and document info."""
    # Remove special characters and spaces, keep alphanumeric
    clean_section = ''.join(c for c in section_name if c.isalnum())
    clean_doc = ''.join(c for c in document_name if c.isalnum())
    timestamp = int(time.time())
    unique = uuid.uuid4().hex[:6]
    return f"{clean_doc}_{clean_section}_{timestamp}_{unique}"

def pad_embedding(embedding: List[float], target_dim: int) -> List[float]:
    """Pad embedding to target dimension by repeating values."""
    current_dim = len(embedding)
    if current_dim >= target_dim:
        return embedding[:target_dim]
    
    num_repeats = target_dim // current_dim
    remainder = target_dim % current_dim
    
    padded = embedding * num_repeats
    padded.extend(embedding[:remainder])
    
    return padded

def generate_embeddings(texts: List[str], model_name: str = MODEL_NAME) -> List[List[float]]:
    """Generate embeddings and pad them to target dimension."""
    print(f"\nGenerating embeddings using model: sentence-transformers/{model_name}")
    model = SentenceTransformer(f'sentence-transformers/{model_name}')
    base_embeddings = model.encode(texts, show_progress_bar=True)
    
    padded_embeddings = []
    for emb in base_embeddings:
        padded = pad_embedding(emb.tolist(), TARGET_DIMENSION)
        padded_embeddings.append(padded)
    
    print(f"Padded embeddings from {len(base_embeddings[0])} to {len(padded_embeddings[0])} dimensions")
    return padded_embeddings

def verify_vector_upload(index: Index, vector_id: str, namespace: str, max_retries: int = 3) -> bool:
    """Verify that a specific vector was uploaded with retries."""
    for attempt in range(max_retries):
        try:
            result = index.fetch(ids=[vector_id], namespace=namespace)
            if vector_id in result.get('vectors', {}):
                # Print metadata to verify it was stored correctly
                vector_data = result['vectors'][vector_id]
                print(f"Verified vector {vector_id} with metadata:")
                print(json.dumps(vector_data.get('metadata', {}), indent=2))
                return True
            print(f"Vector {vector_id} not found, attempt {attempt + 1}/{max_retries}")
            time.sleep(RETRY_DELAY)
        except Exception as e:
            print(f"Error verifying vector {vector_id}: {e}")
            time.sleep(RETRY_DELAY)
    return False

def upsert_vectors(index: Index, vectors: List[Dict[str, Any]], namespace: str, batch_size: int = 5):
    """Upsert vectors to Pinecone with improved error handling and verification."""
    print(f"\nStarting vector upload to namespace: '{namespace}'")
    
    initial_stats = index.describe_index_stats()
    initial_count = initial_stats.namespaces.get(namespace, {}).get('vector_count', 0)
    print(f"Initial vector count in namespace: {initial_count}")
    
    successful_uploads = 0
    
    # Print first vector's metadata for verification
    print("\nSample vector metadata being uploaded:")
    print(json.dumps(vectors[0]['metadata'], indent=2))
    
    for i in range(0, len(vectors), batch_size):
        batch_vectors = vectors[i:i + batch_size]
        retry_count = 0
        
        while retry_count < MAX_RETRIES:
            try:
                response = index.upsert(vectors=batch_vectors, namespace=namespace)
                print(f"Batch {i//batch_size + 1}: Attempted upload of {len(batch_vectors)} vectors")
                
                time.sleep(2)
                
                verified_count = 0
                for vector in batch_vectors:
                    if verify_vector_upload(index, vector['id'], namespace):
                        verified_count += 1
                
                print(f"Batch {i//batch_size + 1}: Verified {verified_count}/{len(batch_vectors)} vectors")
                successful_uploads += verified_count
                break
                
            except Exception as e:
                print(f"Error in batch {i//batch_size + 1}, attempt {retry_count + 1}: {e}")
                retry_count += 1
                if retry_count < MAX_RETRIES:
                    print(f"Retrying in {RETRY_DELAY} seconds...")
                    time.sleep(RETRY_DELAY)
                else:
                    print(f"Failed to upload batch after {MAX_RETRIES} attempts")
                    raise
    
    final_stats = index.describe_index_stats()
    final_count = final_stats.namespaces.get(namespace, {}).get('vector_count', 0)
    
    print(f"\nUpload Summary:")
    print(f"  Initial count: {initial_count}")
    print(f"  Final count: {final_count}")
    print(f"  Successfully verified uploads: {successful_uploads}")
    print(f"  Expected to add: {len(vectors)}")

def main():
    # Initialize Pinecone with index creation
    try:
        index = initialize_pinecone(PINECONE_API_KEY, INDEX_HOST, INDEX_NAME)
    except Exception as e:
        print(f"Failed to initialize Pinecone: {e}")
        return

    # Load JSON data
    print(f"Loading data from {JSON_FILE_PATH}")
    with open(JSON_FILE_PATH, "r", encoding="utf-8") as f:
        sections = json.load(f)
    
    # Generate embeddings
    texts = [section["text"] for section in sections]
    embeddings = generate_embeddings(texts)
    
    # Prepare vectors with complete metadata
    vectors = [
        {
            "id": generate_unique_id(
                section["section_name"],
                section["document_name"]
            ),
            "values": embedding,
            "metadata": {
                "text": section["text"],
                "section_name": section["section_name"],
                "document_name": section["document_name"],
                "page_start": section["page_start"],
                "upload_timestamp": int(time.time())
            }
        }
        for section, embedding in zip(sections, embeddings)
    ]
    
    # Print sample vector for verification
    print("\nSample vector structure:")
    sample_vector = vectors[0]
    print(f"ID: {sample_vector['id']}")
    print("Metadata:")
    print(json.dumps(sample_vector['metadata'], indent=2))
    
    # Upsert vectors
    upsert_vectors(index, vectors, namespace=NAMESPACE, batch_size=5)

if __name__ == "__main__":
    main()