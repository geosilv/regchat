import os
import time
import json
import uuid
import numpy as np
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec, Index
from glob import glob

# Configuration
PINECONE_API_KEY = "pcsk_22reMi_CoW2s5jpVBSzePsuNTNPNbLgVzEb6ZzUynpCpGkECDH2D22NAz4eFdJd94RPmgj"
ENVIRONMENT = "us-east-1"
INDEX_NAME = "regulations3"
INDEX_HOST = "https://regulations3-bf89341.svc.aped-4627-b74a.pinecone.io"
MODEL_NAME = "all-mpnet-base-v2"
TARGET_DIMENSION = 768
MAX_RETRIES = 3
RETRY_DELAY = 2

# Directory containing JSON files to process
JSON_FILES_DIRECTORY = r"C:\Users\gallo\source\VSCode\RegulationsProject\LanchainProcessedDocs\Final_to_upload"

def get_or_create_index(pc: Pinecone, index_name: str) -> None:
    """Get existing index or create a new one if it doesn't exist."""
    try:
        try:
            index_info = pc.describe_index(index_name)
            print(f"Found existing index '{index_name}' with dimension {index_info.dimension}")
            return
        except Exception as e:
            if "not found" not in str(e).lower():
                raise
            
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
        pc = Pinecone(api_key=api_key)
        get_or_create_index(pc, index_name)
        index = pc.Index(host=index_host)
        
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

def generate_unique_id(document_title: str, page_number: int, chunk_index: int) -> str:
    """Generate a unique ID incorporating document and chunk info."""
    filename = os.path.basename(document_title)
    clean_doc = ''.join(c for c in filename if c.isalnum())
    timestamp = int(time.time())
    unique = uuid.uuid4().hex[:6]
    return f"{clean_doc}_p{page_number}_c{chunk_index}_{timestamp}_{unique}"

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

def process_json_file(file_path: str, index: Index):
    """Process a single JSON file and upload its contents to Pinecone."""
    def validate_chunk(chunk):
        """Validate that a chunk has the minimum required fields."""
        if not isinstance(chunk, dict):
            return False, "Chunk must be a dictionary"
        if "text" not in chunk:
            return False, "Chunk missing required 'text' field"
        if "metadata" not in chunk:
            return False, "Chunk missing required 'metadata' field"
        if not isinstance(chunk["metadata"], dict):
            return False, "Chunk metadata must be a dictionary"
        required_metadata = ["document_title", "page_number", "chunk_index"]
        missing = [field for field in required_metadata if field not in chunk["metadata"]]
        if missing:
            return False, f"Chunk metadata missing required fields: {', '.join(missing)}"
        return True, None
    try:
        print(f"\nProcessing file: {file_path}")
        
        # Extract namespace from filename (without extension)
        namespace = os.path.splitext(os.path.basename(file_path))[0]
        print(f"Using namespace: {namespace}")
        
        # Load JSON data
        with open(file_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        
        # Ensure chunks is a list
        if not isinstance(chunks, list):
            raise ValueError(f"Expected JSON array in {file_path}, got {type(chunks)}")
        
        # Validate all chunks before processing
        valid_chunks = []
        for i, chunk in enumerate(chunks):
            is_valid, error_msg = validate_chunk(chunk)
            if is_valid:
                valid_chunks.append(chunk)
            else:
                print(f"Warning: Skipping invalid chunk {i} in {file_path}: {error_msg}")
        
        if not valid_chunks:
            raise ValueError(f"No valid chunks found in {file_path}")
        
        print(f"Found {len(valid_chunks)} valid chunks out of {len(chunks)} total chunks")
        
        # Generate embeddings for valid texts
        texts = [chunk["text"] for chunk in valid_chunks]
        embeddings = generate_embeddings(texts)
        
        # Prepare vectors with metadata
        vectors = []
        for chunk, embedding in zip(chunks, embeddings):
            # Extract metadata fields
            metadata = chunk["metadata"].copy()
            
            # Add fields from the chunk with defaults if not present
            metadata.update({
                "section_name": chunk.get("section_name", ""),
                "page_start": chunk.get("page_start", metadata.get("page_number", 0)),
                "text": chunk["text"]  # text is required
            })
            
            vector = {
                "id": generate_unique_id(
                    metadata["document_title"],
                    metadata["page_number"],
                    metadata["chunk_index"]
                ),
                "values": embedding,
                "metadata": {
                    **metadata,
                    "upload_timestamp": int(time.time())
                }
            }
            vectors.append(vector)
        
        # Upload vectors to namespace
        upsert_vectors(index, vectors, namespace=namespace, batch_size=5)
        
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        raise

def main():
    """Main function that processes all JSON files in the specified directory."""
    # Initialize Pinecone
    try:
        index = initialize_pinecone(PINECONE_API_KEY, INDEX_HOST, INDEX_NAME)
    except Exception as e:
        print(f"Failed to initialize Pinecone: {e}")
        return

    # Get list of JSON files in directory
    json_files = glob(os.path.join(JSON_FILES_DIRECTORY, "*.json"))
    print(f"\nFound {len(json_files)} JSON files in directory: {JSON_FILES_DIRECTORY}")
    
    # Process each JSON file
    for file_path in json_files:
        try:
            process_json_file(file_path, index)
        except Exception as e:
            print(f"Failed to process {file_path}: {e}")
            continue

if __name__ == "__main__":
    main()