import streamlit as st
from sentence_transformers import SentenceTransformer
import Complexity
import tiktoken


def count_tokens(text: str, model: str = "gpt-4") -> int:
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

def generate_embedding(text: str, model) -> list:
    try:
        embedding = model.encode(text)
        return embedding.tolist()
    except Exception as e:
        st.error(f"Error generating embedding: {str(e)}")
        return []

def determine_search_params(complexity_score: int) -> tuple:
    """Determine search parameters based on complexity score."""
    if complexity_score >= 6:
        max_results = 15  # High complexity
        results_per_namespace = 2
    elif complexity_score >= 3:
        max_results = 12   # Medium complexity
        results_per_namespace = 2
    else:
        max_results = 7   # Low complexity
        results_per_namespace = 1
    
    return max_results, results_per_namespace

def get_namespace_results(query_embedding: list, namespace: str, results_per_namespace: int, 
                         index, max_results: int, used_ids: set) -> tuple:
    """Query a single namespace and return results."""
    try:
        # First, check if namespace exists and has vectors
        stats = index.describe_index_stats()
        #st.write(f"Debug: Index stats for {namespace}: {stats.namespaces.get(namespace, 'namespace not found')}")
        
        # Try query with lower threshold and more results to debug
        results = index.query(
            vector=query_embedding,
            top_k=10,  # Ask for more results
            include_metadata=True,
            namespace=namespace,
            include_values=False  # Save bandwidth
        )
        
        #st.write(f"Debug: Raw results from pinecone: {len(results.matches) if results.matches else 0}")
        #if results.matches:
            #st.write(f"Debug: Top match score: {results.matches[0].score}")
        
        namespace_matches = []
        if results.matches:
            for match in results.matches[:results_per_namespace]:
                if len(namespace_matches) < max_results and match.id not in used_ids:
                    namespace_matches.append(match)
                    used_ids.add(match.id)
        
        #st.write(f"Debug: Filtered results: {len(namespace_matches)}")
        return namespace_matches, used_ids
        
    except Exception as e:
        st.error(f"Error querying namespace {namespace}: {str(e)}")
        #st.write(f"Debug: Full error: {str(e)}")
        return [], used_ids
    
def fill_remaining_slots(query_embedding: list, namespace: str, remaining_slots: int, 
                        index, used_ids: set) -> tuple:
    """Fill remaining slots from a namespace."""
    try:
        results = index.query(
            vector=query_embedding,
            top_k=remaining_slots + 3,
            include_metadata=True,
            namespace=namespace
        )
        
        additional_matches = []
        for match in results.matches:
            if match.id not in used_ids and len(additional_matches) < remaining_slots:
                additional_matches.append(match)
                used_ids.add(match.id)
                
        return additional_matches, used_ids
        
    except Exception as e:
        return [], used_ids

def search_regulations(query: str, index, model) -> list:
    """Main search function that coordinates the search process."""
    try:
        # Check index status first
        try:
            stats = index.describe_index_stats()
            #st.write(f"Debug: Index stats: {stats}")
            #st.write(f"Debug: Available namespaces: {list(stats.namespaces.keys())}")
            
            # Check if index is empty
            total_vectors = sum(ns.vector_count for ns in stats.namespaces.values())
            #st.write(f"Debug: Total vectors in index: {total_vectors}")
            
            if total_vectors == 0:
                st.error("Index appears to be empty")
                return []
                
        except Exception as e:
            st.error(f"Error checking index stats: {str(e)}")
        
        # Rest of the search logic...
        complexity_score = Complexity.calculate_complexity_score(query)
        max_results, results_per_namespace = determine_search_params(complexity_score)
        
        query_embedding = generate_embedding(query, model)
        if not query_embedding or len(query_embedding) != 768:
            st.error("Invalid embedding generated")
            return []
 
        if len(query_embedding) != 768:
            st.error(f"Invalid embedding dimension: {len(query_embedding)}")
            return []

        # Define namespaces
        namespace_docs = {
            "processed_ECB_GIM_Feb24_processed": "ECB GIM Feb 2024",
            "processed_ECB_TRIM2017_processed": "ECB TRIM 2017",
            "processed_PRA_ss123_processed": "PRA SS1/23",
            "processed_JFSA_2021_processed": "JFSA 2021",
            "processed_FED_sr1107a1_processed": "FED SR 11-7a1"
        }
        
        # First pass: Get results from each namespace
        all_results = []
        used_ids = set()
        
        for namespace in namespace_docs:
            #st.write(f"Debug: Searching namespace {namespace}")
            namespace_matches, used_ids = get_namespace_results(
                query_embedding, namespace, results_per_namespace, 
                index, max_results, used_ids
            )
            #st.write(f"Debug: Found {len(namespace_matches)} matches in {namespace}")
            all_results.extend(namespace_matches)
            
            if len(all_results) >= max_results:
                break
        
        #st.write(f"Debug: Total results after first pass = {len(all_results)}")
        
        # Second pass: Fill remaining slots if needed
        if len(all_results) < max_results:
            remaining_slots = max_results - len(all_results)
            #st.write(f"Debug: Filling {remaining_slots} remaining slots")
            
            for namespace in namespace_docs:
                if len(all_results) >= max_results:
                    break
                    
                additional_matches, used_ids = fill_remaining_slots(
                    query_embedding, namespace, remaining_slots, 
                    index, used_ids
                )
                #st.write(f"Debug: Found {len(additional_matches)} additional matches in {namespace}")
                all_results.extend(additional_matches)
        
        # Sort and return results
        all_results.sort(key=lambda x: x.score, reverse=True)
        #st.write(f"Debug: Final result count = {len(all_results)}")
        return all_results

    except Exception as e:
        st.error(f"Error during search: {str(e)}")
        return []
