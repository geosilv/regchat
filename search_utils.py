import streamlit as st
from sentence_transformers import SentenceTransformer
import Complexity

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
        max_results = 12  # High complexity
        results_per_namespace = 3
    elif complexity_score >= 3:
        max_results = 8   # Medium complexity
        results_per_namespace = 2
    else:
        max_results = 5   # Low complexity
        results_per_namespace = 1
    
    return max_results, results_per_namespace

def get_namespace_results(query_embedding: list, namespace: str, results_per_namespace: int, 
                         index, max_results: int, used_ids: set) -> tuple:
    """Query a single namespace and return results."""
    try:
        results = index.query(
            vector=query_embedding,
            top_k=results_per_namespace,
            include_metadata=True,
            namespace=namespace
        )
        
        namespace_matches = []
        if results.matches:
            for match in results.matches[:results_per_namespace]:
                if len(namespace_matches) < max_results and match.id not in used_ids:
                    namespace_matches.append(match)
                    used_ids.add(match.id)
        
        return namespace_matches, used_ids
        
    except Exception as e:
        st.warning(f"Error querying namespace {namespace}: {str(e)}")
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
        # Initialize search parameters
        complexity_score = Complexity.calculate_complexity_score(query)
        max_results, results_per_namespace = determine_search_params(complexity_score)
        
        # Generate embedding
        query_embedding = generate_embedding(query, model)
        if not query_embedding or len(query_embedding) != 768:
            st.error("Invalid embedding generated")
            return []
        
        # Define namespaces
        namespace_docs = {
            "FED_SR117": "sr1107a1.pdf",
            "ECB_GIM_Feb24": "ECB_Guidelines.pdf",
            "PRA_ss123": "PRA_SS123.pdf",
            "ECB_TRIM2017": "ECB_TRIM2017.pdf"
        }
        
        # First pass: Get results from each namespace
        all_results = []
        used_ids = set()
        
        for namespace in namespace_docs.keys():
            namespace_matches, used_ids = get_namespace_results(
                query_embedding, namespace, results_per_namespace, 
                index, max_results, used_ids
            )
            all_results.extend(namespace_matches)
            
            if len(all_results) >= max_results:
                break
        
        # Second pass: Fill remaining slots if needed
        if len(all_results) < max_results:
            remaining_slots = max_results - len(all_results)
            
            for namespace in namespace_docs.keys():
                if len(all_results) >= max_results:
                    break
                    
                additional_matches, used_ids = fill_remaining_slots(
                    query_embedding, namespace, remaining_slots, 
                    index, used_ids
                )
                all_results.extend(additional_matches)
        
        # Sort and return results
        all_results.sort(key=lambda x: x.score, reverse=True)
        return all_results

    except Exception as e:
        st.error(f"Error during search: {str(e)}")
        return []