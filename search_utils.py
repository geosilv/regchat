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
        results_per_namespace = 0
    elif complexity_score >= 3:
        max_results = 12   # Medium complexity
        results_per_namespace = 0
    else:
        max_results = 7   # Low complexity
        results_per_namespace = 0
    
    return max_results, results_per_namespace

def search_regulations(query: str, index, model) -> list:
    """Main search function that coordinates the search process."""
    try:
        # Check index status
        try:
            stats = index.describe_index_stats()
            total_vectors = sum(ns.vector_count for ns in stats.namespaces.values())
            
            if total_vectors == 0:
                st.error("Index appears to be empty")
                return []
                
        except Exception as e:
            st.error(f"Error checking index stats: {str(e)}")
            return []
        
        # Get search parameters
        complexity_score = Complexity.calculate_complexity_score(query)
        max_results, results_per_namespace = determine_search_params(complexity_score)
        
        # Generate embedding
        query_embedding = generate_embedding(query, model)
        if not query_embedding or len(query_embedding) != 768:
            st.error("Invalid embedding generated")
            return []

                # Find exact word matches for namespaces
        def find_word_matches(text, target):
            """Find whole word matches, handling whitespace, punctuation, and case"""
            import re
            # Convert both text and target to lowercase for case-insensitive comparison
            text = text.lower()
            target = target.lower()
            # Create a pattern that matches the whole word/phrase
            pattern = r'\b' + re.escape(target) + r'\b'
            return bool(re.search(pattern, text))

        # Define namespaces and their variations
        namespace_docs = {
            "ECB_GIM_Feb24": {
                "display_name": "ECB GIM Feb 2024",
                "variations": ["ecb gim", "gim feb 2024", "ecb guide", "ecb"]
            },
            "ECB_TRIM2017": {
                "display_name": "ECB TRIM 2017",
                "variations": ["ecb trim", "trim 2017", "trim guide", "trim"]
            },
            "PRA_ss123": {
                "display_name": "PRA SS1/23",
                "variations": ["pra", "ss1/23", "ss1 23", "pra ss", "ss1"]
            },
            "JFSA_2021": {
                "display_name": "JFSA 2021",
                "variations": ["jfsa", "japan fsa", "fsa"]
            },
            "FED_sr1107a1": {
                "display_name": "FED SR 11-7a1",
                "variations": ["fed", "sr 11-7", "sr11-7", "fed sr", "sr11"]
            }
        }
        
        # Determine which namespaces to search based on query content
        namespaces_to_search = set()
        
        for namespace, info in namespace_docs.items():
            # Check display name
            if find_word_matches(query, info["display_name"]):
                namespaces_to_search.add(namespace)
                continue
            
            # Check variations
            for variation in info["variations"]:
                if find_word_matches(query, variation):
                    namespaces_to_search.add(namespace)
                    break
        
        # If no specific namespaces mentioned, search all
        if not namespaces_to_search:
            namespaces_to_search = set(namespace_docs.keys())
            st.info("No specific regulatory document mentioned - searching all documents.")
        else:
            selected_docs = [namespace_docs[ns]["display_name"] for ns in namespaces_to_search]
            st.info(f"Searching specifically in: {', '.join(selected_docs)}")
        
        all_matches = []
        used_ids = set()
        
        # First pass: Collect results from specified namespaces
        for namespace in namespaces_to_search:
            try:
                # Query with a higher number of results to ensure we get all relevant matches
                results = index.query(
                    vector=query_embedding,
                    top_k=20,  # Increased to get more potential matches
                    include_metadata=True,
                    namespace=namespace,
                    include_values=False
                )
                
                if results.matches:
                    for match in results.matches:
                        if match.id not in used_ids:
                            match.metadata['namespace'] = namespace  # Ensure namespace is in metadata
                            all_matches.append(match)
                            used_ids.add(match.id)
                            
            except Exception as e:
                st.error(f"Error querying namespace {namespace}: {str(e)}")
                continue
        
        # Sort all results by score
        all_matches.sort(key=lambda x: x.score, reverse=True)
        
        # If results_per_namespace is set, ensure minimum results from each namespace
        if results_per_namespace > 0:
            final_results = []
            namespace_counts = {ns: 0 for ns in namespaces_to_search}
            
            # First, add minimum required results from each namespace
            for match in all_matches:
                namespace = match.metadata.get('namespace')
                if namespace in namespace_counts and namespace_counts[namespace] < results_per_namespace:
                    final_results.append(match)
                    namespace_counts[namespace] += 1
            
            # Then add remaining top results up to max_results
            remaining_slots = max_results - len(final_results)
            if remaining_slots > 0:
                for match in all_matches:
                    if match not in final_results and len(final_results) < max_results:
                        final_results.append(match)
            
            return final_results[:max_results]
        
        # If no minimum results per namespace required, simply return top matches
        return all_matches[:max_results]

    except Exception as e:
        st.error(f"Error during search: {str(e)}")
        return []