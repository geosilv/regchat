# First: Imports
import streamlit as st
import pinecone
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import re
import json
import Complexity
from search_utils import search_regulations, generate_embedding
import search_utils
import utils
import tiktoken

# Second: Page Configuration
st.set_page_config(page_title="Banking Regulations Chatbot", page_icon="📚", layout="wide")

# Third: Constants and Configuration
secrets = Complexity.load_secrets()
PINECONE_API_KEY = secrets["PINECONE_API_KEY"]
INDEX_HOST = secrets["INDEX_HOST"]
INDEX_NAME = secrets["INDEX_NAME"]
OPENAI_API_KEY = secrets["OPENAI_API_KEY"]

NAMESPACES = [
    "ECB_GIM_Feb24", 
    "ECB_TRIM2017", 
    "PRA_ss123", 
    "FED_sr1107a1",
    "JFSA_2021"              
]

# Fourth: Utility Functions
def get_friendly_document_name(doc_name: str) -> str:
    """
    Convert processed filenames to friendly document names.
    Handles both .json extensions and base filenames.
    """
    # Remove .json extension if present
    doc_name = doc_name.replace('.json', '')
    
    # # Add processed_ prefix if not present
    # if not doc_name.startswith('processed_'):
    #     doc_name = f'processed_{doc_name}'
    
    # # Add _processed suffix if not present
    # if not doc_name.endswith('_processed'):
    #     doc_name = f'{doc_name}_processed'
    
    document_mapping = {
        "ECB_GIM_Feb24": "ECB GIM 2024",
        "ECB_TRIM2017": "ECB TRIM 2017",
        "PRA_ss123": "PRA SS1/23",
        "JFSA_2021": "JFSA 2021",
        "FED_sr1107a1": "FED SR 11-7"
    }
    return document_mapping.get(doc_name, doc_name)

# Fifth: Initialization Functions
@st.cache_resource
def initialize_model():
    return SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

@st.cache_resource
def initialize_pinecone():
    try:
        pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
        try:
            index = pc.Index(INDEX_NAME)
            stats = index.describe_index_stats()
            return index
        except Exception as e:
            st.error(f"Error accessing index '{INDEX_NAME}': {str(e)}")
            return None
    except Exception as e:
        st.error(f"Error initializing Pinecone: {str(e)}")
        return None

@st.cache_resource
def initialize_openai():
    return OpenAI(api_key=OPENAI_API_KEY)

# Sixth: Core Processing Functions
def create_synthetic_gpt_prompt(query: str, context: list) -> str:
    prompt = (
        "You are a helpful assistant specializing in model risk management regulations. "
        "Your task is to:\n"
        "1. SYNTHESIZE information from ALL provided regulatory documents\n"
        "2. COMPARE and CONTRAST different regulatory perspectives when available\n"
        "3. HIGHLIGHT any differences or complementary views between documents\n"
        "4. CREATE a comprehensive answer that integrates insights from all relevant sources\n"
        "5. ALWAYS cite specific documents and page numbers for each key point\n\n"
        "6. CHECK IF the question asked makes logical, grammatical, semantic AND contextual sense IF NOT THEN say so and do not reply.\n\n"
        "When citing sources, write citations in running text like this: (ECB GIM 2024, p. 7)\n"
        "Do not put citations at the end of paragraphs or in a separate section.\n\n"
        "If you are asked to summarize or be brief or concise or similar, aim for shorter and informative responses\n\n"    
        "If only one document provides relevant information, explicitly state this and explain\n\n"
        "what aspects other regulations might not cover.\n\n"
    )
    
    docs_context = {}
    for item in context:
        doc_name = get_friendly_document_name(item.metadata.get('document_title', 'N/A'))
        if doc_name not in docs_context:
            docs_context[doc_name] = []
        docs_context[doc_name].append(item)
    
    prompt += "Available Regulatory Context:\n"
    for doc_name, items in docs_context.items():
        prompt += f"\nFrom {doc_name}:\n"
        for item in items:
            metadata = item.metadata
            section = metadata.get('section_name', 'N/A')
            page = metadata.get('page_start', 'N/A')
            if metadata.get('text'):
                prompt += f"- Section: {section}, Page: {page}\n{metadata['text']}\n"
            else:
                print(f"[DEBUG] Skipping item with missing text: {metadata}")
    
    prompt += "\nProvide a comprehensive answer that synthesizes all relevant regulatory perspectives:"
    return prompt

def get_gpt_response(client, prompt: str) -> str:
    document_mapping = {
        "ECB_GIM_Feb24": "ECB GIM 2024",
        "ECB_TRIM2017": "ECB TRIM 2017",
        "PRA_ss123": "PRA SS1/23",
        "JFSA_2021": "JFSA 2021",
        "FED_sr1107a1": "FED SR 11-7"
    }
    
    MAX_TOKENS = 800
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": (
                    "You are a regulatory expert specializing in model risk management. "
                    "When referencing documents, integrate citations naturally into your text "
                    "using this exact format: (Document Name, p. X). For example: 'According "
                    "to guidance (ECB GIM 2024, p. 5), model risk management should...' "
                    "Your role is to synthesize information across different regulatory "
                    "documents, compare perspectives, and provide comprehensive answers "
                    "that integrate insights from all relevant sources. Always weave citations "
                    "naturally into your sentences. Never use underscores or 'processed' in document names."
                    "If the question asked does not make logical or grammatical or semantic sense, say so and do not reply. "
                    "If you are asked to summarize or be brief or concise, aim for clear and informative responses. "
                    "Provide answers with the limit of {MAX_TOKENS} tokens."
                )},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=MAX_TOKENS
        )

        raw_response = response.choices[0].message.content
        
        # Fix floating point page numbers
        fixed_response = re.sub(r'\b(\d+)\.0\b', r'\1', raw_response)
        
        # Standardize page number format
        fixed_response = re.sub(r'Page:?\s*(\d+)', r'p. \1', fixed_response)
        fixed_response = re.sub(r'page:?\s*(\d+)', r'p. \1', fixed_response)
        
        # # First pass: Handle processed names in citations
        # for old_name, friendly_name in document_mapping.items():
        #     old_pattern = rf'\((?:processed_)?{re.escape(old_name.replace("processed_", "").replace("_processed", ""))}(?:_processed)?(?:\.json)?,\s*p\.\s*(\d+)\)'
        #     fixed_response = re.sub(old_pattern, rf'({friendly_name}, p. \1)', fixed_response)
        
        # Second pass: Clean up any remaining non-standard citation formats
        fixed_response = re.sub(r'\[([^,]+?),\s*p\.\s*(\d+)\]', r'(\1, p. \2)', fixed_response)
        fixed_response = re.sub(r'\{([^,]+?),\s*p\.\s*(\d+)\}', r'(\1, p. \2)', fixed_response)
        
        # Final pass: Ensure consistent spacing in citations
        fixed_response = re.sub(r'\(([^,]+?)\s*,\s*p\.\s*(\d+)\)', r'(\1, p. \2)', fixed_response)
            
        return fixed_response
    
    except Exception as e:
        return f"Error generating response: {str(e)}"

# Seventh: UI Functions
def setup_page_header():
    st.markdown(
        """
        <h1 style='font-size:24px; color:black; margin-bottom:0;'>
        Banking Regulations Chatbot: Model Risk v 0.21
        </h1>
        """, 
        unsafe_allow_html=True
    )
    st.markdown(
        """
        <p style='font-size:16px; color:gray;'>
        A tool for synthesizing and comparing regulatory insights on model risk management.
        </p>
        """, 
        unsafe_allow_html=True
    )

def setup_sidebar():
    with st.sidebar:
        st.title("Available Documents")
        display_names = {
            "ECB_GIM_Feb24": "ECB GIM Feb 2024",
            "ECB_TRIM2017": "ECB TRIM 2017",
            "PRA_ss123": "PRA SS1/23",
            "JFSA_2021": "JFSA 2021",
            "FED_sr1107a1": "FED SR 11-7a1"
        }
        
        document_links = {
            "ECB_GIM_Feb24": "https://www.bankingsupervision.europa.eu/ecb/pub/pdf/ssm.supervisory_guides202402_internalmodels.en.pdf",
            "ECB_TRIM2017": "https://www.bankingsupervision.europa.eu/ecb/pub/pdf/trim_guide.en.pdf",
            "PRA_ss123": "https://www.bankofengland.co.uk/-/media/boe/files/prudential-regulation/supervisory-statement/2023/ss123.pdf",
            "JFSA_2021": "https://www.fsa.go.jp/en/news/2021/20210730-1.html",
            "FED_sr1107a1": "https://www.federalreserve.gov/supervisionreg/srletters/sr1107.pdf"
        }
        
        for ns in NAMESPACES:
            if ns in document_links:
                display_name = display_names.get(ns, ns)
                st.markdown(f"- [{display_name}]({document_links[ns]})")
            else:
                st.warning(f"No link found for namespace: {ns}")

def initialize_components():
    try:
        model = initialize_model()
        index = initialize_pinecone()
        client = initialize_openai()
        
        if not index:
            st.error("Failed to initialize Pinecone index")
            return None, None, None
        return model, index, client
    except Exception as e:
        st.error(f"Error initializing components: {str(e)}")
        return None, None, None

def display_chat_history():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

def display_references_by_complexity(results, complexity_score):
    """Display references with friendly document names based on complexity score."""
    st.subheader("📚 References")
    
    # Create a list to store processed results
    processed_results = []
    for match in results:
        # Get original and friendly names
        doc_title = match.metadata.get('document_title', 'N/A')
        friendly_name = get_friendly_document_name(doc_title)
        
        # Store both names in metadata
        new_match = match
        new_match.metadata['original_title'] = doc_title
        new_match.metadata['friendly_name'] = friendly_name
        processed_results.append(new_match)
        
        # Debug log
        #print(f"Converting: {doc_title} -> {friendly_name}")
    
    if complexity_score >= 3:
        # Group by friendly document name
        docs_refs = {}
        for match in processed_results:
            friendly_name = match.metadata['friendly_name']
            if friendly_name not in docs_refs:
                docs_refs[friendly_name] = []
            docs_refs[friendly_name].append(match)
        
        for friendly_name, matches in docs_refs.items():
            with st.expander(f"{friendly_name} ({len(matches)} references)", expanded=False):
                for i, match in enumerate(matches, 1):
                    st.markdown(f"### Reference {i} (Relevance: {match.score:.2f})")
                    _display_reference_details(match)
    else:
        for i, match in enumerate(processed_results, 1):
            friendly_name = match.metadata['friendly_name']
            
            # Debug log
            #print(f"Displaying reference {i}: {friendly_name}")
            
            with st.expander(f"{friendly_name} - Reference {i} (Relevance: {match.score:.2f})", expanded=False):
                _display_reference_details(match)

def _display_reference_details(match):
    """Display details for a single reference using friendly document name."""
    # Get the friendly name we stored earlier
    friendly_name = match.metadata['friendly_name']
    
    # Debug log
    #print(f"Displaying details for: {friendly_name}")
    
    st.markdown(f"**Document:** {friendly_name}")
    
    section_name = match.metadata.get('section_name', 'N/A')
    st.markdown(f"**Section:** {section_name}")
    
    page_number = match.metadata.get('page_start', 'N/A')
    try:
        if isinstance(page_number, (int, float)):
            page_number = int(page_number)
    except (ValueError, TypeError):
        page_number = 'N/A'
    
    st.markdown(f"**Page:** {page_number}")
    
    if text := match.metadata.get('text'):
        st.markdown("**Relevant Text:**")
        st.markdown(f"> {text}")



def process_user_query(prompt, model, index, client):
    results = search_regulations(prompt, index, model)
    complexity_score = Complexity.calculate_complexity_score(prompt)
    
    if results:
        gpt_prompt = create_synthetic_gpt_prompt(prompt, results)
        gpt_response = get_gpt_response(client, gpt_prompt)
        return results, complexity_score, gpt_response
    return results, complexity_score, None


def main():
    setup_page_header()
    model, index, client = initialize_components()
    
    if not all([model, index, client]):
        return
        
    setup_sidebar()
    display_chat_history()
    
    if prompt := st.chat_input("What would you like to know about model risk regulations?"):
        # Display user message
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Generate response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
            with st.spinner("Searching regulations..."):
                results, complexity_score, gpt_response = process_user_query(prompt, model, index, client)
                
            st.markdown(Complexity.format_complexity_display(complexity_score))
        
            if results and gpt_response:
                message_placeholder.markdown(gpt_response)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": gpt_response
                })
                
                display_references_by_complexity(results, complexity_score)
            else:
                message_placeholder.markdown("I couldn't find any relevant regulations. Could you please rephrase your question?")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "I couldn't find any relevant regulations. Could you please rephrase your question?"
                })


if __name__ == "__main__":
    main()