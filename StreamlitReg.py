import streamlit as st
import pinecone
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import re
import json
import Complexity
from search_utils import search_regulations, generate_embedding



st.set_page_config(page_title="Banking Regulations Chatbot", page_icon="📚", layout="wide")



# Load secrets from Streamlit's Secrets Management
secrets = Complexity.load_secrets()
PINECONE_API_KEY = secrets["PINECONE_API_KEY"]
INDEX_HOST = secrets["INDEX_HOST"]
OPENAI_API_KEY = secrets["OPENAI_API_KEY"]


# Fixed list of namespaces
NAMESPACES = ["FED_SR117", "ECB_GIM_Feb24", "PRA_ss123", "ECB_TRIM2017"]

@st.cache_resource
def initialize_model():
    return SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

@st.cache_resource
def initialize_pinecone():
    try:
        pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index(host=INDEX_HOST)
        return index
    except Exception as e:
        st.error(f"Error initializing Pinecone: {str(e)}")
        return None

@st.cache_resource
def initialize_openai():
    return OpenAI(api_key=OPENAI_API_KEY)


def create_synthetic_gpt_prompt(query: str, context: list) -> str:
    prompt = (
        "You are a helpful assistant specializing in model risk management regulations. "
        "Your task is to:\n"
        "1. SYNTHESIZE information from ALL provided regulatory documents\n"
        "2. COMPARE and CONTRAST different regulatory perspectives when available\n"
        "3. HIGHLIGHT any differences or complementary views between documents\n"
        "4. CREATE a comprehensive answer that integrates insights from all relevant sources\n"
        "5. ALWAYS cite specific documents, and page numbers for each key point\n\n"
        "If only one document provides relevant information, explicitly state this and explain "
        "what aspects other regulations might not cover.\n\n"
    )
    
    # Group context by document for better synthesis
    docs_context = {}
    for item in context:
        doc_name = item.metadata.get('document_name', 'N/A')
        if doc_name not in docs_context:
            docs_context[doc_name] = []
        docs_context[doc_name].append(item)
    
    prompt += "Available Regulatory Context:\n"
    
    # Present context grouped by document
    for doc_name, items in docs_context.items():
        prompt += f"\nFrom {doc_name}:\n"
        for item in items:
            metadata = item.metadata
            section = metadata.get('section_name', 'N/A')
            page = metadata.get('page_start', 'N/A')
            prompt += f"- [Section: {section}, p. {page}]\n{metadata['text']}\n"
    
    prompt += f"\nQuestion: {query}\n\n"
    prompt += "Provide a comprehensive answer that synthesizes all relevant regulatory perspectives:"
    
    return prompt

def get_gpt_response(client, prompt: str) -> str:
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": (
                    "You are a regulatory expert specializing in model risk management. "
                    "Your role is to synthesize information across different regulatory "
                    "documents, compare perspectives, and provide comprehensive answers "
                    "that integrate insights from all relevant sources. Always cite sources "
                    "specifically and highlight any differences or complementary views "
                    "between different regulations."
                )},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=1500
        )

        raw_response = response.choices[0].message.content
        # Remove .0 from numbers and ensure consistent page number format
        fixed_response = re.sub(r'\b(\d+)\.0\b', r'\1', raw_response)
        fixed_response = re.sub(r'Page:?\s*(\d+)', r'p. \1', fixed_response)
        fixed_response = re.sub(r'page:?\s*(\d+)', r'p. \1', fixed_response)
        return fixed_response
    
    except Exception as e:
        return f"Error generating response: {str(e)}"

def setup_page_header():
    st.markdown(
        """
        <h1 style='font-size:24px; color:black; margin-bottom:0;'>
        Banking Regulations Chatbot: Model Risk v 0.2
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
        document_links = {
            "FED_SR117": "https://www.federalreserve.gov/supervisionreg/srletters/sr1107.pdf",
            "ECB_GIM_Feb24": "https://www.bankingsupervision.europa.eu/ecb/pub/pdf/ssm.supervisory_guides202402_internalmodels.en.pdf",
            "PRA_ss123": "https://www.bankofengland.co.uk/-/media/boe/files/prudential-regulation/supervisory-statement/2023/ss123.pdf",
            "ECB_TRIM2017": "https://www.bankingsupervision.europa.eu/ecb/pub/pdf/trim_guide.en.pdf"
        }
        for ns in NAMESPACES:
            st.markdown(f"- [{ns}]({document_links[ns]})")

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
    st.subheader("📚 References")
    
    if complexity_score >= 3:
        # Group references by document for medium/high complexity
        docs_refs = {}
        for match in results:
            doc_name = match.metadata.get('document_name', 'N/A')
            if doc_name not in docs_refs:
                docs_refs[doc_name] = []
            docs_refs[doc_name].append(match)
        
        for doc_name, matches in docs_refs.items():
            with st.expander(f"**{doc_name}** ({len(matches)} references)", expanded=False):
                for i, match in enumerate(matches, 1):
                    st.markdown(f"### Reference {i} (Relevance: {match.score:.2f})")
                    _display_reference_details(match)
    else:
        # Simple list for low complexity queries
        for i, match in enumerate(results, 1):
            with st.expander(f"**Source {i}** (Relevance: {match.score:.2f})", expanded=False):
                _display_reference_details(match)

def _display_reference_details(match):
    metadata = match.metadata
    st.markdown(f"**Document:** {metadata.get('document_name', 'N/A')}")
    st.markdown(f"**Section:** {metadata.get('section_name', 'N/A')}")
    
    page_number = metadata.get('page_start', 'N/A')
    try:
        if isinstance(page_number, (int, float)):
            page_number = int(page_number)
    except (ValueError, TypeError):
        page_number = 'N/A'
    
    st.markdown(f"**Page:** {page_number}")
    st.markdown("**Relevant Text:**")
    st.markdown(f"> {metadata.get('text', 'No text available')}")

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