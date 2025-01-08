import streamlit as st
import pinecone
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import re
import json

st.set_page_config(page_title="Banking Regulations Chatbot: Model Risk", page_icon="📚", layout="wide")

# Load secrets from a file
'''
with open('secrets.json', 'r') as f:
    secrets = json.load(f)

PINECONE_API_KEY = secrets['PINECONE_API_KEY']
INDEX_HOST = secrets['INDEX_HOST']
OPENAI_API_KEY = secrets['OPENAI_API_KEY']
'''


# Load secrets from Streamlit's Secrets Management
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
INDEX_HOST = st.secrets["INDEX_HOST"]
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]


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

def generate_embedding(text: str, model) -> list:
    try:
        embedding = model.encode(text)
        return embedding.tolist()
    except Exception as e:
        st.error(f"Error generating embedding: {str(e)}")
        return []

def search_regulations(query: str, index, model) -> list:
    try:
        query_embedding = generate_embedding(query, model)
        
        if not query_embedding or len(query_embedding) != 768:
            st.error("Invalid embedding generated")
            return []
            
        namespace_docs = {
            "FED_SR117": "sr1107a1.pdf",
            "ECB_GIM_Feb24": "ECB_Guidelines.pdf",
            "PRA_ss123": "PRA_SS123.pdf",
            "ECB_TRIM2017": "ECB_TRIM2017.pdf"  # Fixed name to match NAMESPACES
        }
        
        # First pass: Get exactly one result from each namespace
        all_results = []
        namespace_results = {}  # Track results per namespace
        
        # Force query each namespace separately
        for namespace in namespace_docs.keys():
            try:
                results = index.query(
                    vector=query_embedding,
                    top_k=5,
                    include_metadata=True,
                    namespace=namespace
                )
                
                if results.matches:
                    namespace_results[namespace] = results.matches[0]
                    all_results.append(results.matches[0])
            except Exception as e:
                st.warning(f"Error querying namespace {namespace}: {str(e)}")
                continue
        
        # Fill remaining slots if needed
        if len(all_results) < 5:
            remaining_slots = 5 - len(all_results)
            used_ids = {r.id for r in all_results}
            
            for namespace in namespace_docs.keys():
                if len(all_results) >= 5:
                    break
                    
                try:
                    results = index.query(
                        vector=query_embedding,
                        top_k=remaining_slots + 5,
                        include_metadata=True,
                        namespace=namespace
                    )
                    
                    for match in results.matches:
                        if match.id not in used_ids and len(all_results) < 5:
                            all_results.append(match)
                            used_ids.add(match.id)
                            
                except Exception as e:
                    continue
        
        # Sort final results by score
        all_results.sort(key=lambda x: x.score, reverse=True)
        return all_results[:5]
            
    except Exception as e:
        st.error(f"Error during search: {str(e)}")
        return []

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

def main():
 #   st.set_page_config(page_title="Banking Regulations Chatbot: Model Risk", page_icon="📚", layout="wide")
    st.markdown(
    """
    <h1 style='font-size:24px; color:black; margin-bottom:0;'>
    Banking Regulations Chatbot: Model Risk
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
 #   st.subheader("A tool for synthesizing and comparing regulatory insights on model risk management.")
    # Initialize components
    try:
        model = initialize_model()
        index = initialize_pinecone()
        client = initialize_openai()
        
        if not index:
            st.error("Failed to initialize Pinecone index")
            return
    except Exception as e:
        st.error(f"Error initializing components: {str(e)}")
        return

    # Display title and sidebar
    #st.title("Banking Regulations Chatbot: Model Risk")
    
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
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat interface
    if prompt := st.chat_input("What would you like to know about model risk regulations?"):
        # Display user message
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Generate response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
            with st.spinner("Searching regulations..."):
                results = search_regulations(prompt, index, model)
            
            if results:
                gpt_prompt = create_synthetic_gpt_prompt(prompt, results)
                
                with st.spinner("Generating response..."):
                    gpt_response = get_gpt_response(client, gpt_prompt)
                
                message_placeholder.markdown(gpt_response)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": gpt_response
                })

                # Display references with dropdowns for each reference
                st.subheader("📚 References")
                for i, match in enumerate(results, 1):
                    metadata = match.metadata
                    reference_title = f"**Source {i}** (Relevance: {match.score:.2f})"
                    
                    # Each reference as a separate expander
                    with st.expander(reference_title, expanded=False):
                        st.markdown(f"**Document:** {metadata.get('document_name', 'N/A')}")
                        st.markdown(f"**Section:** {metadata.get('section_name', 'N/A')}")
                        
                        # Handle page number conversion properly
                        page_number = metadata.get('page_start', 'N/A')
                        try:
                            if isinstance(page_number, (int, float)):
                                page_number = int(page_number)
                        except (ValueError, TypeError):
                            page_number = 'N/A'
                        
                        st.markdown(f"**Page:** {page_number}")
                        st.markdown("**Relevant Text:**")
                        st.markdown(f"> {metadata.get('text', 'No text available')}")
            else:
                message_placeholder.markdown("I couldn't find any relevant regulations. Could you please rephrase your question?")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "I couldn't find any relevant regulations. Could you please rephrase your question?"
                })

if __name__ == "__main__":
    main()