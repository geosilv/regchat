import os
import json
import re
import Complexity
import search_utils
import streamlit as st




def enhance_query_with_summary(prompt, summaries):
    """
    Check if the prompt relates to any available document and enhance it with summary information.
    """
    # List of document identifiers to check against
    doc_identifiers = {
        "SR117": ["SR117", "SR-11-7", "SR 11-7", "FED_SR117"],
        "ECB_GIM": ["ECB_GIM", "GIM", "ECB GIM", "ECB_GIM_Feb24"],
        "SS123": ["SS123", "SS-123", "SS 123", "PRA_ss123"],
        "TRIM": ["TRIM", "ECB TRIM", "ECB_TRIM2017", "TRIM2017"]
    }
    
    # Convert prompt to lowercase for case-insensitive matching
    prompt_lower = prompt.lower()
    
    enhanced_prompt = prompt
    
    for base_id, variations in doc_identifiers.items():
        # Check if any variation of the document identifier is in the prompt
        if any(var.lower() in prompt_lower for var in variations):
            # Try to find the corresponding summary
            summary = None
            for var in variations:
                if var in summaries:
                    summary = summaries[var]
                    break
            
            if summary and 'summary' in summary:
                # Append the full summary to the prompt
                enhanced_prompt = f"{prompt}\n\nContext from document summary:\n{summary['summary']['full_summary']}"
                break  # Only append one summary even if multiple matches
    
    return enhanced_prompt



from datetime import datetime
import os

def save_gpt_prompt(gpt_prompt: str, base_dir: str = "prompts") -> str:
    """
    Saves the GPT prompt to a file with a timestamp in the filename.
    
    Args:
        gpt_prompt (str): The GPT prompt to save
        base_dir (str): The directory where prompts will be saved. Defaults to "prompts"
    
    Returns:
        str: The path to the saved file
    """
    # Create the directory if it doesn't exist
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create filename with timestamp
    filename = f"gpt_prompt_{timestamp}.txt"
    filepath = os.path.join(base_dir, filename)
    
    # Write the prompt to file
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(gpt_prompt)
        return filepath
    except Exception as e:
        print(f"Error saving prompt to file: {str(e)}")
        return None


def save_gpt_prompt2(gpt_prompt: str, base_dir: str = "prompts") -> str:
    """
    Saves the GPT prompt to a single file, overwriting it each time.

    Args:
        gpt_prompt (str): The GPT prompt to save
        base_dir (str): The directory where the prompt will be saved. Defaults to "prompts"

    Returns:
        str: The path to the saved file
    """
    # Create the directory if it doesn't exist
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    # Use a fixed filename
    filename = "gpt_prompt.txt"
    filepath = os.path.join(base_dir, filename)

    # Write the prompt to the file, overwriting any existing content
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(gpt_prompt)
        return filepath
    except Exception as e:
        print(f"Error saving prompt to file: {str(e)}")
        return None

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

   
def create_synthetic_gpt_prompt(query: str, context: list) -> str:
    # Extract any summaries from the query (they follow "Context from" markers)
    summary_sections = []
    query_lines = query.split('\n')
    clean_query = []
    current_summary = []
    in_summary = False
    
    for line in query_lines:
        line = line.strip()  # Handle any extra whitespace
        if not line:  # Skip empty lines but maintain summary collection
            if not in_summary:
                clean_query.append(line)
            continue
            
        if line.startswith("Context from"):
            if current_summary:
                summary_sections.append('\n'.join(current_summary))
                current_summary = []
            current_summary = [line]
            in_summary = True
        elif in_summary:
            current_summary.append(line)
        else:
            clean_query.append(line)
    
    # Don't forget the last summary if there is one
    if current_summary:
        summary_sections.append('\n'.join(current_summary))
        
    # Start with the original prompt structure
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
        
    # Add summary sections if they exist
    if summary_sections:
        prompt += "Document Summaries:\n"
        for summary in summary_sections:
            prompt += f"{summary}\n\n"
            
    # Add the search results
    prompt += "Available Regulatory Context:\n"
    
    # Group context by document for better synthesis
    docs_context = {}
    for item in context:
        doc_name = item.metadata.get('document_name', 'N/A')
        if doc_name not in docs_context:
            docs_context[doc_name] = []
        docs_context[doc_name].append(item)
    
    # Present context grouped by document
    for doc_name, items in docs_context.items():
        prompt += f"\nFrom {doc_name}:\n"
        for item in items:
            metadata = item.metadata
            section = metadata.get('section_name', 'N/A')
            page = metadata.get('page_start', 'N/A')
            prompt += f"- [Section: {section}, p. {page}]\n{metadata.get('text', '')}\n"
    
    # Clean and join the query lines, ensuring there's actual content
    clean_query_text = ' '.join(line for line in clean_query if line.strip())
    if not clean_query_text:  # If no clean query text, use the original query
        clean_query_text = query.strip()
    
    prompt += f"\nQuestion: {clean_query_text}\n\n"
    prompt += "Provide a comprehensive answer that synthesizes all relevant regulatory perspectives:"
    save_gpt_prompt(prompt)

    save_gpt_prompt2(prompt, "prompts")

    return prompt



def get_friendly_document_name(doc_name: str) -> str:
    """
    Convert processed filenames to friendly document names.
    """
    document_mapping = {
        "processed_ECB_GIM_Feb24_processed": "ECB GIM 2024",
        "processed_ECB_TRIM2017_processed": "ECB TRIM 2017",
        "processed_PRA_ss123_processed": "PRA SS1/23",
        "processed_JFSA_2021_processed": "JFSA 2021",
        "processed_FED_sr1107a1_processed": "FED SR 11-7"
    }
    return document_mapping.get(doc_name, doc_name)


def display_sidebar_contact():
    """
    Displays clickable contact information near the bottom of the sidebar.
    """
    # Reduced number of breaks to move contact info higher
    st.markdown("<br>" *6, unsafe_allow_html=True)
    
    # Add contact info with mailto link
    st.markdown(
        """
        <div style='color: #808080; font-size: 14px; padding: 10px 0;'>
        Contact: <a href="mailto:georgios.argiris@gmail.com" style="color: #808080; text-decoration: none;">georgios.argiris@gmail.com</a>
        </div>
        """,
        unsafe_allow_html=True
    )

def setup_about_section():
    """
    Creates a discrete About section with proper state management
    """
    # Initialize the session state for about section visibility if it doesn't exist
    st.markdown("<br>" * 7  , unsafe_allow_html=True)
    if 'show_about' not in st.session_state:
        st.session_state.show_about = False

    # Create a small button that toggles the about section
    if st.button('ℹ️ About', use_container_width=False):
        st.session_state.show_about = not st.session_state.show_about

    # Show about content only if state is True
    if st.session_state.show_about:
        with st.container():
            st.markdown(
                """
                <div style='background-color: #f8f9fa; padding: 20px; border-radius: 5px; margin: 10px 0;'>
                <h3 style='font-size: 18px; margin-bottom: 10px;'>About this Application</h3>
                <p style='font-size: 16px; line-height: 1.5;'>
                This chatbot provides an intelligent interface to banking regulations focused on model risk management. 
                It synthesizes information from multiple regulatory documents including ECB, PRA, FED, and JFSA guidelines. 
                The tool uses advanced natural language processing to understand queries and provide relevant, accurate responses 
                with proper citations to source documents.
                </p>
                </div>
                """,
                unsafe_allow_html=True
            )