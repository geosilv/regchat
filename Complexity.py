import streamlit as st
import os
import json

def calculate_complexity_score(query):
    score = 0
    
    # Add points for query length
    score += len(query.split()) // 5  # 1 point for every 5 words
    
    # Add points for specific keywords
    keywords = ["compare", "cross-jurisdictional", "audit", "framework", "guidelines"]
    score += sum(1 for word in query.split() if word.lower() in keywords)
    
    # Add points for references to multiple concepts
    if "and" in query or "," in query:
        score += 2
    
    # Adjust points based on question type
    if query.lower().startswith(("how", "why")):
        score += 3
    
    return score

def determine_top_k(score):
    if score < 3:
        return 5  # Low complexity, fewer chunks
    elif score < 6:
        return 10  # Medium complexity
    else:
        return 15  # High complexity, more chunks

def format_complexity_display(score: int) -> str:
    """
    Format complexity score with visual indicators and labels.
    """
    if score < 3:
        return "🟢 Low Complexity (Score: {})".format(score)
    elif score < 6:
        return "🟡 Medium Complexity (Score: {})".format(score)
    else:
        return "🔴 High Complexity (Score: {})".format(score)


def load_secrets():
    """Load secrets from Streamlit or local secrets.json"""
    try:
        # Try Streamlit secrets first
        return {
            "PINECONE_API_KEY": st.secrets["PINECONE_API_KEY"],
            "INDEX_HOST": st.secrets["INDEX_HOST"],
            "OPENAI_API_KEY": st.secrets["OPENAI_API_KEY"],
            "INDEX_NAME": st.secrets["INDEX_NAME"]
        }
    except:
        # Fall back to local secrets.json
        if os.path.exists("secrets.json"):
            with open("secrets.json", "r") as f:
                return json.load(f)
        raise Exception("No secrets found in Streamlit or secrets.json")
