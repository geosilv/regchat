from langchain_community.document_loaders import PyPDFLoader
from datetime import datetime
import spacy
import json
import os
import tiktoken
import time
import re
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential

# Configuration
PDF_DIRECTORY = r"C:\Users\gallo\source\VSCode\RegulationsProject\LanchainProcessedDocs"
OPENAI_API_KEY = "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"
TARGET_TOKENS = 200
MIN_TOKENS = 50
MAX_TOKENS = 300  # Added maximum token limit
OVERLAP_TOKENS = 15
MAX_COMBINE_TOKENS = 50

class PDFProcessor:
    def __init__(self, directory_path: str):
        """Initialize the PDF processor with the given directory path."""
        self.directory_path = directory_path
        self.output_dir = os.path.join(directory_path, "processed_chunks")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        
        # Initialize tokenizer
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        # Load spaCy model for sentence splitting
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Downloading spaCy model...")
            os.system("python -m spacy download en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")
        
        print("Initialization complete.")

    def get_token_count(self, text: str) -> int:
        """Count tokens in text using tiktoken."""
        return len(self.tokenizer.encode(text))

    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using spaCy."""
        doc = self.nlp(text)
        return [str(sent).strip() for sent in doc.sents]

    def create_chunks(self, text: str, page_num: int) -> List[Dict[str, Any]]:
        """Create chunks from text based on sentence boundaries and token parameters."""
        chunks = []
        sentences = self.split_into_sentences(text)
        
        current_chunk = []
        current_tokens = 0
        
        for sentence in sentences:
            sentence_tokens = self.get_token_count(sentence)
            
            # Handle very long sentences
            if sentence_tokens > MAX_TOKENS:
                # If we have accumulated sentences, save them as a chunk
                if current_chunk:
                    chunks.append({
                        "text": " ".join(current_chunk).strip(),
                        "token_count": current_tokens,
                        "page": page_num
                    })
                    current_chunk = []
                    current_tokens = 0
                
                # Split long sentence on punctuation or conjunctions
                sub_sentences = re.split(r'[,;:]', sentence)
                current_sub = []
                current_sub_tokens = 0
                
                for sub in sub_sentences:
                    sub = sub.strip()
                    sub_tokens = self.get_token_count(sub)
                    
                    if current_sub_tokens + sub_tokens <= TARGET_TOKENS:
                        current_sub.append(sub)
                        current_sub_tokens += sub_tokens
                    else:
                        if current_sub:
                            chunks.append({
                                "text": ", ".join(current_sub).strip(),
                                "token_count": current_sub_tokens,
                                "page": page_num
                            })
                        current_sub = [sub]
                        current_sub_tokens = sub_tokens
                
                if current_sub:
                    chunks.append({
                        "text": ", ".join(current_sub).strip(),
                        "token_count": current_sub_tokens,
                        "page": page_num
                    })
                continue
            
            # Normal sentence processing
            if current_tokens + sentence_tokens <= TARGET_TOKENS:
                current_chunk.append(sentence)
                current_tokens += sentence_tokens
            else:
                # Save current chunk if it meets minimum size
                if current_tokens >= MIN_TOKENS:
                    chunks.append({
                        "text": " ".join(current_chunk).strip(),
                        "token_count": current_tokens,
                        "page": page_num
                    })
                
                # Start new chunk with current sentence
                current_chunk = [sentence]
                current_tokens = sentence_tokens
        
        # Add remaining text if it meets minimum size
        if current_chunk and current_tokens >= MIN_TOKENS:
            chunks.append({
                "text": " ".join(current_chunk).strip(),
                "token_count": current_tokens,
                "page": page_num
            })
        
        return chunks

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(3))
    def summarize_with_chatgpt(self, text: str) -> str:
        """Generate a summary using ChatGPT API with retry logic."""
        try:
            prompt = f"""Please provide a brief, one-sentence summary of the following text. Focus on the key points and maintain technical accuracy:

{text}

Summary:"""

            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a precise summarizer. Provide clear, accurate, one-sentence summaries focusing on key regulatory and technical points."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=100
            )
            
            summary = response.choices[0].message.content.strip()
            time.sleep(0.5)
            return summary
            
        except Exception as e:
            print(f"Error in ChatGPT summarization: {e}")
            return text[:200] + "..."

    def combine_small_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Combine adjacent small chunks while preserving sentence boundaries."""
        if not chunks:
            return []
            
        combined = []
        i = 0
        
        while i < len(chunks):
            current_chunk = chunks[i]
            
            if i + 1 < len(chunks):
                next_chunk = chunks[i + 1]
                combined_tokens = current_chunk["token_count"] + next_chunk["token_count"]
                
                if combined_tokens <= MAX_COMBINE_TOKENS:
                    combined.append({
                        "text": current_chunk["text"] + " " + next_chunk["text"],
                        "token_count": combined_tokens,
                        "page": current_chunk["page"]
                    })
                    i += 2
                    continue
            
            combined.append(current_chunk)
            i += 1
        
        return combined

    def process_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Process a single PDF file."""
        try:
            loader = PyPDFLoader(pdf_path)
            pages = loader.load()
            
            all_chunks = []
            for page in tqdm(pages, desc="Processing pages"):
                chunks = self.create_chunks(page.page_content, page.metadata["page"])
                chunks = self.combine_small_chunks(chunks)
                
                for i, chunk in enumerate(chunks):
                    summary = self.summarize_with_chatgpt(chunk["text"])
                    
                    all_chunks.append({
                        "text": chunk["text"],
                        "metadata": {
                            "document_title": os.path.basename(pdf_path),
                            "page_number": chunk["page"],
                            "chunk_index": i,
                            "token_count": chunk["token_count"],
                            "processing_timestamp": datetime.utcnow().isoformat(),
                            "summary": summary
                        }
                    })
            
            return all_chunks
            
        except Exception as e:
            print(f"Error processing {pdf_path}: {e}")
            return []

    def process_directory(self) -> None:
        """Process all PDF files in the directory."""
        if not os.path.exists(self.directory_path):
            print(f"Directory not found: {self.directory_path}")
            return
        
        pdf_files = [f for f in os.listdir(self.directory_path) if f.lower().endswith('.pdf')]
        
        if not pdf_files:
            print(f"No PDF files found in {self.directory_path}")
            return
            
        print(f"Found {len(pdf_files)} PDF files to process")
        
        for filename in tqdm(pdf_files, desc="Processing PDFs"):
            pdf_path = os.path.join(self.directory_path, filename)
            print(f"\nProcessing {filename}...")
            
            chunks = self.process_pdf(pdf_path)
            
            if chunks:
                output_path = os.path.join(self.output_dir, f"{filename}_chunks.json")
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(chunks, f, ensure_ascii=False, indent=4)
                print(f"Saved {len(chunks)} chunks to {output_path}")
                
                print("\nSample chunk:")
                print(json.dumps(chunks[0], indent=2))
            else:
                print(f"No chunks generated for {filename}")

def main():
    print(f"Starting PDF processing in directory: {PDF_DIRECTORY}")
    processor = PDFProcessor(PDF_DIRECTORY)
    processor.process_directory()

if __name__ == "__main__":
    main()