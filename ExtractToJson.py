import fitz  # PyMuPDF
import json
import re
import os
from pathlib import Path
from datetime import datetime

# Configuration
INPUT_DIR = r"C:\Users\gallo\source\VSCode\RegulationsProject\LanchainProcessedDocs"
OUTPUT_DIR = os.path.join(INPUT_DIR, "processed_documents")

def generate_metadata(chunk, document_name, section_name):
    """Generate metadata for a given chunk of text."""
    timestamp = datetime.utcnow().isoformat() + "Z"
    keywords = list(set(re.findall(r'\b\w{5,}\b', chunk)))  # Extract keywords (words with 5+ letters)
    summary = chunk[:200] + ("..." if len(chunk) > 200 else "")  # Simple summary (first 200 chars)

    return {
        "timestamp": timestamp,
        "section_name": section_name,
        "document_name": document_name,
        "keywords": keywords,
        "summary": summary
    }

def split_into_sentences(text):
    """Split text into sentences using regex while handling common abbreviations."""
    text = ' '.join(text.split())
    text = re.sub(r'(?<=Mr)\.', '@', text)
    text = re.sub(r'(?<=Ms)\.', '@', text)
    text = re.sub(r'(?<=Dr)\.', '@', text)
    text = re.sub(r'(?<=Prof)\.', '@', text)
    text = re.sub(r'(?<=e\.g)\.', '@', text)
    text = re.sub(r'(?<=i\.e)\.', '@', text)
    text = re.sub(r'(?<=Fig)\.', '@', text)
    text = re.sub(r'(?<=Vol)\.', '@', text)

    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    sentences = [s.replace('@', '.') for s in sentences]
    return sentences

def chunk_text(text, min_tokens=150, max_tokens=250):
    """Split text into chunks respecting sentence boundaries."""
    sentences = split_into_sentences(text)
    chunks = []
    current_chunk = []
    current_token_count = 0

    for sentence in sentences:
        sentence_tokens = len(re.findall(r'\w+|\S', sentence))

        if current_token_count + sentence_tokens > max_tokens and current_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_token_count = sentence_tokens
        else:
            current_chunk.append(sentence)
            current_token_count += sentence_tokens

            if current_token_count >= min_tokens and sentence.strip().endswith('.'):
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_token_count = 0

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks

def extract_sections_from_pdf(pdf_path, document_name):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    doc = fitz.open(pdf_path)
    sections = []
    current_section = None

    try:
        for page_num in range(doc.page_count):
            page = doc[page_num]
            text_blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_LIGATURES)["blocks"]

            for block in text_blocks:
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        font_name = span.get("font", "").lower()
                        font_size = span.get("size", 0)
                        span_text = span["text"].strip()

                        if ("bold" in font_name or font_size > 12) and len(span_text) > 0:
                            if current_section and current_section["text"]:
                                chunked_text = chunk_text(current_section["text"])
                                for chunk in chunked_text:
                                    if chunk.strip():
                                        metadata = generate_metadata(
                                            chunk, document_name, current_section["section_name"]
                                        )
                                        sections.append({
                                            "section_name": current_section["section_name"],
                                            "page_start": current_section["page_start"],
                                            "text": chunk.strip(),
                                            "metadata": metadata
                                        })

                            current_section = {
                                "section_name": span_text,
                                "page_start": page_num + 1,
                                "text": ""
                            }
                        elif current_section:
                            current_section["text"] += " " + span_text

        if current_section and current_section["text"]:
            chunked_text = chunk_text(current_section["text"])
            for chunk in chunked_text:
                if chunk.strip():
                    metadata = generate_metadata(
                        chunk, document_name, current_section["section_name"]
                    )
                    sections.append({
                        "section_name": current_section["section_name"],
                        "page_start": current_section["page_start"],
                        "text": chunk.strip(),
                        "metadata": metadata
                    })

    finally:
        doc.close()

    return sections

def save_to_file(data, output_file):
    """Save extracted sections to a JSON file."""
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    print(f"Extracted sections saved to {output_file}")

def process_directory(input_dir, output_dir):
    """Process all PDF files in the input directory and save results to output directory."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    pdf_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.pdf')]

    for pdf_file in pdf_files:
        print(f"Processing {pdf_file}...")
        pdf_path = os.path.join(input_dir, pdf_file)
        output_file = os.path.join(output_dir, f"{os.path.splitext(pdf_file)[0]}_processed.json")

        try:
            extracted_sections = extract_sections_from_pdf(pdf_path, pdf_file)
            save_to_file(extracted_sections, output_file)
            print(f"Successfully processed {pdf_file}")

        except Exception as e:
            print(f"Error processing {pdf_file}: {str(e)}")
            continue

if __name__ == "__main__":
    process_directory(INPUT_DIR, OUTPUT_DIR)
