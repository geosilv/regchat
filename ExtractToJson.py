import fitz
import pymupdf  # PyMuPDF
import json

# Configuration
PDF_PATH = r"C:\Users\gallo\OneDrive\Desktop\Regulations\ .pdf"

OUTPUT_FILE = r"C:\Users\gallo\OneDrive\Desktop\Regulations\extracted_ECB_TRIM2017.json"

# Function to extract sections and their text
def extract_sections_from_pdf(pdf_path, document_name):
    doc = fitz.open(pdf_path)
    sections = []
    current_section = None

    for page_num in range(doc.page_count):
        page = doc[page_num]
        text_blocks = page.get_text("dict")["blocks"]

        for block in text_blocks:
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    font_name = span.get("font", "").lower()
                    font_size = span.get("size", 0)
                    span_text = span["text"].strip()

                    # Identify section headers (bold or larger font)
                    if "bold" in font_name or font_size > 12:
                        # Start a new section if a header is found
                        if current_section:
                            sections.append(current_section)
                        current_section = {
                            "section_name": span_text,
                            "page_start": page_num + 1,
                            "text": ""
                        }
                    elif current_section:
                        # Add text to the current section
                        current_section["text"] += " " + span_text

        # Append any remaining text for the current page to the section
        if current_section and page_num + 1 > current_section.get("page_start", 0):
            sections.append(current_section)
            current_section = None

    # Append the last section
    if current_section:
        sections.append(current_section)

    doc.close()

    # Add metadata for the document
    for section in sections:
        section["document_name"] = document_name

    return sections

# Function to save extracted sections to a file
def save_to_file(data, output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    print(f"Extracted sections saved to {output_file}")

# Main Script
if __name__ == "__main__":
    document_name = PDF_PATH.split("\\")[-1]  # Extract document name from the path
    extracted_sections = extract_sections_from_pdf(PDF_PATH, document_name)
    save_to_file(extracted_sections, OUTPUT_FILE)
