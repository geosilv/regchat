import os
import json
from pathlib import Path

def is_relevant_record(record):
    """Determine if a record is relevant based on specific criteria."""
    text = record.get("text", "").strip()
    keywords = record.get("metadata", {}).get("keywords", [])

    # Criteria for exclusion
    if not text:
        return False  # Exclude records with empty text
    if len(text.split()) < 20:
        return False  # Exclude records with fewer than 20 words
    if all(keyword in ["https", "Please", "published"] for keyword in keywords):
        return False  # Exclude boilerplate or redundant records

    return True

def clean_json_file(file_path, output_dir):
    """Clean a JSON file by removing irrelevant records."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Ensure the data is a list of records
        if not isinstance(data, list):
            print(f"Skipping non-list JSON file: {file_path}")
            return

        # Count initial records
        total_records = len(data)

        # Filter records based on relevance
        cleaned_data = [record for record in data if is_relevant_record(record)]

        # Count remaining and deleted records
        remaining_records = len(cleaned_data)
        deleted_records = total_records - remaining_records

        # Save the cleaned JSON file to the output directory
        output_path = os.path.join(output_dir, os.path.basename(file_path))
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(cleaned_data, f, indent=4)

        print(f"File: {file_path}")
        print(f"Total Records: {total_records}, Deleted: {deleted_records}, Remaining: {remaining_records}")

    except Exception as e:
        print(f"Error processing file {file_path}: {e}")

def clean_json_directory(input_dir, output_dir):
    """Clean all JSON files in a directory."""
    # Create the output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Get all JSON files in the input directory
    json_files = [f for f in os.listdir(input_dir) if f.lower().endswith(".json")]

    for json_file in json_files:
        file_path = os.path.join(input_dir, json_file)
        clean_json_file(file_path, output_dir)

if __name__ == "__main__":
    INPUT_DIR = r"C:\Users\gallo\source\VSCode\RegulationsProject\LanchainProcessedDocs\enriched_documents"  # Replace with your input directory
    OUTPUT_DIR = r"C:\Users\gallo\source\VSCode\RegulationsProject\LanchainProcessedDocs\cleaned_documents"  # Replace with your output directory

    clean_json_directory(INPUT_DIR, OUTPUT_DIR)
