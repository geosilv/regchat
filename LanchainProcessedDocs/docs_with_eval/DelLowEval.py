import json
import os
from pathlib import Path

# Configuration
INPUT_DIR = r"C:\Users\gallo\source\VSCode\RegulationsProject\LanchainProcessedDocs\docs_with_eval"  # Change this to your input directory
OUTPUT_DIR = r"C:\Users\gallo\source\VSCode\RegulationsProject\LanchainProcessedDocs\filtered_documents"  # Change this to your desired output directory

# Create output directory if it doesn't exist
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

def filter_json_file(file_path: str, threshold: float) -> None:
    """
    Filter records in a JSON file based on usefulness_score threshold
    
    Args:
        file_path (str): Path to the JSON file
        threshold (float): Minimum usefulness score to keep record
    """
    try:
        # Read the JSON file
        with open(file_path, 'r', encoding='utf-8') as f:
            records = json.load(f)
        
        # Filter records based on threshold
        filtered_records = [
            record for record in records 
            if record.get('usefulness_score', 0) >= threshold
        ]
        
        # Create output filename
        input_filename = os.path.basename(file_path)
        output_path = os.path.join(OUTPUT_DIR, f"filtered_{input_filename}")
        
        # Save filtered records
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(filtered_records, f, indent=4, ensure_ascii=False)
            
        # Print statistics
        print(f"\nProcessed {input_filename}:")
        print(f"Original records: {len(records)}")
        print(f"Filtered records: {len(filtered_records)}")
        print(f"Removed {len(records) - len(filtered_records)} records")
        
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")

def main():
    """
    Main function to process all JSON files in the input directory
    """
    # Process each JSON file in the input directory
    for filename in os.listdir(INPUT_DIR):
        if filename.endswith('.json'):
            file_path = os.path.join(INPUT_DIR, filename)
            
            # Determine threshold based on filename
            if 'GIM' in filename.upper() or 'TRIM' in filename.upper():
                threshold = 0.4
                print(f"\nApplying 0.4 threshold to {filename} (GIM/TRIM file)")
            else:
                threshold = 0.6
                print(f"\nApplying 0.6 threshold to {filename} (standard file)")
            
            filter_json_file(file_path, threshold)

if __name__ == "__main__":
    print("Starting JSON filtering process...")
    main()
    print("\nProcessing complete. Filtered files have been saved to:", OUTPUT_DIR)