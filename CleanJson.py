import json

# Parameters
INPUT_FILE = r"C:\Users\gallo\OneDrive\Desktop\Regulations\extracted_ECB_TRIM2017.json"  # Changed extension to .json
OUTPUT_FILE = r"C:\Users\gallo\OneDrive\Desktop\Regulations\extracted_ECB_TRIM2017clean.json"  # Changed extension to .json
MIN_TEXT_LENGTH = 100  # Change this to your desired minimum text length

def filter_json_by_text_length(input_file, output_file, min_length):
    # Read the JSON file
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: '{input_file}' is not a valid JSON file.")
        return

    # Ensure data is a list
    if not isinstance(data, list):
        data = [data]

    # Filter records
    original_count = len(data)
    filtered_data = [record for record in data if len(record.get('text', '')) >= min_length]
    removed_count = original_count - len(filtered_data)

    # Write filtered data back to file
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(filtered_data, f, indent=4, ensure_ascii=False)
            
        print("\nProcessing Summary:")
        print("=" * 50)
        print(f"Initial number of records: {original_count}")
        print(f"Records deleted (text < {min_length} chars): {removed_count}")
        print(f"Final number of records: {len(filtered_data)}")
        print("=" * 50)
    except IOError as e:
        print(f"Error writing to '{output_file}': {e}")

if __name__ == "__main__":
    # Run the filter
    filter_json_by_text_length(INPUT_FILE, OUTPUT_FILE, MIN_TEXT_LENGTH)