import os
import json
import re
import codecs
from jsonschema import validate, ValidationError

# Define the JSON schema for validation
schema = {
    "type": "object",
    "properties": {
        "id": {"type": "string"},
        "title": {"type": "string"},
        "summary": {
            "type": "object",
            "properties": {
                "full_summary": {"type": "string"},
                "sections": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "section_title": {"type": "string"},
                            "text": {"type": "string"}
                        },
                        "required": ["section_title", "text"]
                    }
                }
            },
            "required": ["full_summary", "sections"]
        },
        "metadata": {
            "type": "object",
            "properties": {
                "author": {"type": "string"},
                "date": {"type": "string", "format": "date"},
                "source_type": {"type": "string"},
                "tags": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "url": {"type": "string", "format": "uri"}
            },
            "required": ["author", "date", "source_type", "tags", "url"]
        }
    },
    "required": ["id", "title", "summary", "metadata"]
}

def sanitize_json_content(content):
    """Sanitizes JSON content by handling various types of control characters and encoding issues."""
    # Replace common problematic characters
    replacements = {
        '\x00': '',  # null
        '\x0A': ' ', # line feed
        '\x0D': ' ', # carriage return
        '\x1A': '',  # substitute
        '\x1E': '',  # record separator
        '\x1F': '',  # unit separator
        '\x7F': '',  # delete
        '\u2028': ' ', # line separator
        '\u2029': ' ', # paragraph separator
    }
    
    # First pass: handle known control characters
    for char, replacement in replacements.items():
        content = content.replace(char, replacement)
    
    # Second pass: remove any remaining control characters
    content = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', content)
    
    # Third pass: handle potential UTF-8 BOM and encoding issues
    try:
        # Try to decode as UTF-8-SIG (UTF-8 with BOM)
        content = content.encode('utf-8').decode('utf-8-sig')
    except UnicodeError:
        pass
    
    # Fourth pass: normalize whitespace
    content = re.sub(r'\s+', ' ', content)
    
    return content.strip()

def sanitize_json(filepath):
    """Reads a JSON file, handles encoding issues, and returns sanitized content."""
    try:
        # First try with UTF-8
        with open(filepath, 'r', encoding='utf-8') as file:
            content = file.read()
    except UnicodeDecodeError:
        try:
            # If UTF-8 fails, try with UTF-8-SIG (BOM)
            with open(filepath, 'r', encoding='utf-8-sig') as file:
                content = file.read()
        except UnicodeDecodeError:
            try:
                # Last resort: try to detect encoding
                with open(filepath, 'rb') as file:
                    raw = file.read()
                    content = raw.decode(detect_encoding(raw))
            except Exception as e:
                print(f"Fatal encoding error in '{filepath}': {e}")
                return None
    
    # Sanitize the content
    sanitized_content = sanitize_json_content(content)
    
    try:
        return json.loads(sanitized_content)
    except json.JSONDecodeError as e:
        print(f"JSON decoding error in '{filepath}': {e}")
        # Additional debug information
        error_context = get_error_context(sanitized_content, e.pos)
        print(f"Context around error:\n{error_context}")
        return None

def detect_encoding(raw_bytes):
    """Attempts to detect the encoding of the file."""
    encodings = ['utf-8', 'utf-8-sig', 'utf-16', 'ascii', 'iso-8859-1']
    for encoding in encodings:
        try:
            raw_bytes.decode(encoding)
            return encoding
        except UnicodeDecodeError:
            continue
    return 'utf-8'  # default to UTF-8 if no encoding is detected

def get_error_context(content, error_position, context_size=50):
    """Returns the content around the error position for debugging."""
    start = max(0, error_position - context_size)
    end = min(len(content), error_position + context_size)
    before = content[start:error_position]
    after = content[error_position:end]
    return f"...{before}👉ERROR HERE👈{after}..."

def fix_and_validate_json(filepath):
    """Fixes common issues in a JSON file and validates it."""
    try:
        data = sanitize_json(filepath)
        if data is None:
            print(f"Could not process '{filepath}' due to persistent JSON errors.")
            return None

        fixed = False
        # Fix missing required fields in the schema
        if "id" not in data:
            data["id"] = os.path.splitext(os.path.basename(filepath))[0]
            print(f"Fixed missing 'id' in {filepath}")
            fixed = True

        if "title" not in data:
            data["title"] = "Untitled Document"
            print(f"Fixed missing 'title' in {filepath}")
            fixed = True

        if "summary" not in data:
            data["summary"] = {
                "full_summary": "No summary available.",
                "sections": []
            }
            print(f"Fixed missing 'summary' in {filepath}")
            fixed = True

        if "metadata" not in data:
            data["metadata"] = {
                "author": "Unknown",
                "date": "1970-01-01",
                "source_type": "Unknown",
                "tags": [],
                "url": "https://example.com"
            }
            print(f"Fixed missing 'metadata' in {filepath}")
            fixed = True

        # Validate the JSON file against the schema
        validate(instance=data, schema=schema)

        # Save fixes if any were made
        if fixed:
            with open(filepath, "w", encoding="utf-8") as file:
                json.dump(data, file, indent=4, ensure_ascii=False)
            print(f"Saved fixes to {filepath}")

        print(f"File '{filepath}' passed validation.")
        return data
    except ValidationError as e:
        print(f"Schema validation error in '{filepath}': {e.message}")
        return None
    except Exception as e:
        print(f"Unexpected error in '{filepath}': {e}")
        return None

def validate_and_fix_json_files(directory):
    """Validates all JSON files in a directory and attempts to fix issues."""
    validated_files = {}
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            filepath = os.path.join(directory, filename)
            data = fix_and_validate_json(filepath)
            if data is not None:
                validated_files[filename] = data
    
    print(f"Successfully loaded {len(validated_files)} out of {len([f for f in os.listdir(directory) if f.endswith('.json')])} JSON files.")
    return validated_files

# Example usage
if __name__ == "__main__":
    directory = "summaries/"
    validated_files = validate_and_fix_json_files(directory)