import json
import os
from datetime import datetime
from openai import OpenAI
import tiktoken
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio

class DocumentProcessor:
    def __init__(self, api_key: str):
        """Initialize the processor with OpenAI API key."""
        self.client = OpenAI(api_key=api_key)
        self.encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        
    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text string."""
        return len(self.encoding.encode(text))
    
    async def get_gpt_analysis(self, text: str, progress_desc: str) -> Tuple[str, List[str]]:
        """Get both summary and keywords using GPT-3.5-turbo."""
        try:
            # Update progress description for the current operation
            tqdm.write(f"Processing: {progress_desc}")
            
            # First, get the summary
            summary_response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Please provide a brief, concise summary of the following text."},
                    {"role": "user", "content": text}
                ],
                max_tokens=100,
                temperature=0.3
            )
            summary = summary_response.choices[0].message.content.strip()

            # Then, get the keywords
            keyword_response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Extract 5-7 key topics or important terms from the following text. Respond with only the keywords separated by commas, without any additional text or explanation."},
                    {"role": "user", "content": text}
                ],
                max_tokens=50,
                temperature=0.3
            )
            keywords_text = keyword_response.choices[0].message.content.strip()
            keywords = [k.strip() for k in keywords_text.split(',')]

            return summary, keywords
        except Exception as e:
            tqdm.write(f"Error getting GPT analysis for {progress_desc}: {e}")
            return "Error generating summary", []

    def process_json_record(self, record: Dict[str, Any], document_title: str, chunk_index: int) -> Dict[str, Any]:
        """Process a single JSON record and add metadata."""
        # Create a copy of the record to avoid modifying the original
        enriched_record = record.copy()
        
        # Add metadata
        enriched_record["metadata"] = {
            "document_title": document_title,
            "page_number": record.get("page_start", 0),
            "chunk_index": chunk_index,
            "token_count": self.count_tokens(record.get("text", "")),
            "processing_timestamp": datetime.utcnow().isoformat(),
            "summary": None,  # Will be filled by GPT
            "keywords": []    # Will be filled by GPT
        }
        
        return enriched_record

    async def process_json_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Process a single JSON file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                records = json.load(f)
            
            document_title = os.path.basename(file_path)
            enriched_records = []
            
            # Create progress bar for records within the file
            total_records = len(records)
            tqdm.write(f"\nProcessing {document_title} - {total_records} records")
            
            for idx, record in tqdm(enumerate(records), total=total_records, desc=f"Processing {document_title}", leave=False):
                # Process basic metadata
                enriched_record = self.process_json_record(record, document_title, idx)
                
                # Get GPT summary and keywords
                progress_desc = f"{document_title} - Record {idx + 1}/{total_records}"
                summary, keywords = await self.get_gpt_analysis(record.get("text", ""), progress_desc)
                enriched_record["metadata"]["summary"] = summary
                enriched_record["metadata"]["keywords"] = keywords
                
                enriched_records.append(enriched_record)
                
            return enriched_records
            
        except Exception as e:
            tqdm.write(f"Error processing file {file_path}: {e}")
            return []

    async def process_directory(self, directory_path: str, output_directory: str = None) -> None:
        """Process all JSON files in a directory."""
        if output_directory is None:
            output_directory = os.path.join(directory_path, 'processed')
            
        os.makedirs(output_directory, exist_ok=True)
        
        # Get list of JSON files
        json_files = [f for f in os.listdir(directory_path) if f.endswith('.json')]
        total_files = len(json_files)
        
        if total_files == 0:
            print("No JSON files found in the directory.")
            return
            
        print(f"\nFound {total_files} JSON files to process")
        
        # Process files with progress bar
        for filename in tqdm(json_files, desc="Processing files", unit="file"):
            input_path = os.path.join(directory_path, filename)
            output_path = os.path.join(output_directory, f'processed_{filename}')
            
            enriched_records = await self.process_json_file(input_path)
            
            if enriched_records:
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(enriched_records, f, indent=2, ensure_ascii=False)
                tqdm.write(f"✓ Completed {filename} -> {output_path}")
            else:
                tqdm.write(f"✗ Failed to process {filename}")

        print("\nProcessing complete!")
        print(f"Processed files can be found in: {output_directory}")

# Usage example
async def main():
    api_key = "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"
    processor = DocumentProcessor(api_key)
    
    # Process a directory of JSON files
    input_directory = r"C:\Users\gallo\source\VSCode\RegulationsProject\LanchainProcessedDocs\processed_documents"
    output_directory = r"C:\Users\gallo\source\VSCode\RegulationsProject\LanchainProcessedDocs\enriched_documents"
   
    
    await processor.process_directory(input_directory, output_directory)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())