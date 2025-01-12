import json
import os
import statistics
from pathlib import Path
import openai
from typing import Dict, List, Tuple  # Added Tuple to the imports
import time
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from openai import OpenAI  # Make sure to import OpenAI class


# Configuration
INPUT_DIR = r"C:\Users\gallo\source\VSCode\RegulationsProject\LanchainProcessedDocs\cleaned_documents"
OUTPUT_DIR = r"C:\Users\gallo\source\VSCode\RegulationsProject\LanchainProcessedDocs\output_statistics"
OPENAI_MODEL = "gpt-3.5-turbo"

def load_api_key():
    """Load OpenAI API key from environment variable or config file"""
    # First try environment variable
    api_key = os.getenv('OPENAI_API_KEY')
    
    # If not found, try to load from config.json
    if not api_key:
        config_path = 'config.json'
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
                api_key = config.get('OPENAI_API_KEY')
    
    return api_key

# Ensure output directory exists
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

def evaluate_text_usefulness(client: openai, text: str) -> float:
    """
    Evaluate the usefulness of a text chunk using GPT-3.5-turbo.
    Returns a score between 0 and 1.
    """
    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert at evaluating text chunks for their usefulness in search contexts. "
                              "Rate the following text on a scale from 0 to 1, where 0 means not useful at all for search "
                              "(e.g., formatting instructions, generic greetings) and 1 means highly useful (e.g., specific "
                              "technical content, unique identifiers, important definitions). "
                              "IMPORTANT: Respond with ONLY the numeric score, nothing else."
                },
                {
                    "role": "user",
                    "content": text
                }
            ],
            temperature=0.3
        )
        
        # Extract the response and clean it
        response_text = response.choices[0].message.content.strip()
        
        # Try to extract just the numeric value if there's additional text
        try:
            # First attempt: direct conversion
            score = float(response_text)
        except ValueError:
            # Second attempt: try to extract first number from the string
            import re
            numbers = re.findall(r"[0-9]*\.?[0-9]+", response_text)
            if numbers:
                score = float(numbers[0])
            else:
                print(f"Could not extract number from response: {response_text}")
                return 0.0
        
        # Ensure the score is between 0 and 1
        return max(0, min(1, score))
    
    except Exception as e:
        print(f"Error evaluating text: {e}")
        print(f"Full response was: {response.choices[0].message.content}")
        return 0.0

def process_json_file(client: OpenAI, file_path: str) -> Tuple[Dict, List[float]]:
    """
    Process a single JSON file and return statistics about the usefulness scores.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        records = json.load(f)

    scores = []
    for record in records:
        text = record.get('text', '')
        if text:
            # Add delay to respect API rate limits
            time.sleep(1)
            score = evaluate_text_usefulness(client, text)
            scores.append(score)
            # Add the score back to the record for future reference
            record['usefulness_score'] = score

    # Calculate statistics
    stats = {
        'file_name': os.path.basename(file_path),
        'total_records': len(scores),
        'average_score': statistics.mean(scores) if scores else 0,
        'median_score': statistics.median(scores) if scores else 0,
        'std_dev': statistics.stdev(scores) if len(scores) > 1 else 0,
        'min_score': min(scores) if scores else 0,
        'max_score': max(scores) if scores else 0
    }

    # Save the processed records with scores
    output_file = os.path.join(OUTPUT_DIR, f"processed_{os.path.basename(file_path)}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(records, f, indent=4, ensure_ascii=False)

    return stats, scores

def main():
    """
    Main function to process all JSON files in the input directory.
    """
    # Load API key
    api_key = load_api_key()
    if not api_key:
        raise ValueError("Please set the OPENAI_API_KEY environment variable or provide it in config.json")
    
    # Initialize the OpenAI client
    client = OpenAI(api_key=api_key)

    all_stats = []
    # Create a dictionary to store scores for each file
    file_scores = {}
    
    # Process each JSON file in the input directory
    for file_name in os.listdir(INPUT_DIR):
        if file_name.endswith('.json'):
            file_path = os.path.join(INPUT_DIR, file_name)
            print(f"Processing {file_name}...")
            
            stats, scores = process_json_file(client, file_path)
            all_stats.append(stats)
            file_scores[file_name] = scores
            
            # Save individual file statistics
            stats_file = os.path.join(OUTPUT_DIR, f"stats_{file_name}")
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=4)

    # Save overall statistics
    overall_stats = {
        'total_files_processed': len(all_stats),
        'average_score_across_files': statistics.mean([s['average_score'] for s in all_stats]),
        'file_statistics': all_stats
    }
    
    with open(os.path.join(OUTPUT_DIR, 'overall_statistics.json'), 'w', encoding='utf-8') as f:
        json.dump(overall_stats, f, indent=4)

    # Create distribution plot
    plt.figure(figsize=(12, 6))
    
    # Create a DataFrame for easier plotting
    plot_data = []
    for file_name, scores in file_scores.items():
        plot_data.extend([(score, file_name) for score in scores])
    
    df = pd.DataFrame(plot_data, columns=['Score', 'File'])
    
    # Create violin plot with overlaid box plot
    sns.violinplot(data=df, x='File', y='Score', inner='box')
    
    plt.title('Distribution of Usefulness Scores Across Files')
    plt.xlabel('File Name')
    plt.ylabel('Usefulness Score')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(os.path.join(OUTPUT_DIR, 'score_distributions.png'))
    plt.close()

if __name__ == "__main__":
    main()

