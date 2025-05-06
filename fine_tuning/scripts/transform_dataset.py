#!/usr/bin/env python3
"""
Dataset Transformation Script for Gemini Fine-Tuning
---------------------------------------------------
This script downloads the OpenAssistant Guanaco dataset and transforms it
into the format required for Gemini model fine-tuning.
"""

import os
import json
import logging
from pathlib import Path
import datasets
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
DATA_DIR = Path("data/openassistant-guanaco")
DATASET_NAME = "timdettmers/openassistant-guanaco"
OUTPUT_JSONL = DATA_DIR / "openassistant_best_replies_train_reformatted.jsonl"
MAX_SAMPLES = 5000  # Limit number of samples to process (set to None for all)

def ensure_dir_exists(directory):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        logger.info(f"Created directory: {directory}")

def download_dataset():
    """Download the OpenAssistant Guanaco dataset."""
    logger.info(f"Downloading dataset: {DATASET_NAME}")
    
    try:
        # Load the dataset using the Hugging Face datasets library
        dataset = datasets.load_dataset(DATASET_NAME, split="train")
        logger.info(f"Dataset downloaded successfully with {len(dataset)} samples")
        return dataset
    except Exception as e:
        logger.error(f"Failed to download dataset: {e}")
        raise

def format_for_gemini(dataset, max_samples=None):
    """
    Format the dataset into the required structure for Gemini fine-tuning.
    
    The format required by Gemini's supervised fine-tuning is:
    {
        "messages": [
            {"role": "user", "content": "USER PROMPT"},
            {"role": "model", "content": "MODEL RESPONSE"}
        ]
    }
    """
    logger.info("Transforming dataset to Gemini fine-tuning format")
    
    formatted_data = []
    sample_count = len(dataset) if max_samples is None else min(max_samples, len(dataset))
    
    for i in tqdm(range(sample_count), desc="Formatting samples"):
        sample = dataset[i]
        
        # Each sample should contain a human prompt and an assistant response
        if "text" not in sample:
            logger.warning(f"Skipping sample {i}: missing 'text' field")
            continue
            
        # Parse the text which is in the format: "Human: {prompt}\n\nAssistant: {response}"
        text = sample["text"]
        
        try:
            # Extract human prompt and assistant response
            parts = text.split("Human: ", 1)
            if len(parts) < 2:
                logger.warning(f"Skipping sample {i}: unable to find 'Human:' prefix")
                continue
                
            human_and_assistant = parts[1]
            human_assistant_parts = human_and_assistant.split("\n\nAssistant: ", 1)
            
            if len(human_assistant_parts) < 2:
                logger.warning(f"Skipping sample {i}: unable to find 'Assistant:' section")
                continue
                
            human_prompt = human_assistant_parts[0].strip()
            assistant_response = human_assistant_parts[1].strip()
            
            # Create the formatted entry
            entry = {
                "messages": [
                    {"role": "user", "content": human_prompt},
                    {"role": "model", "content": assistant_response}
                ]
            }
            
            formatted_data.append(entry)
            
        except Exception as e:
            logger.warning(f"Error processing sample {i}: {e}")
            continue
    
    logger.info(f"Successfully formatted {len(formatted_data)} samples")
    return formatted_data

def save_to_jsonl(data, output_path):
    """Save the formatted data to a JSONL file."""
    logger.info(f"Saving formatted data to {output_path}")
    
    with open(output_path, "w") as f:
        for entry in data:
            f.write(json.dumps(entry) + "\n")
    
    logger.info(f"Saved {len(data)} samples to {output_path}")
    
    # Verify the file size
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    logger.info(f"File size: {file_size_mb:.2f} MB")
    
    # Display a sample entry
    with open(output_path, "r") as f:
        sample_line = f.readline()
        logger.info(f"Sample entry: {sample_line}")

def main():
    """Main execution function."""
    logger.info("Starting dataset transformation process")
    
    # Create necessary directories
    ensure_dir_exists(DATA_DIR)
    
    # Download the dataset
    dataset = download_dataset()
    
    # Format the dataset for Gemini fine-tuning
    formatted_data = format_for_gemini(dataset, max_samples=MAX_SAMPLES)
    
    # Save the formatted data to a JSONL file
    save_to_jsonl(formatted_data, OUTPUT_JSONL)
    
    logger.info(f"Transformation completed successfully! File saved to: {OUTPUT_JSONL}")
    logger.info(f"The dataset is now ready for fine-tuning Gemini models.")

if __name__ == "__main__":
    main() 