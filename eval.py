import os
import json
from tqdm import tqdm
from utils import bleu_eval, cos_similarity
import config

def evaluate_results(input_file, output_file):
    # Load the entire JSON file as a list of entries
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    results = []
    
    # Iterate over each entry with a tqdm progress bar
    for entry in tqdm(data, desc="Evaluating entries"):
        # Extract relevant attributes from each entry
        generated_formal_statement = entry.get("generated_formal_statement", "")
        formal_statement = entry.get("formal_statement", "")
        
        # Calculate BLEU score
        bleu_score = bleu_eval(generated_formal_statement, formal_statement)
        
        # Calculate cosine similarity
        cosine_sim = cos_similarity(generated_formal_statement, formal_statement)
        
        # Add evaluation metrics to the entry
        entry["bleu"] = bleu_score
        entry["cosine_similarity"] = cosine_sim
        
        # Append the updated entry to results
        results.append(entry)

    # Save results with added metrics
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Evaluated results have been saved to {output_file}")

# Define file paths based on config
input_file = os.path.join(config.RESULTS_DIR, f"{config.OUTPUT_NAME}_{config.DEFAULT_MODEL}.json")
output_file = os.path.join(config.EVAL_RESULTS_DIR, f"{config.OUTPUT_NAME}_{config.DEFAULT_MODEL}_evaluated.json")

# Create evaluation directory if it doesn't exist
os.makedirs(config.EVAL_RESULTS_DIR, exist_ok=True)

# Run evaluation
evaluate_results(input_file, output_file)
