import os
import json
from tqdm import tqdm
from repl.server import RobustLeanServer

from utils import bleu_eval, cos_similarity, get_repl_errors
import config

def load_checkpoint(checkpoint_file):
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            return json.load(f)
    return []

def save_checkpoint(results, checkpoint_file):
    with open(checkpoint_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Checkpoint saved to {checkpoint_file}")

def evaluate_results(input_file, output_file, checkpoint_file, save_every=5):
    with open(input_file, 'r') as f:
        data = json.load(f)
        
    # Load checkpoint if it exists and get already processed entries
    results = load_checkpoint(checkpoint_file)
    processed_entries = {entry["name"] for entry in results}

    print("Starting Lean server...")
    lean_server = RobustLeanServer()
    print("Lean server ready!")
    
    for i, entry in enumerate(tqdm(data, desc="Evaluating entries"), 1):
        entry_name = entry.get("name")

        if entry_name in processed_entries:
            continue  # Skip entries that have already been processed

        generated_formal_statement = entry.get("generated_formal_statement", "")
        formal_statement = entry.get("formal_statement", "")
        
        bleu_score = bleu_eval(generated_formal_statement, formal_statement)
        cosine_sim = cos_similarity(generated_formal_statement, formal_statement)

        try:
            repl_errors = get_repl_errors(generated_formal_statement, entry.get("header", ""), lean_server)
        except Exception as e:
            repl_errors = str(e)

        

        entry["bleu"] = bleu_score
        entry["cosine_similarity"] = cosine_sim
        entry["repl_errors"] = repl_errors

        if isinstance(repl_errors, str):
            repl = 0  # an exception occurred
        else:
            repl = 0 if any(error["severity"] == "error" for error in repl_errors) else 1
        entry["repl"] = repl

        # type check is a necessary condition for equivalence
        if repl == 0:
            entry['beq'] = 0
        
        results.append(entry)
        processed_entries.add(entry_name)

        # Save checkpoint
        if i % save_every == 0:
            save_checkpoint(results, checkpoint_file)

    # Save final results to output file
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Evaluated results have been saved to {output_file}")

if __name__ == "__main__":
    input_file = os.path.join(config.RESULTS_DIR, f"{config.OUTPUT_NAME}_{config.DEFAULT_MODEL}.json")
    output_file = os.path.join(config.EVAL_RESULTS_DIR, f"{config.OUTPUT_NAME}_{config.DEFAULT_MODEL}_eval1.json")
    checkpoint_file = os.path.join(config.CHECKPOINT_DIR, f"{config.OUTPUT_NAME}_{config.DEFAULT_MODEL}_checkpoint1.json")

    os.makedirs(config.EVAL_RESULTS_DIR, exist_ok=True)
    evaluate_results(input_file, output_file, checkpoint_file)
