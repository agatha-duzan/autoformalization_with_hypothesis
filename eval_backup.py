import os
import json
from tqdm import tqdm
from repl.server import RobustLeanServer
from utils import bleu_eval, cos_similarity, get_repl_errors
import config

def evaluate_results(input_file, output_file):
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    results = []

    print("Starting Lean server...")
    lean_server = RobustLeanServer()
    print("Lean server ready!")
    
    for entry in tqdm(data, desc="Evaluating entries"):
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
        
        results.append(entry)

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Evaluated results have been saved to {output_file}")

input_file = os.path.join(config.RESULTS_DIR, f"{config.OUTPUT_NAME}_{config.DEFAULT_MODEL}.json")
output_file = os.path.join(config.EVAL_RESULTS_DIR, f"{config.OUTPUT_NAME}_{config.DEFAULT_MODEL}_evaluated.json")

os.makedirs(config.EVAL_RESULTS_DIR, exist_ok=True)

evaluate_results(input_file, output_file)
