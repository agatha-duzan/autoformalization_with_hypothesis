import os
import json
from datasets import load_dataset
import config
from encoding_retrieval import leansearch
from tqdm import tqdm

def main():
    # Load both test and valid splits of the PAug/ProofNetSharp dataset
    ds_test = load_dataset("PAug/ProofNetSharp", split="test")
    ds_valid = load_dataset("PAug/ProofNetSharp", split="valid")
    
    # Combine the two splits into one list of examples
    all_examples = list(ds_test) + list(ds_valid)
    
    # Ensure the output directory exists
    os.makedirs(config.PREMISES_DIR, exist_ok=True)
    output_file = os.path.join(config.PREMISES_DIR, f"{config.OUTPUT_NAME}_{config.DEFAULT_MODEL}.json")
    
    with open(output_file, "w") as out_f:
        for ex in tqdm(all_examples, desc="Processing examples"):
            # Query using the natural language statement
            query = ex["nl_statement"]
            # Retrieve top 5 premises via leansearch
            retrieved = leansearch(query, k=5)
            # Collect the 'formal_name' from each retrieved premise
            retrieved_names = [item["formal_name"] for item in retrieved]
            
            out_example = {
                "id": ex["id"],
                "nl_statement": ex["nl_statement"],
                "lean4_src_header": ex["lean4_src_header"],
                "lean4_formalization": ex["lean4_formalization"],
                "retrieved_premises": retrieved_names
            }
            out_f.write(json.dumps(out_example) + "\n")
    
    print(f"Saved premises to {output_file}")

if __name__ == "__main__":
    main()