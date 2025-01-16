import os
import time
import json
from tqdm import tqdm
from datasets import load_dataset, concatenate_datasets

import config
from utils import *
from encoding_retrieval import *

os.environ['OPENAI_API_KEY'] = config.OPENAI_API_KEY

dataset = load_dataset(config.DATASET_NAME)
all_data = concatenate_datasets([dataset['test'], dataset['validation']])
few_shot_examples = load_few_shot_examples(config.FEW_SHOT_EXAMPLES_PATH) if config.FEWSHOT else []

# for leandojo retrieval only: setup LeanDojo model and tokenizer
if config.METHOD == 'leandojo':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    premises, encodings = get_premises_and_encodings(premises_file = "/home/agatha/Desktop/MA3/sem proj/autoformalization_with_hypothesis/data/premises_defs.pkl")
    print("Premises and encodings loaded!")
    leandojo_tokenizer = AutoTokenizer.from_pretrained("kaiyuy/leandojo-lean4-retriever-byt5-small")
    leandojo_model = AutoModelForTextEncoding.from_pretrained("kaiyuy/leandojo-lean4-retriever-byt5-small")
    leandojo_model.eval()
    leandojo_model.to(device)
    print("LeanDojo model and tokenizer loaded!")

# create results dir if it doesn't exist
os.makedirs(config.RESULTS_DIR, exist_ok=True)
results = []

for item in tqdm(all_data):
    informal_statement = item['informal_statement']
    
    try:
        if config.METHOD == 'informal_decomp':
            decomp = informal_hypothesis_decomp(informal_statement, 
                                           model=config.DEFAULT_MODEL, 
                                           max_tokens=1000,)
        elif config.METHOD == 'leandojo':
            formal_try = translate_statement(
            informal_statement,
            few_shot_examples,
            model=config.DEFAULT_MODEL,
            temperature=0.0,
            max_tokens=1000,
            )

            query = proof_state_query(formal_try)
            retrieved = retrieve(
                query = query, 
                premises=premises, 
                encodings=encodings, 
                k=5, 
                tokenizer=leandojo_tokenizer, 
                model = leandojo_model
            )

        elif config.METHOD == 'leansearch':
            decomp = leansearch_hypothesis_decomp(informal_statement, few_shot_examples)
            retrieved = [leansearch(query) for query in decomp.values()]       
        else:
            decomp = None
            # retrieved = leansearch(informal_statement, k=5)

        formal_statement = translate_statement(
            informal_statement,
            few_shot_examples,
            # hypothesis_decomp=decomp,
            retrieved = retrieved,
            model=config.DEFAULT_MODEL,
            temperature=0.0,
            max_tokens=1000,
            )
        
        results.append({
            'name': item['name'],
            'informal_statement': informal_statement,
            'generated_formal_statement': formal_statement,
            'formal_statement': item['formal_statement'],
            # 'hypothesis_decomp': decomp, # ADAPT
            'retrieved': retrieved, # ADAPT
            'tags': item['tags'],
            'header': item['header'],
            'split': item['split'],
            'model': config.DEFAULT_MODEL,
            'provider': config.DEFAULT_PROVIDER,
        })

        # if rate limits:
        # time.sleep(1)

    except Exception as e:
        print(f"An error occurred for item {item['name']}: {e}")
        continue

output_file = os.path.join(config.RESULTS_DIR, f"{config.OUTPUT_NAME}_{config.DEFAULT_MODEL}.json")

with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"Results have been saved to {output_file}")
