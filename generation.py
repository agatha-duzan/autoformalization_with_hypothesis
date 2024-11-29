import os
import time
import json
from tqdm import tqdm
from datasets import load_dataset, concatenate_datasets

import config
from utils import load_few_shot_examples, translate_statement, informal_hypothesis_decomp

os.environ['OPENAI_API_KEY'] = config.OPENAI_API_KEY

dataset = load_dataset(config.DATASET_NAME)
all_data = concatenate_datasets([dataset['test'], dataset['validation']])
few_shot_examples = load_few_shot_examples(config.FEW_SHOT_EXAMPLES_PATH) if config.FEWSHOT else []

# create results dir if it doesn't exist
os.makedirs(config.RESULTS_DIR, exist_ok=True)
results = []

for item in tqdm(all_data):
    informal_statement = item['informal_statement']
    
    try:
        if config.HYPOTHESIS_DECOMP == 'informal':
            decomp = informal_hypothesis_decomp(informal_statement, 
                                           model=config.DEFAULT_MODEL, 
                                           max_tokens=1000,)
        elif config.HYPOTHESIS_DECOMP == 'formal':
            formal_try = formal_statement = translate_statement(
            informal_statement,
            few_shot_examples,
            model=config.DEFAULT_MODEL,
            temperature=0.0,
            max_tokens=1000,
            )
        else:
            decomp = None

        formal_statement = translate_statement(
            informal_statement,
            few_shot_examples,
            hypothesis_decomp=decomp,
            model=config.DEFAULT_MODEL,
            temperature=0.0,
            max_tokens=1000,
            )

        # other passes: hypothesis decomp, REPL feedback, etc
        
        results.append({
            'name': item['name'],
            'informal_statement': informal_statement,
            'generated_formal_statement': formal_statement,
            'formal_statement': item['formal_statement'],
            'hypothesis_decomp': decomp, # OPTIONAL
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
