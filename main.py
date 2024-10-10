import os
import time
import json
from tqdm import tqdm
from datasets import load_dataset

import config
from utils import load_few_shot_examples, translate_statement

os.environ['OPENAI_API_KEY'] = config.OPENAI_API_KEY
# os.environ['ANTHROPIC_API_KEY'] = config.ANTHROPIC_API_KEY
# os.environ['COHERE_API_KEY'] = config.COHERE_API_KEY

dataset = load_dataset(config.DATASET_NAME)
all_data = dataset['test'] + dataset['valid']
few_shot_examples = load_few_shot_examples(config.FEW_SHOT_EXAMPLES_PATH)

# creates 'results' directory if it doesn't exist
os.makedirs(config.RESULTS_DIR, exist_ok=True)

results = []

for item in tqdm(all_data):
    informal_statement = item['informal_statement']
    
    try:
        formal_statement = translate_statement(
            informal_statement,
            few_shot_examples,
            model=config.DEFAULT_MODEL,
            provider=config.DEFAULT_PROVIDER,
            temperature=0.0,
            max_tokens=500,
        )

        results.append({
            'name': item['name'],
            'informal_statement': informal_statement,
            'generated_formal_statement': formal_statement,
            'formal_statement': item['formal_statement'],
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

output_file = os.path.join(config.RESULTS_DIR, config.OUTPUT_FILE)

with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"Results have been saved to {output_file}")
