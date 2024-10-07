import os
from datasets import load_dataset
from litellm import completion
import time
from tqdm import tqdm
import json

dataset = load_dataset('agatha-duzan/number_theory_af')
all_data = dataset['test'] + dataset['valid']

def load_few_shot_examples(filepath):
    examples = []
    with open(filepath, 'r') as f:
        for line in f:
            data = json.loads(line)
            examples.append({
                'formal_statement': data['formal_statement'],
                'nl_statement': data['nl_statement']
            })
    return examples

few_shot_examples = load_few_shot_examples('/home/duzan/autoformalization_with_hypothesis/data/12_shot_proofnet_lean4.jsonl')

# LATER: test adding sorry to fewshot examples
def generate_prompt(informal_statement, few_shot_examples):
    examples_text = ""
    for example in few_shot_examples:
        examples_text += f"Informal statement:\n{example['nl_statement']}\n\nFormal statement in Lean 4:\n{example['formal_statement']}\n\n---\n\n"

    instruction = "You are an expert in formalizing mathematical statements in Lean 4. Given the following informal mathematical statement, write the corresponding formal statement in Lean 4 syntax."

    prompt = f"{instruction}\n{examples_text}\n\nInformal statement:\n{informal_statement}\n\nFormal statement in Lean 4:"
    return prompt


DEFAULT_MODEL = 'gpt-3.5-turbo'
DEFAULT_PROVIDER = 'openai'
os.environ['OPENAI_API_KEY'] = 'YOUR_OPENAI_API_KEY'
os.environ['ANTHROPIC_API_KEY'] = 'YOUR_ANTHROPIC_API_KEY'
os.environ['COHERE_API_KEY'] = 'YOUR_COHERE_API_KEY'

def translate_statement(informal_statement, model=DEFAULT_MODEL, provider=DEFAULT_PROVIDER, **kwargs):
    prompt = generate_prompt(informal_statement)
    response = completion(
        prompt=prompt,
        model=model,
        provider=provider,
        **kwargs
    )
    formal_statement = response.strip()
    return formal_statement

results_dir = 'results'
os.makedirs(results_dir, exist_ok=True)

results = []

for item in tqdm(all_data):
    informal_statement = item['informal_statement']

    try:
        formal_statement = translate_statement(
            informal_statement,
            model=DEFAULT_MODEL,
            provider=DEFAULT_PROVIDER,
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
            'model': DEFAULT_MODEL,
            'provider': DEFAULT_PROVIDER,
        })

        # if needed for rate limits
        # time.sleep(1)

    except Exception as e:
        print(f"An error occurred for item {item['name']}: {e}")
        continue

output_file = os.path.join(results_dir, 'translation_results.json')

with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"Results have been saved to {output_file}")