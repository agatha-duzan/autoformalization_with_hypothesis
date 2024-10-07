import os
from datasets import load_dataset
from litellm import completion
import time
from tqdm import tqdm
import json

# Load the dataset
dataset = load_dataset('agatha-duzan/number_theory_af')
all_data = dataset['test'] + dataset['valid']

def generate_prompt(informal_statement):
    prompt = f"""You are an expert in formalizing mathematical statements in Lean 4. Given the following informal mathematical statement, write the corresponding formal statement in Lean 4 syntax.

Informal statement:
{informal_statement}

Formal statement in Lean 4:
"""
    return prompt

# Set default model and provider
DEFAULT_MODEL = 'gpt-3.5-turbo'     # Change this to the desired model
DEFAULT_PROVIDER = 'openai'         # Change this to the desired provider

# Set your API keys as environment variables or directly in the script
# For security reasons, it's recommended to set these as environment variables
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
        # Call the translation function
        formal_statement = translate_statement(
            informal_statement,
            model=DEFAULT_MODEL,
            provider=DEFAULT_PROVIDER,
            temperature=0.0,
            max_tokens=500,
        )

        # Save the result
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

# Define the output file path within the 'results' directory
output_file = os.path.join(results_dir, 'translation_results.json')

# Save the results
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"Results have been saved to {output_file}")