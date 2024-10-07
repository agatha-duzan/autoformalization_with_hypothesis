import json
from litellm import completion
from config import DEFAULT_MODEL, DEFAULT_PROVIDER

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

# LATER: test adding sorry to fewshot examples
def generate_prompt(informal_statement, few_shot_examples):
    examples_text = ""
    for example in few_shot_examples:
        examples_text += f"Informal statement:\n{example['nl_statement']}\n\nFormal statement in Lean 4:\n{example['formal_statement']}\n\n---\n\n"
    
    instruction = "You are an expert in formalizing mathematical statements in Lean 4. Given the following informal mathematical statement, write the corresponding formal statement in Lean 4 syntax."
    
    prompt = f"{instruction}\n\n{examples_text}\n\nInformal statement:\n{informal_statement}\n\nFormal statement in Lean 4:"
    return prompt

def translate_statement(informal_statement, few_shot_examples, model=DEFAULT_MODEL, provider=DEFAULT_PROVIDER, **kwargs):
    prompt = generate_prompt(informal_statement, few_shot_examples)
    response = completion(
        prompt=prompt,
        model=model,
        provider=provider,
        **kwargs
    )
    formal_statement = response.strip()
    return formal_statement
