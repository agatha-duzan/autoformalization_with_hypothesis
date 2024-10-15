import json
import re

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
def generate_prompt(informal_statement, few_shot_examples, instruction=None):
    examples_text = ""
    for example in few_shot_examples:
        examples_text += f"Informal statement:\n{example['nl_statement']}\n\nFormal statement in Lean 4:\n{example['formal_statement']}\n\n---\n\n"
    
    instruction = f"You are an expert in formalizing mathematical statements in Lean 4. Given the following informal mathematical statement, write the corresponding formal statement in Lean 4 syntax.\nOutput format: The translated LEAN 4 theorem should be provided as a single cohesive code block, displaying the correct syntax and formatting expected for a LEAN 4 theorem statement. Do not enclose the code block in backticks. write sorry as the proof."
    
    prompt = f"{instruction}\n\n{examples_text}\n\nInformal statement:\n{informal_statement}\n\nFormal statement in Lean 4:"
    return prompt

# add modularity: many ways to generate prompt and messages
# few shot : static or NN?
# messages: all in one prompt or as 'chat history' ?

def translate_statement(informal_statement, few_shot_examples, model=DEFAULT_MODEL, **kwargs):
    prompt = generate_prompt(informal_statement, few_shot_examples)
    messages = [{"role": "user", "content": prompt}]
    
    response = completion(
        messages=messages,
        model=model,
        **kwargs
    )

    formal_statement = response.choices[0].message.content
    
    return formal_statement


def clean_theorem_string(theorem_string: str, new_theorem_name: str = "dummy") -> str | None:
    try:
        # find where the "theorem" keyword is
        clean_formal = theorem_string[theorem_string.index("theorem") :]

        # for each line, remove Lean comments
        clean_formal = " ".join([re.sub(r"--.*", "", line) for line in clean_formal.split("\n")])

        # replace all whitespaces by single spaces
        clean_formal = re.sub(r"\s+", " ", clean_formal).strip()

        # add ":=" at the end of the string if it is missing
        if ":=" not in clean_formal:
            clean_formal += " :="

        # if a proof is provided we remove it
        for start_proof_kw in ["begin", "by"]:
            if f":= {start_proof_kw}" in clean_formal:
                clean_formal = clean_formal[: clean_formal.find(f":= {start_proof_kw}") + 2]

        # remove everything after last ":="
        clean_formal = clean_formal[: clean_formal.rfind(":=") + 2].strip()

        # remove "theorem" and the theorem name
        clean_formal = clean_formal[clean_formal.index(" ") :].strip()
        clean_formal = clean_formal[clean_formal.index(" ") :].strip()

        return f"theorem {new_theorem_name} " + clean_formal + "\nsorry"
    except Exception:
        return None
