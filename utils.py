import json
import re
import os
import config

os.environ['OPENAI_API_KEY'] = config.OPENAI_API_KEY

import openai
openai.api_key = os.environ['OPENAI_API_KEY']

from litellm import completion
from openai import OpenAI
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from sklearn.metrics.pairwise import cosine_similarity
from repl.server import LeanServer, RobustLeanServer

from config import *
from encoding import retrieve

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

def generate_prompt(informal_statement, few_shot_examples, hypothesis_decomp):
    examples_text = ""
    for example in few_shot_examples:
        examples_text += f"Informal statement:\n{example['nl_statement']}\n\nFormal statement in Lean 4:\n{example['formal_statement']} sorry\n\n"
    
    instruction = f"You are an expert in formalizing mathematical statements in Lean 4. Given the following informal mathematical statement, write the corresponding formal statement in Lean 4 syntax.\nOutput format: The translated LEAN 4 theorem should be provided as a single cohesive code block, displaying the correct syntax and formatting expected for a LEAN 4 theorem statement. Do not enclose the code block in backticks. write sorry as the proof."
    
    prompt = f"{instruction}\n\n"

    if few_shot_examples:
        prompt += f"{examples_text}\n\n"
        
    prompt += f"Informal statement:\n{informal_statement}\n\n"
    
    if hypothesis_decomp:
        prompt += f"Identified premisces and goal of the statement:\n{str(hypothesis_decomp)}\n\n"

    prompt += "Formal statement in Lean 4:"
    return prompt

# add modularity: many ways to generate prompt and messages
# few shot : static or NN?
# messages: all in one prompt or as 'chat history' ?

def translate_statement(informal_statement, few_shot_examples= [], hypothesis_decomp=None, model=DEFAULT_MODEL, **kwargs):
    prompt = generate_prompt(informal_statement, few_shot_examples, hypothesis_decomp)
    messages = [{"role": "user", "content": prompt}]
    
    response = completion(
        messages=messages,
        model=model,
        **kwargs
    )

    formal_statement = response.choices[0].message.content
    
    return formal_statement

def informal_hypothesis_decomp(informal_statement, model=DEFAULT_MODEL, **kwargs):
    instruction = f'''Extract the premises and goal from an informal theorem statement to assist in formalizing it in LEAN 4.

# Steps

1. **Identify Premises**:
   - Read the informal statement carefully.
   - Identify the assumptions or conditions that are provided in the theorem.
   - Extract these premises clearly and mark them as conditions that must hold true.

2. **Identify the Goal**:
   - Identify what is being proven or concluded from the assumptions in the statement.
   - Clearly separate the goal from the premises.

# Output Format

- A concise breakdown using the following structure, do not enclose the code block in backticks.:
{{
  "premises": [
    "[Premise 1]",
    "[Premise 2]",
    "... (List other premises)"
  ],
  "goal": "[The goal that follows from the premises]"
}}

# Notes 

- Carefully distinguish terms that indicate premises ("if," "given," "assume") from those that indicate goals ("then," "thus," "is").
- Ensure that all premises are complete and independently meaningful.
- The goal should directly represent what the theorem is asserting, without including extraneous details.'''
    
    prompt = f"{instruction}\n\nInformal statement:\n{informal_statement}\n\nOutput:"
    messages = [{"role": "user", "content": prompt}]

    response = completion(
        messages=messages,
        model=model,
        **kwargs
    )

    decomp = response.choices[0].message.content

    return decomp

def formal_hypothesis_decomp(informal_statement, model=DEFAULT_MODEL, **kwargs):
    # generate first try
    # extract hypothesis from it
    # for each hypothesis, retrieve top k premises from corpora
    # second pass: retieval augmented generation
    return

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

def bleu_eval(generated_formal_statement, formal_statement):
    # clean up both statements
    generated = clean_theorem_string(generated_formal_statement)
    reference = clean_theorem_string(formal_statement)

    return sentence_bleu(
        [reference.split()],
        generated.split(),
        smoothing_function=SmoothingFunction().method4
    )

def get_embedding(text, model="text-embedding-ada-002"):
    client = OpenAI()
    response = client.embeddings.create(input=text,model=model)
    return response.data[0].embedding

def cos_similarity(generated_formal_statement, formal_statement, model="text-embedding-ada-002"):
    generated_embedding = get_embedding(generated_formal_statement, model=model)
    reference_embedding = get_embedding(formal_statement, model=model)

    cosine_sim = cosine_similarity([generated_embedding], [reference_embedding])
    return cosine_sim[0][0]

def get_repl_errors(formal_statement, header, lean_server):
    if not header:
        header = 'import Mathlib\n\n'
    full_message = header + clean_theorem_string(formal_statement)

    result = lean_server.run_code(full_message, timeout=60)
    messages = result['messages']

    error_messages = []
    for message in messages:
        if message['severity'] == 'error':
            error_messages.append(message)

    return error_messages