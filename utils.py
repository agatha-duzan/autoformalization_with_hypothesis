import json
import re
import itertools
import os
import ast
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
from encoding_retrieval import retrieve

def load_few_shot_examples(filepath):
    examples = []
    with open(filepath, 'r') as f:
        for line in f:
            data = json.loads(line)
            example = {
                'formal_statement': data['formal_statement'],
                'nl_statement': data['nl_statement']
            }
            if 'decomp' in filepath and 'hyp_decomp' in data:
                example['hyp_decomp'] = data['hyp_decomp']
            examples.append(example)
    return examples

def generate_prompt(informal_statement, few_shot_examples, hypothesis_decomp=None, retrieved=None):
    examples_text = ""
    for example in few_shot_examples:
        examples_text += f"Informal statement:\n{example['nl_statement']}\nFormal statement in Lean 4:\n{example['formal_statement']} sorry\n\n"
    
    instruction = f"You are an expert in formalizing mathematical statements in Lean 4. Given the following informal mathematical statement, write the corresponding formal statement in Lean 4 syntax.\nOutput format: The translated LEAN 4 theorem should be provided as a single cohesive code block, displaying the correct syntax and formatting expected for a LEAN 4 theorem statement. Do not enclose the code block in backticks. write sorry as the proof."
    
    prompt = f"{instruction}\n\n"

    if few_shot_examples:
        prompt += f"Some examples:\n"
        prompt += f"{examples_text}\n\n"
        
    prompt += f"Now it's your turn: \nInformal statement:\n{informal_statement}\n\n"
    
    if hypothesis_decomp:
        prompt += f"Identified premisces and goal of the statement:\n{str(hypothesis_decomp)}\n\n"

    if retrieved:
        prompt += f"Here are some snippets from the Lean documentation that could be useful:\n"
        for snippet in retrieved:
            if snippet:
                prompt += f"{snippet}\n"
    
    prompt += f"\nFormal statement in Lean 4:"
    return prompt

def translate_statement(informal_statement, few_shot_examples= [], hypothesis_decomp=None, retrieved = None, model=DEFAULT_MODEL, **kwargs):
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

def proof_state_query(formal_statement: str) -> str:
    """
    Naive parser that tries to convert a Lean theorem statement into a Lean proof-state-like format.

    1. Extracts the theorem name and its parameters from curly braces `{}`, square brackets `[]`,
       and round parentheses `()`.
    2. Converts leading universal quantifiers (∀) from the main statement into context parameters.
    3. Outputs the context lines (x : Type, etc.), followed by the goal line:
         ⊢ <remaining_statement>
    """

    # Remove newlines and extra spaces
    stmt = " ".join(line.strip() for line in formal_statement.splitlines())
    # Remove leading/trailing spaces
    stmt = stmt.strip()

    # A very naive approach to detect the 'theorem' or 'lemma' heading and remove it
    # e.g. "theorem name { ... } [ ... ] ( ... ) : statement := ..." 
    # or   "lemma name ... : statement := ..."
    # We'll remove everything up to and including the theorem name.
    # Then we'll parse out the part after that.

    # Regex to capture something like "theorem THM_NAME" or "lemma THM_NAME"
    #   group(1) => theorem/lemma
    #   group(2) => name
    #   group(3) => the remainder
    pattern_header = r"^(?:theorem|lemma)\s+([A-Za-z0-9_']+)\s+(.*)$"
    m = re.search(pattern_header, stmt)
    if not m:
        # If we can't find a recognized pattern, return the original statement
        return f"Could not parse theorem header in:\n{formal_statement}"

    theorem_name = m.group(1)
    remainder = m.group(2).strip()

    # We'll parse out all bracket groups: { ... }, [ ... ], ( ... )
    # We will store them in a list that includes whether it came from curly, bracket, or paren
    # so that we can guess how to name them if needed.
    # Then we will remove them from the remainder and figure out the part after the colon.
    curly_pattern  = r"\{([^{}]*)\}"
    square_pattern = r"\[([^]]*)\]"
    paren_pattern  = r"\(([^)]*)\)"

    # We'll create a helper function to extract these bracketed segments in order
    # from left to right, while removing them from the remainder.
    
    def extract_segments(text):
        """
        Return a list of (kind, content) in the order found, where kind in {'curly','square','paren'}.
        Also return the leftover text with those segments removed.
        """
        # We do a simple repeated find from left to right, searching for the earliest bracket match.
        pattern_combined = re.compile(
            r"(?P<curly>\{([^{}]*)\})|(?P<square>\[([^]]*)\])|(?P<paren>\(([^)]*)\))"
        )
        segments = []
        start_idx = 0
        result_text = ""
        
        for match in pattern_combined.finditer(text):
            # text from start_idx to match.start() is "outside" brackets
            outside = text[start_idx:match.start()]
            result_text += outside

            if match.lastgroup == 'curly':
                # group(0) is the entire bracketed substring, group(2) is the content
                content = match.group(2)
                segments.append(('curly', content.strip()))
            elif match.lastgroup == 'square':
                content = match.group(4)
                segments.append(('square', content.strip()))
            elif match.lastgroup == 'paren':
                content = match.group(6)
                segments.append(('paren', content.strip()))

            start_idx = match.end()

        # leftover after the last match
        result_text += text[start_idx:]
        return segments, result_text.strip()

    # Extract bracketed segments in order
    segments, remainder_no_brackets = extract_segments(remainder)

    # Now remainder_no_brackets should look like: 
    #   ": ∀ (H : Subgroup G), H.Normal := sorry"
    # or possibly: ": ∃ ... := sorry"
    # or maybe just ": <statement> := sorry"

    # We want to find what's after the first colon but before " := " or " sorry"
    # We'll do a quick parse:
    colon_index = remainder_no_brackets.find(':')
    if colon_index == -1:
        return f"Could not find ':' in the remainder:\n{remainder_no_brackets}"

    # Everything before the colon might be extra stuff (in Lean, sometimes there is no extra stuff).
    # We typically want everything after the colon and before ":=" or "sorry".
    # e.g. "∀ (H : Subgroup G), H.Normal := sorry"
    statement_part = remainder_no_brackets[colon_index+1:].strip()

    # We can remove any trailing ":= sorry", ":=sorry", "sorry", etc.
    # We'll do it gently:
    statement_part = re.sub(r":=\s*sorry.*$", "", statement_part)
    statement_part = re.sub(r"sorry\s*$", "", statement_part)
    statement_part = statement_part.strip()

    # Now statement_part might be "∀ (H : Subgroup G), H.Normal" or
    # "(q : ℚ) (hq : 0 < q) : ∃ ..."
    #
    # Meanwhile, segments might contain e.g.
    #   [('curly', 'G : Type _'), ('square', 'CommGroup G')]
    #   or [('paren', 'q : ℚ'), ('paren', 'hq : 0 < q')]
    #
    # We'll parse them into context assumptions. 
    # For curly braces and square brackets, if there's no variable name, we guess something
    # like `_inst_1`, `_inst_2`, etc.  

    # Regex to capture a variable with a possible name, e.g. "G : Type _"
    # or "CommGroup G"
    # or "x y : Nat"
    # We'll do a naive approach: "^(.*?)\s*:\s*(.*)$"
    # We might have multiple names on the left: "x y z : SomeType"
    # We'll handle that by splitting on spaces if we see them.

    typeclass_counter = itertools.count(1)

    context_lines = []

    def parse_declaration(content: str, kind: str):
        """
        Given something like 'G : Type _' or 'CommGroup G' and the bracket kind,
        produce one or more lines of 'identifier : type'.
        If no identifier is found for a bracket-type that *should* have it,
        we generate `_inst_1 : content`.
        For curly braces, we do similarly (they can be implicit variables).
        For parentheses, we require an explicit variable (like (x : ℕ)).
        """
        content = content.strip()
        # If there's a colon, we treat it as 'x : Type'
        if ':' in content:
            # Could be multiple variables: "x y : SomeType"
            # We'll do a naive parse:
            left, right = content.split(':', 1)
            left_vars = left.split()
            type_ = right.strip()
            return [f"{v} : {type_}" for v in left_vars]
        else:
            # There's no colon. Possibly something like "CommGroup G".
            # We'll treat that as a typeclass argument, so we guess a name.
            if kind in ['square', 'curly']:
                idx = next(typeclass_counter)
                return [f"_inst_{idx} : {content}"]
            else:
                # For parentheses with no colon, we can't do much else.
                # We'll guess a name as well, though it's weird to do so for a parenthesized argument.
                idx = next(typeclass_counter)
                return [f"_inst_{idx} : {content}"]

    # Convert each segment into one or more lines
    for (kind, seg_content) in segments:
        # skip empty segments
        if not seg_content.strip():
            continue
        decls = parse_declaration(seg_content, kind)
        context_lines.extend(decls)

    # Now we handle the possibility that the main statement after the colon
    # has leading ∀ ( ... ) or a sugar version (x : T) ...
    # 
    # We will repeatedly parse:
    #   "∀ (x : T), rest"
    # or
    #   "(x : T) (y : U) ... -> rest"
    # 
    # until no more are found, collecting them into context.

    # We'll do a small loop that tries to detect patterns:
    #   "∀ (x : T), STMT" or "(x : T) -> STMT" or "(x : T) (y : U) -> STMT"
    # In Lean, `(x : T) (y : U) : final` is the same as `(x : T) → (y : U) → final`
    # We'll treat them as repeated context, so that eventually the leftover is the goal.

    # A small helper to extract leading parentheses of the form (x : T).
    # We'll reuse the same pattern for the "segments" approach:

    def extract_paren_params(text):
        """
        Extract leading (x : T) or (x y : T) from the front of text repeatedly,
        returning a list of 'x : T' context lines, plus the leftover text.
        """
        pattern = re.compile(r'^\(\s*([^)]*)\)\s*(.*)$')
        collected = []
        leftover = text.strip()
        while True:
            match = pattern.match(leftover)
            if not match:
                break
            inside = match.group(1).strip()
            leftover = match.group(2).strip()
            # parse the inside
            decls = parse_declaration(inside, 'paren')
            collected.extend(decls)
        return collected, leftover

    # We'll define a loop to handle repeated "∀ ..." or repeated parenthesis.
    def extract_foralls(text):
        """
        Repeatedly parse '∀ (x : T), ...' from the front of 'text'.
        Return (list_of_context_lines, leftover).
        """
        # pattern for ∀ (x : T), ...
        # We'll do a naive approach:  ^∀\s*\(([^)]*)\)\s*,\s*(.*)
        pattern = re.compile(r'^∀\s*\(\s*([^)]*)\)\s*,\s*(.*)$')
        cxts = []
        leftover = text.strip()
        while True:
            match = pattern.match(leftover)
            if not match:
                break
            inside = match.group(1).strip()  # e.g. "H : Subgroup G"
            leftover = match.group(2).strip()
            decls = parse_declaration(inside, 'paren')
            cxts.extend(decls)
        return cxts, leftover

    # First, parse leading ∀ (x : T), ...
    new_ctx, statement_part = extract_foralls(statement_part)
    context_lines.extend(new_ctx)

    # Then parse leading repeated parentheses as function arguments
    # e.g. (q : ℚ) (hq : 0 < q) ...
    new_ctx, statement_part = extract_paren_params(statement_part)
    context_lines.extend(new_ctx)

    # It's also possible the statement uses arrow -> notation, e.g. (x : T) -> (y : U) -> final
    # We'll do a small loop: if statement_part looks like something of the form
    # (x : T) -> rest, we keep extracting.
    def extract_arrow_parens(text):
        """
        Extract sequences like (x : T) -> ...
        Return (list_of_context_lines, leftover).
        """
        pattern = re.compile(r'^\(\s*([^)]*)\)\s*->\s*(.*)$')
        cxts = []
        leftover = text.strip()
        while True:
            match = pattern.match(leftover)
            if not match:
                break
            inside = match.group(1).strip()
            leftover = match.group(2).strip()
            decls = parse_declaration(inside, 'paren')
            cxts.extend(decls)
        return cxts, leftover

    new_ctx, statement_part = extract_arrow_parens(statement_part)
    context_lines.extend(new_ctx)

    # Now if there's still a leading '∀ (x : T), ...' we should parse again
    # because sometimes Lean has multiple Pi's. We'll just call extract_foralls again:
    new_ctx, statement_part = extract_foralls(statement_part)
    context_lines.extend(new_ctx)

    # The leftover is presumably the goal
    goal = statement_part.strip()

    # Build the final proof-state-like string
    # Each context line on its own line. Then "⊢ goal".
    # We will also do a small cleanup if it is something like "Type _" => "Type ?"
    # purely to mimic Lean's display in your example.
    context_lines_cleaned = []
    for line in context_lines:
        # purely aesthetic replacement: "Type _" -> "Type ?"
        # (since Lean often shows "Type ?")
        line = re.sub(r"Type\s*_[^,]*", "Type ?", line)
        context_lines_cleaned.append(line)

    # Deduplicate lines if needed (not always desirable, but sometimes helpful).
    # Let's skip dedup because different braces might declare the same variable with different roles.

    # Now produce the final string
    # If no context, we just show "⊢ goal"
    if not context_lines_cleaned:
        return f"⊢ {goal}"

    context_part = "\n".join(context_lines_cleaned)
    return f"{context_part}\n⊢ {goal}"

def leansearch_hypothesis_decomp(informal_statement, few_shot_examples, model=DEFAULT_MODEL, **kwargs):
    instruction = f'''You are a helpful assistant specializing in mathematical reasoning. You will be given a mathematical statement in natural language. Your task is to:

1. Break down the statement into separate premises or components.
2. For each premise, propose a natural language query that will be used by a documentation search tool (Leansearch) to retrieve relevant Lean documentation or definitions.
3. Present the result as a dictionary in the following format, do not enclose your answer in backticks:

{{<premise in plain language>: <Leansearch query>,...,<premise in plain language>: <Leansearch query>}}

Make sure to:
- Identify all important objects, functions, sets, properties, and relationships within the statement.
- Generate a short natural language query for each premise. 
- Do not attempt to formalize the statement in Lean yet. Only provide the premises and corresponding queries.

Below are some examples:
'''
    for item in few_shot_examples:
        instruction += f"natural language statement: {item['nl_statement']} \n"
        instruction += f"hyp_decomp: {item['hyp_decomp']} \n\n"

    instruction += f"Now it's your turn: \nnatural language statement: {informal_statement} \nhyp_decomp: "

    messages = [{"role": "user", "content": instruction}]
    response = completion(
        messages=messages,
        model=model,
        **kwargs
    )

    decomp = response.choices[0].message.content
    decomp_dict = ast.literal_eval(decomp)
    return decomp_dict

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