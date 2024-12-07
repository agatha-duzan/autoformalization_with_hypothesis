import json
import os
import multiprocessing as mp
import re
import jsonlines
from multiprocessing.pool import Pool
from rich.console import Console
from rich.syntax import Syntax
from tqdm import tqdm

import config
from eval import load_checkpoint, save_checkpoint
from pyleanrepl import RobustLeanServer, clean_cache

console = Console()


def _is_valid_proof(
    lean_output: dict, proof_start_line: int | None = None, proof_end_line: int | None = None, verbose: bool = False
):
    def message_intersects_proof(message, start_line, end_line):
        res = True
        if start_line is not None:
            if message["endPos"]:
                res = res and message["endPos"]["line"] >= start_line
        if end_line is not None:
            if message["startPos"]:
                res = res and message["startPos"]["line"] <= end_line
        return res

    # check only the messages intersecting the proof
    sorries = [
        message
        for message in lean_output.get("sorries", [])
        if message_intersects_proof(message, proof_start_line, proof_end_line)
    ]
    errors = [
        message
        for message in lean_output.get("messages", [])
        if message_intersects_proof(message, proof_start_line, proof_end_line)
        and message["severity"] == "error"
        and not message["data"] == "no goals to be solved"  # goal is solved but useless tactics were applied at the end
    ]
    if verbose:
        console.print(f"Errors: {errors}")
        console.print(f"Sorries: {sorries}")
    return not sorries and not errors


def lean_comments_ranges(lean_code: str) -> list[tuple[int, int]]:
    """Extract the ranges of Lean comments from a Lean code snippet."""
    # TODO: this method does not handle strings and potentially other edge cases (i.e. this method will probably crash if `/-`, `-/` or `--` are used in a string)

    # multiline comments
    open_comment_indices = [m.start() for m in re.finditer(r"/-", lean_code)]
    close_comment_indices = [m.start() + 2 for m in re.finditer(r"-/", lean_code)]
    if len(open_comment_indices) != len(close_comment_indices):
        raise ValueError("Mismatched open and close comment indices.")

    # trick to handle nested comments in a simple way
    multiline_comment_ranges = list(zip(open_comment_indices, close_comment_indices))

    # single line comments
    single_line_comment_ranges = [(m.start(), lean_code.find("\n", m.start())) for m in re.finditer(r"--", lean_code)]

    # merge potential overlapping ranges
    comment_ranges = sorted(multiline_comment_ranges + single_line_comment_ranges, key=lambda x: x[0])
    merged_comment_ranges = []
    for start, end in comment_ranges:
        if merged_comment_ranges and start <= merged_comment_ranges[-1][1]:
            merged_comment_ranges[-1] = (merged_comment_ranges[-1][0], max(merged_comment_ranges[-1][1], end))
        else:
            merged_comment_ranges.append((start, end))

    return merged_comment_ranges


def remove_lean_comments(lean_code: str) -> str | None:
    try:
        comment_ranges = lean_comments_ranges(lean_code)

        new_lean_code = ""
        prev_start = 0
        for start, end in comment_ranges:
            new_lean_code += lean_code[prev_start:start]
            prev_start = end

        new_lean_code += lean_code[prev_start:]
        return new_lean_code

    except Exception:
        return None


def split_implementation(declaration: str, start: int = 0):
    # for a theorem, an implementation is the proof
    if ":=" in declaration:
        # we have to be careful here as ":=" can be used inside the declaration itself
        indices = set([m.start() for m in re.finditer(r":=", declaration)])

        # implementation using pcre2 blows up the memory, and it turns out it is faster to use a python loop
        counters = {"(": 0, "{": 0, "[": 0}
        closing = {")": "(", "}": "{", "]": "["}
        for i, c in enumerate(declaration[start:]):
            if c in counters:
                counters[c] += 1
            elif c in [")", "}", "]"]:
                counters[closing[c]] -= 1
            if all([v == 0 for v in counters.values()]) and (i + start) in indices:
                return i + start + 2
    return None


def clean_theorem_string(theorem_string: str, new_theorem_name: str = "dummy", add_sorry: bool = True) -> str | None:
    try:
        # clean the theorem string
        clean_formal = remove_lean_comments(theorem_string)
        clean_formal = re.sub(r"\s+", " ", clean_formal).strip()

        # find where the "theorem" keyword is
        clean_formal = clean_formal[clean_formal.index("theorem") :]

        # if a proof is provided we remove it
        idx_implement = split_implementation(clean_formal)
        if idx_implement is not None:
            clean_formal = clean_formal[:idx_implement].strip()

        # add ":=" at the end of the string if it is missing
        if ":=" not in clean_formal:
            clean_formal += " :="

        # remove "theorem" and the theorem name
        clean_formal = clean_formal[clean_formal.index(" ") :].strip()
        clean_formal = clean_formal[clean_formal.index(" ") :].strip()

        clean_formal = f"theorem {new_theorem_name} " + clean_formal
        if add_sorry:
            clean_formal += " sorry"
        return clean_formal
    except Exception:
        return None


def extract_last_theorem(lean_code: str) -> int:
    """Extract the last theorem from a Lean code snippet. It assumes that the Lean code snippet ends with a theorem."""
    comments_ranges = lean_comments_ranges(lean_code)

    # find last theorem by looking for `theorem` keyword surrounded by whitespaces, or by being at the beginning of the string
    theorem_indices = [m.start() for m in re.finditer(r"\btheorem\s", lean_code)]
    if not theorem_indices:
        raise ValueError(f"No theorem found in the provided Lean code:\n{lean_code}")

    # remove matches that are inside comments
    theorem_indices = [idx for idx in theorem_indices if not any(start <= idx <= end for start, end in comments_ranges)]

    return theorem_indices[-1]


def clean_last_theorem_string(lean_code: str, new_theorem_name: str = "dummy") -> str:
    """Clean the last theorem string from a Lean code snippet. It assumes that the Lean code snippet ends with a theorem."""
    idx_last_theorem = extract_last_theorem(lean_code)
    clean_thm = clean_theorem_string(lean_code[idx_last_theorem:], new_theorem_name, add_sorry=False)
    if clean_thm is not None:
        return lean_code[:idx_last_theorem] + clean_thm

    raise ValueError(f"Theorem extraction failed for the following Lean code:\n{lean_code}")


def indent_code(code: str, nb_spaces: int = 2) -> str:
    return "\n".join(" " * nb_spaces + line for line in code.split("\n"))


def split_conclusion(declaration: str, start: int = 0) -> int | None:
    counters = {"(": 0, "{": 0, "[": 0}
    closing = {")": "(", "}": "{", "]": "["}
    for i, c in enumerate(declaration[start:]):
        if c in counters:
            counters[c] += 1
        elif c in [")", "}", "]"]:
            counters[closing[c]] -= 1
        if all([v == 0 for v in counters.values()]) and c == ":":
            return i + start
    return None


class BEqMetricCPU:
    """Implement a stricter version of the BEq (Bidirectional Extended Definitional Equivalence) metric from
    "RETHINKING AND IMPROVING AUTOFORMALIZATION: TOWARDS A FAITHFUL METRIC AND A DEPENDENCY RETRIEVAL-BASED APPROACH" paper."""

    def __call__(
        self,
        formalization_pairs: list,
        verbose: bool = False,
        output_file: str | None = None,
        nb_process: int | None = None,
        timeout_per_eq: int = 600,
        timeout_per_proof: int = 60,
    ) -> list[bool]:
        """
        Compare the given formalization pairs using the BEq metric.

        Args:
            formalization_pairs: List of tuples containing two formalizations to compare as strings and the source header in the format (formalization_1, formalization_2, src_header).
            verbose: Whether to print additional information during the verification process.
            output_file: File to save the results in JSONL format.
            nb_process: Number of processes to use for the verification. If None, the number of processes is set to the number of CPUs.
            timeout_per_proof: Timeout in seconds per proof. This is used to avoid getting stuck on a single proof, but will not interrupt the overall verification process.
        """
        idx_formalization_pairs = list(enumerate(formalization_pairs))

        proof_results = [(None, None) for _ in formalization_pairs]
        with Pool(nb_process, maxtasksperchild=1) as p:
            iterator = p.imap_unordered(
                _check_equivalence,
                [(idx, timeout_per_proof, context_proofs) for idx, context_proofs in idx_formalization_pairs],
                chunksize=1,
            )
            pbar = tqdm(total=len(formalization_pairs), desc="BEq CPU - metric computation")
            for i, _ in enumerate(idx_formalization_pairs):
                try:
                    idx, proof_result = iterator.next(timeout_per_eq)
                    proof_results[idx] = proof_result
                    pbar.update(1)
                except mp.TimeoutError:
                    console.log(
                        f"Timeout during BEq computation. {len(formalization_pairs) - i} elements from the list have been left unchecked."
                    )
                    p.terminate()
                    p.join()
                    break

        if output_file:
            with jsonlines.open(output_file, "w") as writer:
                for (formalization_1, formalization_2, src_header), (proof_1, proof_2) in zip(
                    formalization_pairs, proof_results
                ):
                    writer.write(
                        {
                            "formalization_1": formalization_1,
                            "formalization_2": formalization_2,
                            "src_header": src_header,
                            "proof:formalization_1=>formalization_2": proof_1,
                            "proof:formalization_2=>formalization_1": proof_2,
                            "equivalence_proven": proof_1 is not None and proof_2 is not None,
                        }
                    )

        return [proof_1 is not None and proof_2 is not None for proof_1, proof_2 in proof_results]


def _check_equivalence(args: tuple[int, int, tuple[str, str]]) -> tuple[int, tuple[str | None, str | None]]:
    """
    Filter function to check if at least one proof is valid for a given context and formalization.
    """
    idx, timeout_per_proof, context_proofs = args[0], args[1], args[2]
    formalization_1, formalization_2, src_header = context_proofs

    server = RobustLeanServer(rev="v4.11.0-rc1", require="mathlib")
    # using the cache accelerates the verification process by at least one order of magnitude
    # it also drastically reduces the memory usage
    context_env = server.run_code(src_header, add_to_persistent_cache=True)["env"]

    base_thm_name = "base_theorem"
    reformulated_thm_name = "reformulated_theorem"

    def prove_all(tactics: list[str]) -> str:
        prove_independent = " ; ".join([f"(all_goals try {t})" for t in tactics])
        prove_combined = "all_goals (" + " ; ".join([f"(try {t})" for t in tactics]) + ")"
        return "all_goals intros\nfirst | (" + prove_independent + ") | (" + prove_combined + ")"

    solver_tactics_apply = ["tauto", "simp_all_arith!", "noncomm_ring", "exact?"]
    solver_tactics_have = ["tauto", "simp_all_arith!", "exact? using this"]
    proof_all_apply = prove_all(solver_tactics_apply)
    proof_all_have = prove_all(solver_tactics_have)

    res = [None, None]
    for i, (base_thm, reform_thm) in enumerate(
        [
            (formalization_1, formalization_2),
            (formalization_2, formalization_1),
        ]
    ):
        try:
            formal_1_code = f"{clean_last_theorem_string(base_thm, base_thm_name)} sorry\n\n"
            formal_2_start_line = formal_1_code.count("\n") + 1
            formal_2_code = f"{clean_last_theorem_string(reform_thm, reformulated_thm_name)} by"
        except ValueError:
            continue

        proofs = []
        prepended_proof = "\nintros\nsymm_saturate"

        # 1. try to apply the base theorem directly
        proofs.append(indent_code(prepended_proof + f"\napply {base_thm_name}\n" + proof_all_apply, 2))

        # 2. try to add the conlusion of the base theorem as hypothesis
        # sanity check: if we can prove `reform_thm` using a tactic in `solver_tactics_have` without introducing the hypothesis,
        # then we should skip this `have` step as it may introduce a false positive
        provable_without_have = False
        try:
            provable_without_have = _is_valid_proof(
                server.run_code(formal_2_code + proof_all_have, env=context_env, timeout=timeout_per_proof)
            )
        except (TimeoutError, EOFError, json.JSONDecodeError):
            server.restart()

        if not provable_without_have:
            idx_conclusion = split_conclusion(formal_1_code)
            if idx_conclusion:
                idx_end_conclusion = formal_1_code.rfind(":=")
                conclusion = formal_1_code[idx_conclusion:idx_end_conclusion].strip()
                have_stmt_proof = (
                    prepended_proof
                    + f"\nhave {conclusion} := by\n"
                    + indent_code(f"apply_rules [{base_thm_name}]\n" + proof_all_apply, 2)
                    + "\n"
                )
                proofs.append(indent_code(have_stmt_proof + proof_all_have, 2))

        # 3. try to apply the base theorem with some tolerance on the differences in the conclusion
        for max_step in range(0, 5):
            proofs.append(
                indent_code(
                    prepended_proof
                    + f"\nconvert (config := .unfoldSameFun) {base_thm_name} using {max_step}\n"
                    + proof_all_apply,
                    2,
                )
            )

        # 4. last attempt: use `exact?` tactic. We make sure later that it is using the base theorem in the proof.
        proofs.append(indent_code(prepended_proof + "\nexact?", 2))

        for j, proof in enumerate(proofs):
            try:
                lean_output = server.run_code(
                    formal_1_code + formal_2_code + proof, env=context_env, timeout=timeout_per_proof
                )
            except (TimeoutError, EOFError, json.JSONDecodeError):
                server.restart()
                continue

            if _is_valid_proof(lean_output, proof_start_line=formal_2_start_line):
                if j == len(proofs) - 1:  # `exact?` tactic
                    proof = _extract_proof(lean_output, proof_start_line=formal_2_start_line)
                    if base_thm_name not in proof:
                        continue
                res[i] = proof
                break

    return idx, tuple(res)


def _extract_proof(
    lean_output: dict, proof_start_line: int | None = None, proof_end_line: int | None = None, verbose: bool = False
) -> str | None:
    def message_intersects_proof(message, start_line, end_line):
        res = True
        if start_line is not None:
            res = res and message["endPos"]["line"] >= start_line
        if end_line is not None:
            res = res and message["startPos"]["line"] <= end_line
        return res

    # check only the messages intersecting the proof
    for message in lean_output.get("messages", []):
        if message_intersects_proof(message, proof_start_line, proof_end_line):
            if message["severity"] == "error":
                return None
            if message["severity"] == "info" and message["data"].startswith("Try this:"):
                return message["data"].split("Try this:")[1].strip()
    return None


def toy_example():
    # clean_cache()
    beq_metric = BEqMetricCPU()

    # src_header = "import Mathlib"
    src_header = """import Mathlib

open Fintype Group Monoid
open Set Real Ideal Polynomial
open scoped BigOperators"""

    formalization_pairs = [
        (
            "theorem random_name_1 {G : Type*} [Group G] [Fintype G] (h : Fintype.card G % 2 = 0) :\n  ∃ a : G, a ≠ 1 ∧ a = a⁻¹ :=",
            "theorem random_name_2 {G : Type*} [Group G] [Fintype G] (hG2 : Even (card G)) :\n  ∃ (a : G), a ≠ 1 ∧ a = a⁻¹ :=",
            src_header,
        ),
        (
            "theorem sP : Infinite {p : Nat.Primes // p ≡ -1 [ZMOD 6]} :=",
            "theorem sQ : Set.Infinite {p : ℕ | Nat.Prime p ∧ p % 6 = 5} :=",
            src_header,
        ),
        (
            "theorem sP {R : Type u_1} [Ring R] (h : ∀ (x : R), x ^ 3 = x) (x : R) (y : R) : x * y = y * x :=",
            "theorem sQ {R : Type*} [Ring R] (h : ∀ x : R, x ^ 3 = x) : Nonempty (CommRing R) :=",
            src_header,
        ),
        (
            "theorem sPpp {G : Type*} [Group G] [Fintype G] {p q : ℕ} (hp : Prime p) (hq : Prime q) (hG : card G = p*q) :  IsSimpleGroup G → False :=",
            "theorem sQqq (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (G : Type _) [Group G] [Fintype G] (hG : Fintype.card G = p * q) : ¬ IsSimpleGroup G :=",
            src_header,
        ),
        (
            "theorem sPppp {f : ℝ → ℝ} (hf : ∀ x y, |f x - f y| ≤ |x - y| ^ 2) : ∃ c, f = λ x => c :=",
            "theorem sQqqq (f : ℝ → ℝ) (h : ∀ (t x : ℝ), |f t - f x| ≤ |t - x| ^ 2) (x : ℝ) (y : ℝ) : f x = f y :=",
            src_header,
        ),
        (
            "theorem dummy (n : ℕ) (hn : n % 2 = 1) : 8 ∣ n^2 - 1 :=",
            "theorem dummy {n : ℕ} (hn : Odd n) : 8 ∣ (n^2 - 1) :=",
            src_header,
        ),
        (
            "theorem dumssmy {G : Type*} [Group G] (x : G) : x ^ 2 = 1 ↔ orderOf x = 1 ∨ orderOf x = 2 :=",
            "theorem dumssfmy {G : Type*} [Group G] : ∀ (x : G), orderOf x = 1 ∨ orderOf x = 2 ↔ x ^ 2 = 1 :=",
            src_header,
        ),
        (
            "theorem dummy83 : Irreducible (Polynomial.C (12 : ℚ) + Polynomial.C (6 : ℚ) * Polynomial.X + Polynomial.X ^ 3) :=",
            "theorem dummy84 : Irreducible (12 + 6 * X + X ^ 3 : Polynomial ℚ) :=",
            src_header,
        ),
        # (
        #     "theorem dummy90 {p : ℕ} (hp : Nat.Prime p) (n : ℕ) (hn : 0 < n) : Irreducible (Polynomial.C (1 : ℚ) * Polynomial.X ^ n - Polynomial.C (p : ℚ)) :=",
        #     "theorem dummy91 (p : ℕ) (hp : Prime p) (n : ℕ) (hn : n > 0) : Irreducible (X ^ n - (p : Polynomial ℚ) : Polynomial ℚ) :=",
        #     src_header,
        # ),
        # (
        #     "theorem dummy64 {X X' : Type*} [TopologicalSpace X] [TopologicalSpace X'] (π₁ : X × X' → X) (π₂ : X × X' → X') (h₁ : π₁ = Prod.fst) (h₂ : π₂ = Prod.snd) : IsOpenMap π₁ ∧ IsOpenMap π₂ :=",
        #     "theorem dummy63 {X X' : Type*} [TopologicalSpace X] [TopologicalSpace X'] : (∀ U : Set (X × X'), IsOpen U → IsOpen (Prod.fst '' U)) ∧ (∀ U : Set (X × X'), IsOpen U → IsOpen (Prod.snd '' U)) :=",
        #     src_header,
        # ),
        # (
        #     "theorem dummy {x : ℝ} (r : ℚ) (hr : r ≠ 0) (hx : Irrational x) : Irrational (r + x) :=",
        #     "theorem dummy (x : ℝ) (y : ℚ) (hy : y ≠ 0) : ( Irrational x ) -> Irrational ( x + y ) :=",
        #     src_header,
        # ),
    ]

    res = beq_metric(formalization_pairs, verbose=True, nb_process=8)

    console.print("BEq metric results:")
    for (formalization_1, formalization_2, _), result in zip(formalization_pairs, res):
        console.print()
        console.rule()
        console.print("Comparing formalizations:")
        console.print(Syntax(formalization_1, "lean4"))
        console.print(Syntax(formalization_2, "lean4"))
        console.print(f"Result: {'[green]Equivalent[/green]' if result else '[yellow]Not conclusive[/yellow]'}")

if __name__ == "__main__":
    NB_PROCESS = 6 # setup for my laptop's cpu (12 cores), increase if you can

    input_file = os.path.join(config.EVAL_RESULTS_DIR, f"{config.OUTPUT_NAME}_{config.DEFAULT_MODEL}_eval1.json")
    output_file = os.path.join(config.EVAL_RESULTS_DIR, f"{config.OUTPUT_NAME}_{config.DEFAULT_MODEL}_eval2.json")
    checkpoint_file = os.path.join(config.CHECKPOINT_DIR, f"{config.OUTPUT_NAME}_{config.DEFAULT_MODEL}_checkpoint2.json")
    os.makedirs(config.EVAL_RESULTS_DIR, exist_ok=True)

    with open(input_file, 'r') as f:
        data = json.load(f)


    # if no header is specified, we import Mathlib
    for item in data:
        if not item['header'].strip():
            item['header'] = 'import Mathlib'

    # load checkpoint if it exists and get already processed entries
    results = load_checkpoint(checkpoint_file)
    processed_entries = {entry["name"] for entry in results}

    data_done = [item for item in data if item['repl'] == 0]
    data_eval = [item for item in data if item['repl'] == 1 and item['name'] not in processed_entries]

    print("Loading BEq metric...")
    beq_metric = BEqMetricCPU()
    print("BEq metric loaded!")

    for i in tqdm(range(0, len(data_eval), NB_PROCESS)):
        batch = data_eval[i:i + NB_PROCESS]

        formalization_pairs = [
            (entry["formal_statement"], entry["generated_formal_statement"], entry["header"])
            for entry in batch
        ]

        res = beq_metric(formalization_pairs, verbose=True, nb_process=NB_PROCESS)
        res = [int(eval_result) for eval_result in res]

        for entry, result in zip(batch, res):
            entry["beq"] = result
            results.append(entry)
            processed_entries.add(entry["name"])

        # save checkpoint
        save_checkpoint(results, checkpoint_file)

    # Save final results to output file
    all_data = data_done + results
    with open(output_file, 'w') as f:
        json.dump(all_data, f, indent=4)

