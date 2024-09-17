import json
import os
import random
import re
from collections.abc import Callable
from copy import deepcopy
from multiprocessing.pool import Pool

import evaluate
import jsonlines
import numpy as np
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
REPL_DIR = os.path.join(ROOT_DIR, "repl")
DATA_DIR = os.path.join(ROOT_DIR, "data")

console = Console()


def merge_consecutive_roles(messages):
    """Merge consecutive messages with the same role.
    Some APIs don't allow sending multiple consecutive messages with the same role."""
    merged_messages = []
    for message in messages:
        if merged_messages and merged_messages[-1]["role"] == message["role"]:
            merged_messages[-1]["content"] += "\n\n" + message["content"]
        else:
            merged_messages.append(message)
    return merged_messages


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


## Self-consistency code used to get results in the paper
HF_BLEU = evaluate.load("bleu")


def self_consistency_legacy(generations: list[str]):
    # Source: More Agents Is All You Need (https://arxiv.org/abs/2402.05120)
    # compute BLEU score between all pairs of generations
    # return the generation with the highest cumulative BLEU score
    assert len(generations) != 0
    bleu_scores = [0 for _ in range(len(generations))]
    for i in range(len(generations)):
        for j in range(len(generations)):
            if i != j:
                bleu_scores[i] += HF_BLEU.compute(predictions=[generations[i]], references=[generations[j]])["bleu"]
    return generations[bleu_scores.index(max(bleu_scores))]


## New self-consistency code
def hf_bleu_score(prediction: str | None, reference: str | None) -> float:
    return HF_BLEU.compute(predictions=[prediction], references=[reference])["bleu"]


def nltk_bleu_score(prediction: str | None, reference: str | None) -> float:
    if prediction is None or reference is None:
        return None
    return sentence_bleu(
        [reference.split()],
        prediction.split(),
        smoothing_function=SmoothingFunction().method4,
    )


def self_consistency(
    generations: list[str], pair_similarity_fn: Callable[[str, str], float], return_score=False
) -> str | tuple[str, float] | None:
    # Source: More Agents Is All You Need (https://arxiv.org/abs/2402.05120)
    # return the generation with the highest cumulative similarity score
    if len(generations) == 0:
        return None

    bleu_scores = [
        sum(pair_similarity_fn(generations[i], generations[j]) for j in range(len(generations)) if i != j)
        for i in range(len(generations))
    ]
    max_score = max(bleu_scores)
    if return_score:
        return generations[bleu_scores.index(max_score)], max_score
    return generations[bleu_scores.index(max_score)]


def similarity_fn_wrapper(args):
    similarity_fn, idx, x = args[0], args[1], args[2]
    return idx, sum(similarity_fn(a, b) for b in x for a in x) / (len(x) ** 2)


def similarity_matrix_fn_wrapper(args):
    similarity_fn, idx, x = args[0], args[1], args[2]
    return idx, [[similarity_fn(a, b) for b in x] for a in x]


def generate_n_samples_sequence(max_n: int) -> list[int]:
    """Generate the sequence 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, ... until max_n."""
    repeating_seq = [1, 2, 5]
    sequence = []
    i = 1
    while i <= max_n:
        sequence.extend([i * r for r in repeating_seq if i * r <= max_n])
        i *= 10
    return sequence


def statistics_predictions(
    input_file: str,
    output_file: str | None = None,
    print_subsampled: bool = False,
) -> None:
    with jsonlines.open(input_file) as reader:
        predictions = list(reader)

    nb_samples = max(len(prediction["typecheck_results"]) for prediction in predictions)
    if print_subsampled:
        n_samples_sequence = generate_n_samples_sequence(nb_samples)
    else:
        n_samples_sequence = [nb_samples]

    nb_distincts = {i: [] for i in n_samples_sequence}
    nb_type_check = {i: [] for i in n_samples_sequence}
    for prediction in predictions:
        for i in n_samples_sequence:
            first_ith_typecheck_results = prediction["typecheck_results"][:i]
            nb_distincts[i].append(
                len({t for t, _ in first_ith_typecheck_results}) / len(first_ith_typecheck_results)
                if first_ith_typecheck_results
                else 0
            )
            nb_type_check[i].append(len([t for t, type_check in first_ith_typecheck_results if type_check]))

    # cumulated_self_bleu_scores = [None for _ in predictions]
    # num_processes = os.cpu_count()
    # data = [[t for t, _ in prediction["typecheck_results"] if isinstance(t, str)] for prediction in predictions]
    # with Pool(num_processes) as p:
    #     for idx, x in tqdm(
    #         p.imap_unordered(
    #             similarity_fn_wrapper,
    #             [(bleu_fn, idx, x) for idx, x in enumerate(data)],
    #             chunksize=1,
    #         ),
    #         total=len(predictions),
    #         desc="Computing predictions similarity",
    #     ):
    #         cumulated_self_bleu_scores[idx] = x

    nb_type_check_global = {i: sum(1 for t in nb_type_check[i] if t > 0) for i in n_samples_sequence}
    nb_type_check_sample = {i: sum(nb_type_check[i]) / i for i in n_samples_sequence}
    eval_agg_summary = {
        (f"{i}-samples" if i > 1 else "1-sample"): {
            "Distinct": {
                "total": sum(nb_distincts[i]),
                "normalized": sum(nb_distincts[i]) / len(predictions),
                "std": np.std(nb_distincts[i]),
            },
            # "Similarity (Self-BLEU)": {
            #     "total": sum(cumulated_self_bleu_scores),
            #     "normalized": sum(cumulated_self_bleu_scores) / len(predictions),
            #     "std": np.std(cumulated_self_bleu_scores),
            # },
            "Type-Check (sample)": {
                "total": nb_type_check_sample[i],
                "normalized": nb_type_check_sample[i] / len(predictions),
                "std": np.std(np.array(nb_type_check[i]) / i),
            },
            "Type-Check (global)": {
                "total": nb_type_check_global[i],
                "normalized": nb_type_check_global[i] / len(predictions),
            },
        }
        for i in n_samples_sequence
    }

    if output_file is not None:
        with open(output_file, "w") as f:
            f.write(json.dumps(eval_agg_summary, indent=4))

    table = Table(title="Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Total", style="green")
    table.add_column("Normalized", style="green")
    table.add_column("Std", style="green")
    for i_samples, values in eval_agg_summary.items():
        table.add_row(i_samples, "", "", "", style="bold grey70")
        for metric, values in values.items():
            value = f"{values['total']:.2f}"
            percentage = f"{values['normalized']*100:.2f}%"
            std = f"{values['std']:.2f}" if "std" in values else "-"
            table.add_row(metric, value, percentage, std)
        table.add_section()
    console.print(table)


def eval_predictions(
    prediction_file: str,
    output_file: str | None = None,
    bleu_fn: Callable[[str, str], float] = nltk_bleu_score,
) -> None:
    with jsonlines.open(prediction_file) as reader:
        predictions = list(reader)

    bleu_scores = []
    nb_type_check = 0

    for prediction in predictions:
        if len(prediction["predicted_formal_statement"]) == 0:
            continue
        nb_type_check += 1
        if prediction["formal_statement"]:
            bleu_scores.append(
                bleu_fn(
                    clean_theorem_string(prediction["predicted_formal_statement"], new_theorem_name="dummy"),
                    clean_theorem_string(prediction["formal_statement"], new_theorem_name="dummy"),
                )
            )

    bleu_sum = sum(bleu_scores)
    bleu_2_scores = [b**2 for b in bleu_scores]
    bleu_2_sum = sum(bleu_2_scores)
    eval_agg_summary = {
        "BLEU": {
            "total": bleu_sum,
            "normalized": bleu_sum / nb_type_check if nb_type_check != 0 else 0,
            "std": np.std(bleu_scores) if len(bleu_scores) > 1 else 0,
        },
        "TC-BLEU": {
            "total": bleu_sum,
            "normalized": bleu_sum / len(predictions),
            "std": np.std(bleu_scores + [0] * (len(predictions) - nb_type_check)) if len(bleu_scores) > 1 else 0,
        },
        "TC-BLEU²": {
            "total": bleu_2_sum,
            "normalized": bleu_2_sum / len(predictions),
            "std": np.std(bleu_2_scores) if len(bleu_2_scores) > 1 else 0,
        },
    }

    table = Table(title="Evaluation results")
    table.add_column("Metric", style="bold cyan")
    table.add_column("Total", style="bold green")
    table.add_column("Normalized", style="bold green")
    table.add_column("Std", style="bold green")
    for metric, values in eval_agg_summary.items():
        value = f"{values['total']:.2f}"
        percentage = f"{values['normalized']*100:.2f}%"
        std = f"{values['std']:.2f}" if "std" in values else "-"
        table.add_row(metric, value, percentage, std)
    console.print(table)

    if output_file is not None:
        with open(output_file, "w") as f:
            f.write(json.dumps(eval_agg_summary, indent=4))


def subsample_predictions(
    type_checked_predictions_file: str,
    sub_samples: list[int],
    sub_samples_filenames: list[str],
    shuffle: bool = True,
    seed: int = 42,
):
    random.seed(seed)

    with jsonlines.open(type_checked_predictions_file) as reader:
        predictions = list(reader)

    for j, sub_samples_filename in zip(sub_samples, sub_samples_filenames):
        subsample_content = []
        for prediction in predictions:
            if shuffle:
                random.shuffle(prediction["typecheck_results"])
            subsample_content.append(deepcopy(prediction))
            subsample_content[-1]["typecheck_results"] = prediction["typecheck_results"][:j]
            subsample_content[-1]["predicted_formal_statement"] = [
                t for t, type_check in prediction["typecheck_results"][:j] if type_check
            ]
        with jsonlines.open(sub_samples_filename, "w") as writer:
            writer.write_all(subsample_content)
