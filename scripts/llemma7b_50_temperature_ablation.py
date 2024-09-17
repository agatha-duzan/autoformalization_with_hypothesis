import json
import os
import random
from collections import Counter
from datetime import datetime
from functools import partial

from autoformalization_typechecking.dump_predictions import (
    preprocess_predictions,
    select_predictions,
)
from autoformalization_typechecking.eval_api import EvalConfig, Evaluator
from autoformalization_typechecking.lean_repl import (
    compute_topn_typecheck_from_precomputed_file,
    run_typechecking_filter,
    run_typechecking_filter_multiprocess,
)
from autoformalization_typechecking.utils import (
    ROOT_DIR,
    eval_predictions,
    hf_bleu_score,
    nltk_bleu_score,
    self_consistency,
    subsample_predictions,
)
from rich.console import Console


def majority_voting(x):
    random.shuffle(x)
    return Counter(x).most_common(1)[0][0]


if __name__ == "__main__":
    console = Console()

    # load eval config from JSON file
    with open(os.path.join(ROOT_DIR, "configs", "llemma-7b-50-temperature", "config_valid.json"), "r") as f:
        eval_config_dict: EvalConfig = json.load(f)
    eval_config = EvalConfig(**eval_config_dict)

    date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    for temperature in [0.3, 0.5, 0.7, 0.9, 1.1, 1.3]:
        eval_config.temperature = temperature

        console.print(f"[bold]Generating predictions with temperature {temperature}...[/bold]")
        evaluator = Evaluator(eval_config)
        save_dir = f"results/llemma-7b-50-temperature/benchmark/ProofNet_validation_autoformalization/{date}/temperature_{temperature}"
        os.makedirs(save_dir, exist_ok=True)
        evaluator.set_save_dir(save_dir)
        console.print(f"Set save directory to {evaluator.config.save_dir}")
        console.print("Preparing inputs...")
        evaluator.prepare_inputs()
        evaluator.compute_predictions()
        console.print("Predictions computed.")

        output_dir = evaluator.config.save_dir

        # transform the prediction file and add src header for the Lean type-checker
        output_filename = f"{evaluator.config.nb_passes}pass"
        preprocess_predictions(
            raw_prediction_file=os.path.join(output_dir, "predictions.json"),
            output_file=os.path.join(output_dir, f"{output_filename}.jsonl"),
            split=evaluator.config.dataset_split,
        )

        # run the type-checker on the predictions
        output_type_checked_filename = f"{output_filename}_typechecked"
        console.print("\n[bold]Running type-checker...[/bold]")
        run_typechecking_filter_multiprocess(
            prediction_file=os.path.join(output_dir, f"{output_filename}.jsonl"),
            output_file=os.path.join(output_dir, f"{output_type_checked_filename}.jsonl"),
        )

        # subsample the predictions
        console.print("\n[bold]Sub-sampling predictions...[/bold]")
        sub_samples = [1, 5, 10, 20, 50]
        sub_samples_filenames = [f"{i}pass_typechecked" for i in sub_samples]
        subsample_predictions(
            type_checked_predictions_file=os.path.join(output_dir, f"{output_type_checked_filename}.jsonl"),
            sub_samples=sub_samples,
            sub_samples_filenames=[os.path.join(output_dir, f"{f}.jsonl") for f in sub_samples_filenames],
        )

        for i, f in zip(sub_samples, sub_samples_filenames):
            # select one prediction using different strategies
            console.print(f"\n[bold]Selecting predictions for n={i} sub-samples...[/bold]")
            console.print("Random selection")
            select_predictions(
                prediction_file=os.path.join(output_dir, f"{output_type_checked_filename}.jsonl"),
                output_file=os.path.join(output_dir, f"{output_type_checked_filename}_random.jsonl"),
                select_fn=random.choice,
            )

            console.print("\nMajority voting")
            select_predictions(
                prediction_file=os.path.join(output_dir, f"{output_type_checked_filename}.jsonl"),
                output_file=os.path.join(output_dir, f"{output_type_checked_filename}_majority_voting.jsonl"),
                select_fn=majority_voting,
            )

            console.print("\nSelf-BLEU")
            select_predictions(
                prediction_file=os.path.join(output_dir, f"{output_type_checked_filename}.jsonl"),
                output_file=os.path.join(output_dir, f"{output_type_checked_filename}_self_bleu.jsonl"),
                # select_fn=partial(self_consistency, pair_similarity_fn=hf_bleu_score),
                select_fn=partial(self_consistency, pair_similarity_fn=nltk_bleu_score),
            )

            # evaluate the predictions
            console.print(f"\n[bold]Evaluating predictions for n={i} sub-samples...[/bold]")
            console.print("Random selection")
            eval_predictions(
                prediction_file=os.path.join(output_dir, f"{output_type_checked_filename}_random.jsonl"),
                bleu_fn=nltk_bleu_score,
            )

            console.print("\nMajority voting")
            eval_predictions(
                prediction_file=os.path.join(output_dir, f"{output_type_checked_filename}_majority_voting.jsonl"),
                bleu_fn=nltk_bleu_score,
            )

            console.print("\nSelf-BLEU")
            eval_predictions(
                prediction_file=os.path.join(output_dir, f"{output_type_checked_filename}_self_bleu.jsonl"),
                bleu_fn=nltk_bleu_score,
            )
