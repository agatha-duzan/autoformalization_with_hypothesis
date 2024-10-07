import argparse
import os
import random
from collections import Counter
from functools import partial

from autoformalization_typechecking.generator import (
    Generator,
    load_generator_config,
)
from autoformalization_typechecking.lean_repl import (
    run_typechecking_filter,
    run_typechecking_filter_multiprocess,
)
from autoformalization_typechecking.predictions_processing import (
    preprocess_predictions,
    select_predictions,
)
from autoformalization_typechecking.utils import (
    console,
    eval_predictions,
    hf_bleu_score,
    nltk_bleu_score,
    self_consistency,
    statistics_predictions,
)


def majority_voting(x):
    random.shuffle(x)
    return Counter(x).most_common(1)[0][0]


if __name__ == "__main__":
    # get config file from command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--continue-run-dir", type=str, default=None)
    args = parser.parse_args()

    # load the generator config
    eval_config = load_generator_config(args.config)
    # can change instructions here
    evaluator = Generator(eval_config, args.continue_run_dir)
    console.print(f"Save directory: {evaluator.config.save_dir}")

    console.print()
    console.rule("[bold]Sampling[/bold]")
    with console.status("[bold]Preparing inputs[/bold]"):
        evaluator.prepare_inputs()
    evaluator.compute_predictions()
    console.print("[bold]Predictions successfully computed[/bold]")

    output_dir = evaluator.config.save_dir

    # transform the prediction file, clean predictions, and add src header for the Lean type-checker
    output_filename = f"{evaluator.config.nb_passes}pass"
    if os.path.exists(os.path.join(output_dir, f"{output_filename}.jsonl")):
        console.print("[bold]Predictions already preprocessed[/bold]")
    else:
        preprocess_predictions(
            raw_prediction_file=os.path.join(output_dir, "predictions.json"),
            output_file=os.path.join(output_dir, f"{output_filename}.jsonl"),
            split=evaluator.config.dataset_split,
            input_dataset_file=evaluator.config.dataset_path,
        )
        console.print("[bold]Predictions preprocessed[/bold]")

    console.print()
    console.rule("[bold]Filtering[/bold]")
    # run the type-checker on the predictions
    output_typechecked_filename = f"{output_filename}_typechecked"
    if os.path.exists(os.path.join(output_dir, f"{output_typechecked_filename}.jsonl")):
        console.print("[bold]Type-checking already computed[/bold]")
    else:
        run_typechecking_filter_multiprocess(
            prediction_file=os.path.join(output_dir, f"{output_filename}.jsonl"),
            output_file=os.path.join(output_dir, f"{output_typechecked_filename}.jsonl"),
        )
        console.print("[bold]Type-checking computed[/bold]")

    # compute statistics
    statistics_predictions(
        input_file=os.path.join(output_dir, f"{output_typechecked_filename}.jsonl"),
        output_file=os.path.join(output_dir, f"{output_typechecked_filename}_statistics.json"),
    )

    console.print()
    console.rule("[bold]Selection[/bold]")

    if eval_config.nb_passes == 1:
        output_file = os.path.join(output_dir, f"{output_typechecked_filename}_final.jsonl")
        if not os.path.exists(output_file):
            select_predictions(
                prediction_file=os.path.join(output_dir, f"{output_typechecked_filename}.jsonl"),
                output_file=output_file,
                select_fn=random.choice,  # there is only one prediction, so random.choice will select it (and it's pickable)
            )

        with console.status("[bold]Evaluating predictions[/bold]"):
            eval_predictions(
                prediction_file=output_file,
                output_file=os.path.join(output_dir, f"{output_typechecked_filename}_final_eval_results.json"),
                bleu_fn=nltk_bleu_score,
            )

    else:
        # select one prediction using different strategies
        selection_methods = [
            (method, fn, f"{output_typechecked_filename}_{method.lower().replace(' ', '_')}")
            for method, fn in [
                ("Random", random.choice),
                ("Majority voting", majority_voting),
                ("Self BLEU", partial(self_consistency, pair_similarity_fn=nltk_bleu_score)),
            ]
        ]
        for method, select_fn, output_prefix in selection_methods:
            console.print(f"\nMethod: {method}")
            output_file = os.path.join(output_dir, f"{output_prefix}.jsonl")
            if os.path.exists(output_file):
                console.print("Predictions already selected")
                continue
            select_predictions(
                prediction_file=os.path.join(output_dir, f"{output_typechecked_filename}.jsonl"),
                output_file=output_file,
                select_fn=select_fn,
            )

        # evaluate the selected predictions
        with console.status("[bold]Evaluating predictions[/bold]"):
            for method, _, output_prefix in selection_methods:
                console.print(f"Method: {method}")
                eval_predictions(
                    prediction_file=os.path.join(output_dir, f"{output_prefix}.jsonl"),
                    output_file=os.path.join(output_dir, f"{output_prefix}_eval_results.json"),
                    bleu_fn=nltk_bleu_score,
                )
