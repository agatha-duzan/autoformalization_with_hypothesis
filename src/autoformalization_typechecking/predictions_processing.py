import json
import os
from multiprocessing.pool import Pool

import jsonlines
from datasets import load_dataset
from tqdm import tqdm

from autoformalization_typechecking.utils import DATA_DIR, clean_theorem_string


def preprocess_predictions(
    raw_prediction_file: str,
    output_file: str,
    input_dataset_file: str,
    split: str | None = None,
    lean_version: str = "lean4",
):
    with open(raw_prediction_file) as f:
        raw_predictions = json.load(f)
    id_to_prediction = {pred["id"]: pred["prediction"] for pred in raw_predictions}

    processed_predictions = []

    if "proofnet" in input_dataset_file.lower():
        proofnet = load_dataset("json", data_files=os.path.join(DATA_DIR, f"proofnet_lean3-4_{split}.jsonl"))["train"]
        for proofnet_row in proofnet:
            exname = proofnet_row["id"][proofnet_row["id"].index("|") + 1 :]
            prediction = id_to_prediction.get(proofnet_row["id"], [])
            processed_predictions.append(
                {
                    "id": proofnet_row["id"],
                    "nl_statement": proofnet_row["nl_statement"],
                    "nl_proof": proofnet_row["nl_proof"],
                    "formal_statement": proofnet_row[f"{lean_version}_statement"],
                    "predicted_formal_statement": [clean_theorem_string(pred, exname) for pred in prediction],
                    "src_header": proofnet_row[f"{lean_version}_src_header"],
                }
            )
    else:
        dataset = load_dataset("json", data_files=input_dataset_file)["train"]
        with open(os.path.join(DATA_DIR, "import_mathlib.lean")) as f:
            default_src_header = f.read()
        for dataset_row in dataset:
            prediction = id_to_prediction.get(dataset_row["id"], [])
            processed_predictions.append(
                {
                    "id": dataset_row["id"],
                    "nl_statement": dataset_row["nl_statement"],
                    "formal_statement": dataset_row["formal_statement"],
                    "predicted_formal_statement": [clean_theorem_string(pred) for pred in prediction],
                    "src_header": default_src_header,
                }
            )

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with jsonlines.open(output_file, "w") as f:
        f.write_all(processed_predictions)


def select_fn_wrapper(args):
    select_fn, idx, x = args[0], args[1], args[2]

    if isinstance(x["predicted_formal_statement"], list) and len(x["predicted_formal_statement"]) >= 1:
        x["predicted_formal_statement"] = select_fn(x["predicted_formal_statement"])

    return idx, x


def select_predictions(
    prediction_file: str,
    output_file: str,
    select_fn,
    num_processes: int = os.cpu_count(),
    progress_bar: bool = True,
) -> None:
    with jsonlines.open(prediction_file, "r") as f:
        data = list(f)

    res = [None for _ in data]
    with Pool(num_processes) as p:
        iterator = p.imap_unordered(
            select_fn_wrapper,
            [(select_fn, idx, x) for idx, x in enumerate(data)],
            chunksize=1,
        )
        if progress_bar:
            iterator = tqdm(iterator, total=len(data))
        for idx, x in iterator:
            res[idx] = x

    with jsonlines.open(output_file, "w") as f:
        f.write_all(res)
