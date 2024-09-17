import json
import os
import random

import jsonlines
from autoformalization_typechecking.utils import DATA_DIR, clean_theorem_string, console


def extract_pda_dataset():
    pda_raw_dir = os.path.join(DATA_DIR, "pda", "raw")
    pda_processed_dir = os.path.join(DATA_DIR, "pda", "stmt_only")

    for filename in os.listdir(pda_raw_dir):
        if not filename.endswith(".json"):
            continue

        with console.status(f"Processing {filename}"):
            with open(os.path.join(pda_raw_dir, filename)) as f:
                data = json.load(f)

            res = []
            nb_null_formal = 0
            for id, example in enumerate(data):
                try:
                    formal_statement = clean_theorem_string(example["output"])
                    if formal_statement is not None:
                        formal_statement = formal_statement.strip("sorry").strip()
                    nb_null_formal += not formal_statement
                    begin_nl_statement = example["input"].find("# Statement:") + len("# Statement:")
                    end_nl_statement = example["input"].find("# Proof:")
                    nl_statement = example["input"][begin_nl_statement:end_nl_statement].strip()
                    res.append(
                        {
                            "id": id,
                            "formal_statement": formal_statement,
                            "nl_statement": nl_statement,
                        }
                    )
                except Exception as e:
                    console.print(f"Error in {filename} example {id}: {e}")

            with jsonlines.open(
                os.path.join(pda_processed_dir, os.path.splitext(filename)[0] + ".jsonl"), "w"
            ) as writer:
                writer.write_all(res)
        console.print(f"Finished processing {filename}")
        console.print(f"Number of null formal statements: {nb_null_formal} / {len(data)}")
        console.print()


def subsample_pda_dataset(nb_samples: int):
    pda_stmt_only_dir = os.path.join(DATA_DIR, "pda", "stmt_only")
    pda_subsampled_dir = os.path.join(DATA_DIR, "pda", "subsampled")

    for filename in os.listdir(pda_stmt_only_dir):
        if not filename.endswith(".jsonl"):
            continue

        with console.status(f"Subsampling {filename}"):
            with jsonlines.open(os.path.join(pda_stmt_only_dir, filename)) as reader:
                data = list(reader)

            indices = list(range(len(data)))
            random.shuffle(indices)
            indices = indices[:nb_samples]

            res = [data[i] for i in indices]

            os.makedirs(pda_subsampled_dir, exist_ok=True)
            with jsonlines.open(os.path.join(pda_subsampled_dir, filename), "w") as writer:
                writer.write_all(res)

            with open(os.path.join(pda_subsampled_dir, os.path.splitext(filename)[0] + "_indices.txt"), "w") as f:
                f.write("\n".join(map(str, indices)))


if __name__ == "__main__":
    subsample_pda_dataset(50)
