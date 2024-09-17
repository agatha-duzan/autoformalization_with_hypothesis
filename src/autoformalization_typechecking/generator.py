import json
import os
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime

import jsonlines
import yaml
from datasets import load_dataset

from autoformalization_typechecking.llm_api_parallel import process_requests
from autoformalization_typechecking.rag import RAG
from autoformalization_typechecking.utils import (
    ROOT_DIR,
    console,
    merge_consecutive_roles,
)


@dataclass
class GeneratorConfig:
    config_name: str
    model_name: str
    dataset_path: str
    dataset_input_column: str
    save_dir: str | None = None
    save_dir_suffix: str | None = None
    instructions: str | None = None
    few_shot_examples: list[tuple[str, str]] | str | None = None  # list of (input, output) tuples, or path to a file
    nb_few_shot_examples: int | None = None
    shuffle_examples: bool = False
    turn_based_few_shot: bool = True  # if True, each few-shot example is a turn-based conversation (only for chats)
    input_template: str | None = None
    dataset_split: str | None = None
    nb_passes: int = 1
    max_new_tokens: int = 512
    temperature: float = 0.0
    stop_words: list[str] | None = None
    api_base_url: str | None = None
    api_key: str | None = None
    endpoint: str = "chats"
    concurrency: int = 10
    max_samples_per_request: int | None = None  # vLLM can have issues when `nb_passes` is too large
    max_retries: int = 1
    seed: int = 42
    extra_body: dict | None = None
    rag_options: dict | None = None  # if None, RAG is not used


def load_generator_config(config_path: str) -> GeneratorConfig:
    """`config_path` can either point to a JSON or a YAML file"""
    if config_path.endswith(".json"):
        with open(config_path) as f:
            return GeneratorConfig(**json.load(f))
    elif config_path.endswith(".yaml"):
        with open(config_path) as f:
            return GeneratorConfig(**yaml.safe_load(f))


class Generator:
    def __init__(self, generator_config: GeneratorConfig, save_dir: str | None = None):
        self.config = generator_config

        self.set_save_dir(save_dir)

        # Check if the config file already exists, and if it does, if it's the same as the current config
        config_path = os.path.join(self.config.save_dir, "config.json")
        if os.path.exists(config_path):
            with open(config_path, encoding="utf-8") as f:
                existing_config = json.load(f)
            if existing_config != self.config.__dict__:
                console.print("[bold red]Warning:[/bold red] Previous generation used a different config file:")

                # Find which keys are different and ask the user if they want to continue
                different_keys = {
                    k: (existing_config[k], self.config.__dict__[k])
                    for k in existing_config
                    if k in self.config.__dict__ and existing_config[k] != self.config.__dict__[k]
                }
                if len(different_keys) > 0:
                    console.print("  [bold yellow]Per key differences[/bold yellow]:")
                    for k, (old_val, new_val) in different_keys.items():
                        console.print(f"  - [bold]{k}[/bold]")
                        console.print(f"    - [bold]Old[/bold]: {old_val}")
                        console.print(f"    - [bold]New[/bold]: {new_val}")

                # Find which keys are missing/added
                missing_keys = set(existing_config) - set(self.config.__dict__)
                added_keys = set(self.config.__dict__) - set(existing_config)
                if missing_keys:
                    console.print(f"  [bold yellow]Missing keys[/bold yellow]: {missing_keys}")
                if added_keys:
                    console.print(f"  [bold yellow]Added keys[/bold yellow]: {added_keys}")

                # Ask the user if they want to continue
                choice = input("Do you want to continue? It will overwrite the previous configuration file. (y/[n]): ")
                if choice.lower() != "y":
                    sys.exit()

        # Save the config
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(self.config.__dict__, f, indent=4)

        if self.config.rag_options is not None:
            self._rag = RAG()

    def set_save_dir(self, save_dir: str | None = None):
        if save_dir is None:
            save_dir = self.config.save_dir
            if save_dir is None:
                if os.path.exists(self.config.model_name):
                    save_dir = self.config.model_name
                else:
                    save_dir = f"results/{self.config.config_name}"

            if self.config.save_dir_suffix is None:
                save_dir = os.path.join(save_dir, "benchmark", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
            else:
                save_dir = os.path.join(
                    save_dir, "benchmark", self.config.save_dir_suffix, datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                )
        self.config.save_dir = os.path.join(ROOT_DIR, save_dir)
        os.makedirs(self.config.save_dir, exist_ok=True)

    def _load_dataset(self):
        # if there is an extension, we load the dataset with the extension
        if self.config.dataset_path.endswith(("json", "jsonl")):
            return load_dataset("json", data_files=self.config.dataset_path)["train"]
        return load_dataset(self.config.dataset_path)[self.config.dataset_split]

    def prepare_inputs(self):
        output_file = os.path.join(self.config.save_dir, "requests.jsonl")
        if os.path.exists(output_file):
            console.print(f"Requests already exist at {output_file}. Skipping preparation.")
            return

        dataset = self._load_dataset()

        # Load few-shot examples
        raw_few_shot_examples: list[tuple[str, str]] | None = None
        if isinstance(self.config.few_shot_examples, str):
            with jsonlines.open(self.config.few_shot_examples) as f:
                file_few_shot_examples: list[dict[str, str]] = list(f)
            input_key = self.config.dataset_input_column
            try:
                assert all(len(elt.keys()) == 2 for elt in file_few_shot_examples)
                assert all(input_key in elt for elt in file_few_shot_examples)
                output_key = [k for k in file_few_shot_examples[0].keys() if k != input_key][0]
                raw_few_shot_examples = [(ex[input_key], ex[output_key]) for ex in file_few_shot_examples]
            except (AssertionError, KeyError):
                raise ValueError(
                    f"Few-shot examples should be a list of dictionaries with 2 keys, including `{input_key}` for the inputs."
                )
        else:
            raw_few_shot_examples = self.config.few_shot_examples

        # Prepare few-shot examples
        few_shot_examples: list[tuple[str, str]] | None = None
        if raw_few_shot_examples is not None:
            few_shot_examples = []
            for few_shot_example_raw_input, few_shot_example_output in raw_few_shot_examples:
                few_shot_examples.append((self._format_input(few_shot_example_raw_input), few_shot_example_output))

        random.seed(self.config.seed)
        all_requests = []
        for sample in dataset:
            # Possible MODIF here: to iterate on just 1 or few samples
            # can add break statement to shorten process

            # when shuffling the examples, we do it for each sample
            # we therefore can't use the "n" parameter of the API and need to make multiple requests
            nb_requests = self.config.nb_passes if self.config.shuffle_examples else 1

            # Prepare the sample input
            sample_input = self._format_input(sample[self.config.dataset_input_column])

            for i in range(nb_requests):
                messages = []

                # Base instructions
                if self.config.instructions is not None:
                    system_role = "system"
                    if self.config.api_base_url is not None:
                        # if not OpenAI API, we fallback to the user role as some models don't support system role
                        system_role = "user"
                    messages.append({"role": system_role, "content": self.config.instructions})

                # Few-shot examples
                if few_shot_examples is not None:
                    if self.config.shuffle_examples:
                        random.shuffle(few_shot_examples)

                    # Select a subset of few-shot examples if requested
                    nb_few_shot_examples = self.config.nb_few_shot_examples
                    if nb_few_shot_examples is None:
                        nb_few_shot_examples = len(few_shot_examples)
                    local_few_shot_examples = few_shot_examples[:nb_few_shot_examples]

                    if self.config.turn_based_few_shot and self.config.endpoint == "chats":
                        for few_shot_example_input, few_shot_example_output in local_few_shot_examples:
                            messages.append({"role": "user", "content": few_shot_example_input})
                            messages.append({"role": "assistant", "content": few_shot_example_output})
                    else:
                        concatenated_few_shot = "\n\n".join(
                            [
                                few_shot_example_input + "\n" + few_shot_example_output
                                for few_shot_example_input, few_shot_example_output in local_few_shot_examples
                            ]
                        )
                        messages.append({"role": "user", "content": concatenated_few_shot})

                messages.append({"role": "user", "content": sample_input})

                # Format the request depending on the endpoint
                messages = merge_consecutive_roles(messages)
                if self.config.endpoint == "chats":
                    all_requests.append({"messages": messages, "metadata": {**sample}, "seed": self.config.seed + i})

                elif self.config.endpoint == "completions":
                    prompt = "\n\n".join([m["content"] for m in messages])
                    all_requests.append({"prompt": prompt, "metadata": {**sample}, "seed": self.config.seed + i})

                else:
                    raise ValueError(f"Unknown endpoint: {self.config.endpoint}")

        os.makedirs(self.config.save_dir, exist_ok=True)
        with jsonlines.open(output_file, "w") as f:
            f.write_all(all_requests)

    # TODO : prepare_second_stage_inputs(self):
    # Read intermediate predictions
    # Prepare new requests for the second stage : append intermediate predictions to input
    # How to deal with multiple intermediate predictions?
    # Save new requests
    # Do it in prepare_inputs directly

    def _format_input(self, raw_input: str) -> str:
        formatted_input = self.config.input_template.format(**{self.config.dataset_input_column: raw_input})

        # Add RAG content if requested
        if self.config.rag_options is not None:
            rag_content = "Potentially relevant content for the next input:\n"
            rag_content += "\n".join(
                [doc.doc for doc in self._rag.get_top_k_theorems(raw_input, **self.config.rag_options)]
            )
            return rag_content + "\n\n" + formatted_input

        return formatted_input

    # MODIF here: add args to accept requests_file and output_file 
    # (different files for each stage)
    def _inference(self, show_progress: bool) -> None:
        process_requests(
            requests_file=os.path.join(self.config.save_dir, "requests.jsonl"),
            output_file=os.path.join(self.config.save_dir, "output.jsonl"),
            client_params={
                "api_key": self.config.api_key,
                "api_base_url": self.config.api_base_url,
                "concurrency": self.config.concurrency,
                "max_retries": self.config.max_retries,
            },
            generation_params={
                "model": self.config.model_name,
                "n": 1 if self.config.shuffle_examples else self.config.nb_passes,
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_new_tokens,
                "stop": self.config.stop_words,
                "extra_body": self.config.extra_body,
            },
            max_samples_per_request=self.config.max_samples_per_request,
            show_progress=show_progress,
        )

    # MODIF here: extend method for two stage generation
    def compute_predictions(self, show_progress: bool = True) -> None:
        self._inference(show_progress)
        self.export_predictions()

        # self.prepare_second_stage_inputs() # TODO
        
        # second round of inference + export predictions

    # MODIF here: add args for input and output files
    def export_predictions(self) -> None:
        with jsonlines.open(os.path.join(self.config.save_dir, "output.jsonl"), "r") as f:
            raw_predictions = list(f)

        # reorder the predictions by idx
        raw_predictions = sorted(raw_predictions, key=lambda x: x["idx"])

        # merge batches with the same idx
        merged_predictions = []
        prev_idx = None
        for prediction in raw_predictions:
            if prev_idx == prediction["idx"]:
                merged_predictions[-1]["responses"].extend(prediction["responses"])
            else:
                merged_predictions.append(prediction)
            prev_idx = prediction["idx"]

        # merge predictions with same id
        id_to_idx = defaultdict(list)
        for idx, prediction in enumerate(merged_predictions):
            id_to_idx[prediction["metadata"]["id"]].append(idx)
        predictions = []
        for id, idxs in id_to_idx.items():
            prediction = merged_predictions[idxs[0]]
            for idx in idxs[1:]:
                prediction["responses"].extend(merged_predictions[idx]["responses"])
            predictions.append(prediction)

        results = []
        for prediction in predictions:
            assert len(prediction["responses"]) == self.config.nb_passes, "Found a prediction with missing passes"
            results.append(
                {
                    "id": prediction["metadata"]["id"],
                    "input": prediction["metadata"][self.config.dataset_input_column],
                    "prediction": prediction["responses"],
                }
            )

        # print number of predictions
        console.print(f"Number of predictions: {len(results)}")

        with open(os.path.join(self.config.save_dir, "predictions.json"), "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4)
