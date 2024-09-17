import gc
import json
import os
from collections import Counter
from multiprocessing.pool import Pool

import jsonlines
import pexpect
import psutil
from tqdm import tqdm

from autoformalization_typechecking.utils import REPL_DIR, console


class LeanServer:
    # Inspired from: https://github.com/zhangir-azerbayev/repl/blob/bddf452deda0df2240b248e651bcc37fb8e59d01/pylean/pylean/__init__.py
    def __init__(self):
        os.chdir(REPL_DIR)
        self.proc = pexpect.spawn("lake exe repl", cwd=REPL_DIR, encoding="utf-8")

    def run_code(self, code, env=None, verbose=False):
        if env is not None:
            command = json.dumps(dict(cmd=code, env=env), ensure_ascii=False)
        else:
            command = json.dumps(dict(cmd=code), ensure_ascii=False)

        if verbose:
            print(command)

        self.proc.sendline(command)
        self.proc.expect_exact(command + "\r\n")
        self.proc.sendline()
        self.proc.expect_exact("\r\n")
        try:
            index = self.proc.expect(r'env": \d+\}', timeout=20)
            output = self.proc.before + self.proc.match.group()

            # clean up the output
            output = output.replace("\r\n", "\n")
            output = output[output.rfind("\r") + 1 :]
            if verbose:
                print(output)

            return json.loads(output)
        except pexpect.exceptions.TIMEOUT:
            raise pexpect.exceptions.TIMEOUT

    def __del__(self):
        self.proc.terminate(force=True)


def theorem_type_check(theorem_code: str, server: LeanServer, env: int | None = None, verbose: bool = False):
    if theorem_code is None or theorem_code == "":
        return False

    try:
        res = server.run_code(theorem_code, env=env, verbose=verbose)
    except Exception as e:
        console.log(f"Error while typechecking: {e}")
        return False

    try:
        if len(res["messages"]) == 1:
            return res["messages"][0]["severity"] == "warning"
        else:
            return False
    except Exception as e:
        console.log(f"Error while processing typecheck results:\n{e}\n{res=}")
        return False


def run_typechecking_filter(
    prediction_file: str,
    output_file: str,
):
    with jsonlines.open(prediction_file) as reader:
        predictions = list(reader)

    # group predictions by common src_header to avoid reloading the same imports
    predictions = sorted(predictions, key=lambda x: x["src_header"])

    prev_src_header = None
    for prediction in tqdm(predictions, desc="Checking theorems"):
        curr_src_header = prediction["src_header"]
        theorem = prediction["predicted_formal_statement"]

        if isinstance(theorem, str):
            theorem = [theorem]

        if curr_src_header != prev_src_header:
            server = LeanServer()
            gc.collect()
            res = server.run_code(curr_src_header)
            prev_src_header = curr_src_header

        # transform the list into a Counter to avoid checking duplicates multiple times
        theorems = Counter(theorem)
        typechecks = {}
        for t, nb_occurrences in theorems.items():
            if psutil.virtual_memory().percent > 80:
                print("Memory usage is too high, reloading the server")
                server = LeanServer()
                gc.collect()
                res = server.run_code(curr_src_header)

            typechecks[t] = theorem_type_check(t, server, res["env"])

        typecheck_results = []
        predicted_formal_statement_typechecking = []
        for i, t in enumerate(theorem):
            typecheck_results.append((t, typechecks[t]))
            if typechecks[t]:
                predicted_formal_statement_typechecking.append(t)

        prediction["typecheck_results"] = typecheck_results
        prediction["predicted_formal_statement"] = predicted_formal_statement_typechecking

    with jsonlines.open(output_file, "w") as writer:
        writer.write_all(predictions)


def filter_fn_wrapper(args):
    idx, prediction = args[0], args[1]
    src_header, theorems = prediction["src_header"], prediction["predicted_formal_statement"]
    if isinstance(theorems, str):
        theorems = [theorems]

    server = LeanServer()
    gc.collect()
    res = server.run_code(src_header)

    # iterate over deduplicated theorems and type-check them
    type_check = {}
    for theorem in set(theorems):
        if psutil.virtual_memory().percent > 80:
            print("Memory usage is too high, reloading the server")
            server = LeanServer()
            gc.collect()
            res = server.run_code(src_header)

        type_check[theorem] = theorem_type_check(theorem, server, res["env"])

    # store the type-check results and the well-typed theorems
    # /!\ we keep duplicates in the output for statistical reasons in the self-consistency step
    type_check_results = []
    well_typed_theorems = []
    for theorem in theorems:
        type_check_results.append((theorem, type_check[theorem]))
        if type_check[theorem]:
            well_typed_theorems.append(theorem)

    prediction["typecheck_results"] = type_check_results
    prediction["predicted_formal_statement"] = well_typed_theorems

    return idx, prediction


def run_typechecking_filter_multiprocess(
    prediction_file: str,
    output_file: str,
    num_processes: int = os.cpu_count(),
):
    with jsonlines.open(prediction_file) as reader:
        predictions = list(reader)

    res = [None for _ in predictions]
    with Pool(num_processes) as p:
        iterator = p.imap_unordered(
            filter_fn_wrapper,
            [(idx, prediction) for idx, prediction in enumerate(predictions)],
            chunksize=1,
        )
        for idx, prediction in tqdm(iterator, desc="Checking theorems", total=len(predictions)):
            res[idx] = prediction

    with jsonlines.open(output_file, "w") as writer:
        writer.write_all(res)
