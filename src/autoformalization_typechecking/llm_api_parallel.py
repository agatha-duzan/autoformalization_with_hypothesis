# Inspired from: https://github.com/rafayaar/openai-multi-client/blob/b22dffacd7ac667d4d6c7ff39d3e4cf5edf56e15/openai_multi_client/__init__.py

import asyncio
import logging
import os
from collections import defaultdict
from dataclasses import dataclass
from threading import Thread
from typing import Any

import jsonlines
from aioprocessing import AioJoinableQueue, AioQueue
from openai import AsyncOpenAI
from openai.types import Completion
from openai.types.chat import ChatCompletion
from tenacity import (
    AsyncRetrying,
    RetryError,
    stop_after_attempt,
    wait_random_exponential,
)
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class Payload:
    endpoint: str
    data: dict
    metadata: dict | None
    max_retries: int
    retry_multiplier: float
    retry_max: float
    attempt: int = 0
    failed: bool = False
    response: Any = None
    callback: Any = None

    def call_callback(self):
        if self.callback:
            self.callback(self)


class OpenAIMultiClient:
    def __init__(
        self,
        concurrency: int = 10,
        max_retries: int = 10,
        wait_interval: float = 0,
        retry_multiplier: float = 1,
        retry_max: float = 60,
        endpoint: str | None = None,
        data_template: dict | None = None,
        metadata_template: dict | None = None,
        api_key: str | None = None,
        api_base_url: str | None = None,
        organization: str | None = None,
    ):
        self._endpoint = endpoint
        self._wait_interval = wait_interval
        self._data_template = data_template or {}
        self._metadata_template = metadata_template or {}
        self._max_retries = max_retries
        self._retry_multiplier = retry_multiplier
        self._retry_max = retry_max
        self._concurrency = concurrency
        self._loop = asyncio.new_event_loop()
        self._in_queue = AioJoinableQueue(maxsize=concurrency)
        self._out_queue = AioQueue(maxsize=concurrency)
        self._event_loop_thread = Thread(target=self._run_event_loop)
        self._event_loop_thread.start()
        self._aclient = AsyncOpenAI(api_key=api_key, base_url=api_base_url, organization=organization)

        for i in range(concurrency):
            asyncio.run_coroutine_threadsafe(self._worker(i), self._loop)

    def run_request_function(self, input_function, *args, stop_at_end=True, **kwargs):
        if stop_at_end:

            def f(*args, **kwargs):
                input_function(*args, **kwargs)
                self.close()
        else:
            f = input_function
        input_thread = Thread(target=f, args=args, kwargs=kwargs)
        input_thread.start()

    def _run_event_loop(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    async def _process_payload(self, payload: Payload) -> Payload:
        logger.debug(f"Processing {payload}")
        if payload.endpoint == "completions":
            payload.response = await self._aclient.completions.create(**payload.data)
        elif payload.endpoint == "chat.completions" or payload.endpoint == "chats":
            payload.response = await self._aclient.chat.completions.create(**payload.data)
        elif payload.endpoint == "embeddings":
            payload.response = await self._aclient.embeddings.create(**payload.data)
        elif payload.endpoint == "images":
            payload.response = await self._aclient.images.generate(**payload.data)
        else:
            raise ValueError(f"Unknown endpoint {payload.endpoint}")
        logger.debug(f"Processed {payload}")
        return payload

    async def _worker(self, i):
        while True:
            payload: Payload = await self._in_queue.coro_get()

            if payload is None:
                logger.debug(f"Exiting worker {i}")
                self._in_queue.task_done()
                break

            try:
                async for attempt in AsyncRetrying(
                    wait=wait_random_exponential(multiplier=payload.retry_multiplier, max=payload.retry_max),
                    stop=stop_after_attempt(payload.max_retries),
                ):
                    with attempt:
                        try:
                            payload.attempt = attempt.retry_state.attempt_number
                            payload = await self._process_payload(payload)
                            await self._out_queue.coro_put(payload)
                            self._in_queue.task_done()
                        except Exception:
                            logger.exception(f"Error processing {payload}")
                            raise
            except RetryError:
                payload.failed = True
                logger.error(f"Failed to process {payload}")
                await self._out_queue.coro_put(payload)
                self._in_queue.task_done()
            await asyncio.sleep(self._wait_interval)

    def close(self):
        try:
            for _ in range(self._concurrency):
                self._in_queue.put(None)
            self._in_queue.join()
            self._out_queue.put(None)
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._event_loop_thread.join()
        except Exception as e:
            logger.error(f"Error closing: {e}")

    def __iter__(self):
        return self

    def __next__(self):
        out: Payload = self._out_queue.get()
        if out is None:
            raise StopIteration
        out.call_callback()
        return out

    def request(
        self,
        data: dict,
        endpoint: str | None = None,
        metadata: dict | None = None,
        callback: Any = None,
        max_retries: int | None = None,
        retry_multiplier: float | None = None,
        retry_max: float | None = None,
    ):
        payload = Payload(
            endpoint=endpoint or self._endpoint,
            data={**self._data_template, **data},
            metadata={**self._metadata_template, **(metadata or {})},
            callback=callback,
            max_retries=max_retries or self._max_retries,
            retry_multiplier=retry_multiplier or self._retry_multiplier,
            retry_max=retry_max or self._retry_max,
        )
        self._in_queue.put(payload)

    def pull_all(self):
        for _ in self:
            pass


class OrderedPayload(Payload):
    put_counter: int

    def __init__(self, *args, put_counter, **kwargs):
        super().__init__(*args, **kwargs)
        self.put_counter = put_counter


class OpenAIMultiOrderedClient(OpenAIMultiClient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._put_counter = 0
        self._get_counter = 0
        self._get_cache = {}
        self._stopped = False

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            out: Payload | None = None
            if not self._stopped:
                out = self._out_queue.get()
            if out is None:
                self._stopped = True
                if self._get_counter == self._put_counter:
                    raise StopIteration
                else:
                    out = self._get_cache[self._get_counter]
                    del self._get_cache[self._get_counter]
                    self._get_counter += 1
                    out.call_callback()
                    return out

            data_counter = out.put_counter
            if data_counter == self._get_counter:
                self._get_counter += 1
                out.call_callback()
                return out
            self._get_cache[data_counter] = out
            if self._get_counter in self._get_cache:
                out = self._get_cache[self._get_counter]
                del self._get_cache[self._get_counter]
                self._get_counter += 1
                out.call_callback()
                return out

    def request(
        self,
        data: dict,
        endpoint: str | None = None,
        metadata: dict | None = None,
        callback: Any = None,
        max_retries: int | None = None,
        retry_multiplier: float | None = None,
        retry_max: float | None = None,
    ):
        payload = OrderedPayload(
            endpoint=endpoint or self._endpoint,
            data={**self._data_template, **data},
            metadata={**self._metadata_template, **(metadata or {})},
            callback=callback,
            max_retries=max_retries or self._max_retries,
            retry_multiplier=retry_multiplier or self._retry_multiplier,
            retry_max=retry_max or self._retry_max,
            put_counter=self._put_counter,
        )
        self._put_counter += 1
        self._in_queue.put(payload)


def process_requests(
    requests_file: str,
    output_file: str,
    client_params: dict,
    generation_params: dict,
    max_samples_per_request: int | None = None,
    show_progress: bool = True,
):
    with jsonlines.open(requests_file, "r") as f:
        all_requests = list(f)

    # if max_samples_per_request is None, we generate all samples in one request
    if max_samples_per_request is None:
        max_samples_per_request = generation_params["n"]

    nb_batches_per_request = generation_params["n"] // max_samples_per_request + (
        generation_params["n"] % max_samples_per_request > 0
    )

    previous_run_ids = defaultdict(set)
    if os.path.exists(output_file):
        # a (partial) generation has already been done, we need to skip the ones that have already been generated
        with jsonlines.open(output_file, "r") as f:
            previous_run = list(f)
        print("Previous run found with", len(previous_run), "items, skipping them")
        for item in previous_run:
            previous_run_ids[item["idx"]].add(item["batch_idx"] if "batch_idx" in item else 0)
        previous_run_idx = {
            item["idx"] for item in previous_run if len(previous_run_ids[item["idx"]]) == nb_batches_per_request
        }
        all_requests = [(idx, item) for idx, item in enumerate(all_requests) if idx not in previous_run_idx]
        # remove ids for which we have already generated all batches
        for idx in previous_run_idx:
            previous_run_ids.pop(idx)
    else:
        # otherwise we create the file
        print("No previous run found, creating output file")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with jsonlines.open(output_file, "w") as f:
            pass
        all_requests = list(enumerate(all_requests))

    if not all_requests:
        print("All requests have already been generated")
        return

    client = OpenAIMultiClient(**client_params)

    if "messages" in all_requests[0][1]:
        key = "messages"
        endpoint = "chats"
    elif "prompt" in all_requests[0][1]:
        key = "prompt"
        endpoint = "completions"
    else:
        raise ValueError("Unknown request format")

    def make_requests():
        for idx, request in all_requests:
            try:
                for i in range(0, generation_params["n"], max_samples_per_request):
                    if idx in previous_run_ids and i in previous_run_ids[idx]:
                        continue
                    local_generation_params = generation_params.copy()
                    local_generation_params["n"] = min(max_samples_per_request, generation_params["n"] - i)
                    local_generation_params["seed"] = request["seed"] + i
                    client.request(
                        data={key: request[key], **local_generation_params},
                        metadata={"metadata": request["metadata"], "idx": idx, "batch_idx": i},
                        endpoint=endpoint,
                    )
            except Exception as e:
                logger.error(f"Error making request {idx}: {e}")

    client.run_request_function(make_requests)

    count_tokens = {"input": 0, "output": 0}
    max_tokens = {"input": 0, "output": 0}
    if show_progress:
        total = len(all_requests) * nb_batches_per_request
        iterator = tqdm(client, total=total, desc="Generating completions")
    else:
        iterator = client

    result: Payload | None = None
    for result in iterator:
        if result is None or result.response is None:
            continue

        assert (isinstance(result.response, Completion) and endpoint == "completions") or (
            isinstance(result.response, ChatCompletion) and endpoint == "chats"
        )

        count_tokens["input"] += result.response.usage.prompt_tokens
        count_tokens["output"] += result.response.usage.completion_tokens
        max_tokens["input"] = max(max_tokens["input"], result.response.usage.prompt_tokens)
        max_tokens["output"] = max(max_tokens["output"], result.response.usage.completion_tokens)
        if show_progress:
            iterator.set_postfix_str(f"Input tokens: {count_tokens['input']}  Output tokens: {count_tokens['output']}")

        with jsonlines.open(output_file, "a") as f:
            if endpoint == "chats":
                responses = [choice.message.content for choice in result.response.choices]
            elif endpoint == "completions":
                responses = [choice.text for choice in result.response.choices]
            else:
                raise ValueError("Unknown endpoint")
            f.write(
                {
                    "idx": result.metadata["idx"],
                    "batch_idx": result.metadata["batch_idx"],
                    "metadata": result.metadata["metadata"],
                    "responses": responses,
                }
            )

    print(f"Max tokens used in a single request: input {max_tokens['input']}, output {max_tokens['output']}")
