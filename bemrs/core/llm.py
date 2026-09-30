import http.client
import json
import tiktoken
import logging
import concurrent.futures
import os
import time
from utils.utils import chat_completion, format_messages


class ConfigBeMRS:
    def __init__(self, model):
        self.model = model


class InterfaceAPI:
    def __init__(self, api_endpoint, api_key, model_LLM, debug_mode):
        self.api_endpoint = api_endpoint
        self.api_key = api_key
        self.model_LLM = model_LLM
        self.debug_mode = debug_mode
        self.n_trial = 5
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def cal_usage_LLM(self, lst_prompt, lst_completion, encoding_name="cl100k_base"):
        """Returns the number of tokens in a text string."""
        encoding = tiktoken.get_encoding(encoding_name)
        for i in range(len(lst_prompt)):
            for message in lst_prompt[i]:
                for key, value in message.items():
                    self.prompt_tokens += len(encoding.encode(value))

            self.completion_tokens += len(encoding.encode(lst_completion[i]))

    def get_response(self, prompt_content):
        return self.get_responses([prompt_content], max_workers=1)[0]

    def get_responses(self, prompt_contents, max_workers=None):
        """Generate one response per prompt while preserving input order."""
        prompt_contents = list(prompt_contents or [])
        if not prompt_contents:
            return []

        if max_workers is None:
            max_workers = int(os.environ.get("BEMRS_LLM_MAX_WORKERS", 6))
        max_workers = max(1, min(int(max_workers), len(prompt_contents)))

        def request_one(prompt_content):
            pre_messages = {"system": "", "user": prompt_content}
            cfg = ConfigBeMRS(model=self.model_LLM)
            messages = format_messages(cfg, pre_messages)
            response = chat_completion(
                1,
                [messages[1]],
                self.model_LLM,
                temperature=1.0,
            )
            return messages, response[0].message.content

        start = time.perf_counter()
        if max_workers == 1:
            results = [request_one(prompt) for prompt in prompt_contents]
        else:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=max_workers,
                thread_name_prefix="bemrs-llm",
            ) as executor:
                results = list(executor.map(request_one, prompt_contents))

        messages_batch = [messages for messages, _ in results]
        responses = [response for _, response in results]
        self.cal_usage_LLM(messages_batch, responses)
        logging.info(
            "[LLMBatch] prompts=%s workers=%s time=%.3fs "
            "prompt_tokens=%s completion_tokens=%s",
            len(prompt_contents),
            max_workers,
            time.perf_counter() - start,
            self.prompt_tokens,
            self.completion_tokens,
        )
        return responses
