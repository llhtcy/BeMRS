"""Loopback-only, unquantized Qwen transport for the existing InterfaceAPI."""
import json
import logging
import os
import time
from types import SimpleNamespace
import urllib.request

MODEL = 'Qwen/Qwen3-8B'
BASE_URL = 'http://127.0.0.1:8000/v1'
API_KEY = 'local-vllm'


def local_completion(n, messages, model, temperature, api_base=None, api_key=None):
    if model not in (MODEL, 'openai/' + MODEL):
        raise ValueError(f'Local vLLM refuses unexpected model {model}')
    if api_base is not None and api_base.rstrip('/') != BASE_URL:
        raise ValueError('Local vLLM refuses non-local endpoint overrides')
    choices = []
    # Match the old Ollama model's effective top_p/top_k/repetition settings.
    # InterfaceAPI continues supplying temperature=1.0.
    payload = dict(model=MODEL, messages=messages, stream=False, temperature=temperature,
                   top_p=0.95, top_k=20, min_p=0.0, repetition_penalty=1.0,
                   max_tokens=8192, chat_template_kwargs={'enable_thinking': False})
    for _ in range(n):
        for attempt in range(3):
            try:
                request = urllib.request.Request(BASE_URL + '/chat/completions',
                    data=json.dumps(payload).encode(),
                    headers={'Content-Type': 'application/json', 'Authorization': 'Bearer ' + API_KEY})
                opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
                with opener.open(request, timeout=1800) as response:
                    result = json.load(response)
                choice = result['choices'][0]
                content = choice.get('message', {}).get('content', '')
                if choice.get('finish_reason') != 'stop' or not isinstance(content, str) or not content.strip():
                    raise RuntimeError('vLLM returned truncated, incomplete, or empty content')
                usage = result.get('usage', {})
                logging.info('[LocalVLLM] model=%s dtype=BF16 tensor_parallel=4 enable_thinking=False '
                             'prompt_tokens=%s output_tokens=%s', MODEL,
                             usage.get('prompt_tokens'), usage.get('completion_tokens'))
                choices.append(SimpleNamespace(message=SimpleNamespace(content=content)))
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(2)
    return choices


def install_local_adapter():
    """Install before importing bemrs.core.llm; never fall back to cloud/Ollama."""
    for name in ('OPENAI_API_KEY', 'LITELLM_API_KEY'):
        os.environ[name] = API_KEY
    for name in ('OPENAI_BASE_URL', 'OPENAI_API_BASE', 'LITELLM_API_BASE'):
        os.environ[name] = BASE_URL
    for name in ('NO_PROXY', 'no_proxy'):
        os.environ[name] = ','.join(filter(None, [os.environ.get(name, ''), '127.0.0.1', 'localhost']))
    import utils.utils as shared
    shared.chat_completion = local_completion
