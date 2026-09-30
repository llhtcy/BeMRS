import os
import logging
import concurrent.futures
import time
import re
import inspect
from urllib.parse import urlsplit

def _sanitize_no_proxy_for_legacy_httpx():
    """Drop unbracketed IPv6 entries that HTTPX 0.27 parses as bad ports."""
    for env_name in ('NO_PROXY', 'no_proxy'):
        raw = os.environ.get(env_name)
        if not raw:
            continue
        entries = [entry.strip() for entry in raw.split(',') if entry.strip()]
        compatible = [entry for entry in entries if not entry.startswith('::')]
        if compatible != entries:
            os.environ[env_name] = ','.join(compatible)

def _normalize_proxy_scheme_for_httpx():
    """Normalize SOCKS URLs accepted by the installed HTTPX release.

    OpenSSH exposes this tunnel as a SOCKS5 proxy.  Some users write the
    curl-style ``socks5h://`` scheme to request proxy-side DNS resolution,
    but HTTPX 0.26/0.27 rejects that scheme during client construction.  The
    SOCKS transport still resolves through the proxy, so ``socks5://`` is the
    compatible spelling here.
    """
    proxy_names = ('BEMRS_DOWNLOAD_PROXY', 'HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy')
    for env_name in proxy_names:
        raw = os.environ.get(env_name)
        if not raw:
            continue
        stripped = raw.strip()
        if stripped.lower().startswith('socks5h://'):
            os.environ[env_name] = 'socks5://' + stripped[len('socks5h://'):]

def _propagate_download_proxy():
    """Preserve the historical behavior of BEMRS_DOWNLOAD_PROXY.

    The application uses BEMRS_DOWNLOAD_PROXY as its single proxy setting, but
    LiteLLM/httpx reads the standard proxy environment variables during
    import.  Copy the configured value before importing LiteLLM so both code
    paths use the same endpoint.
    """
    raw = os.environ.get('BEMRS_DOWNLOAD_PROXY', '')
    if not raw.strip():
        return
    proxy = raw.strip()
    if proxy.lower().startswith('socks5h://'):
        proxy = 'socks5://' + proxy[len('socks5h://'):]
        os.environ['BEMRS_DOWNLOAD_PROXY'] = proxy
    elif not os.environ.get('BEMRS_DOWNLOAD_PROXY'):
        os.environ['BEMRS_DOWNLOAD_PROXY'] = proxy
    for env_name in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy'):
        os.environ[env_name] = proxy

def _log_proxy_configuration():
    """Log proxy presence without exposing credentials or full URLs."""
    proxy_names = ('BEMRS_DOWNLOAD_PROXY', 'HTTPS_PROXY', 'https_proxy', 'ALL_PROXY', 'all_proxy')
    for env_name in proxy_names:
        raw = os.environ.get(env_name)
        if not raw:
            continue
        try:
            parsed = urlsplit(raw)
            scheme = parsed.scheme.lower() or 'unknown'
            host = parsed.hostname or 'unknown'
            port = parsed.port
            endpoint = f'{host}:{port}' if port is not None else host
        except ValueError:
            scheme = 'invalid'
            endpoint = 'unparseable'
        logging.info('[NetworkProxy] source=%s configured=True scheme=%s endpoint=%s', env_name, scheme, endpoint)
        return
    logging.info('[NetworkProxy] configured=False')
_sanitize_no_proxy_for_legacy_httpx()
_normalize_proxy_scheme_for_httpx()
_propagate_download_proxy()
from litellm import completion
_log_proxy_configuration()

def file_to_string(filename):
    with open(filename, 'r') as file:
        return file.read()

def filter_traceback(s):
    lines = s.split('\n')
    filtered_lines = []
    for i, line in enumerate(lines):
        if line.startswith('Traceback'):
            for j in range(i, len(lines)):
                if 'Set the environment variable HYDRA_FULL_ERROR=1' in lines[j]:
                    break
                filtered_lines.append(lines[j])
            return '\n'.join(filtered_lines)
    return ''

def block_until_running(stdout_filepath, log_status=False, iter_num=-1, response_id=-1):
    while True:
        log = file_to_string(stdout_filepath)
        if len(log) > 0:
            if log_status and 'Traceback' in log:
                logging.info(f'Iteration {iter_num}: Code Run {response_id} execution error!')
            else:
                logging.info(f'Iteration {iter_num}: Code Run {response_id} successful!')
            break

def chat_completion(n: int, messages: list[dict], model: str, temperature: float, api_base: str=None, api_key: str=None) -> list[dict]:
    """
    Generate n responses using OpenAI Chat Completions API
    """
    resolved_api_base = api_base or os.environ.get('LITELLM_API_BASE') or os.environ.get('OPENAI_API_BASE')
    resolved_api_key = api_key or os.environ.get('LITELLM_API_KEY') or os.environ.get('OPENAI_API_KEY')
    request_kwargs = {'model': model, 'messages': messages, 'temperature': temperature, 'n': n, 'extra_body': {'enable_thinking': False}}
    if resolved_api_base:
        request_kwargs['api_base'] = resolved_api_base
    if resolved_api_key:
        request_kwargs['api_key'] = resolved_api_key
    response_cur = None
    for attempt in range(30):
        try:
            response_cur = completion(**request_kwargs)
            break
        except Exception as e:
            logging.info(f'Attempt {attempt + 1} failed with error: {e}')
            time.sleep(3)
    if response_cur is None:
        logging.info('Code terminated due to too many failed attempts!')
        exit()
    return response_cur.choices

def get_heuristic_name(module, possible_names: list[str]):
    for func_name in possible_names:
        if hasattr(module, func_name):
            if inspect.isfunction(getattr(module, func_name)):
                return func_name

def format_messages(cfg, pre_messages):
    messages = [{'role': 'system', 'content': pre_messages['system']}, {'role': 'user', 'content': pre_messages['user']}]
    return messages
