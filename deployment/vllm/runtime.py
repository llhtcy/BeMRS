"""Four-GPU vLLM lifecycle: load for a command, release on exit, no daemon."""
from contextlib import contextmanager
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import time
import urllib.request

from deployment.vllm.transport import BASE_URL, MODEL

ROOT = Path(__file__).resolve().parents[2]
PYTHON = Path('/root/lhc/bemrs-vllm-venv/bin/python')
CHECKPOINT = Path('/root/lhc/models/Qwen3-8B')
GPU_IDS = ('0', '1', '2', '3')
GPU_MEMORY_UTILIZATION = 0.35


def server_command():
    return [str(PYTHON), '-m', 'vllm.entrypoints.openai.api_server',
            '--model', str(CHECKPOINT), '--served-model-name', MODEL,
            '--dtype', 'bfloat16', '--tensor-parallel-size', '4',
            '--distributed-executor-backend', 'mp',
            '--host', '127.0.0.1', '--port', '8000',
            '--gpu-memory-utilization', str(GPU_MEMORY_UTILIZATION),
            '--max-model-len', '40960', '--max-num-seqs', '6',
            '--max-num-batched-tokens', '4096',
            '--generation-config', 'vllm', '--disable-custom-all-reduce', '--enforce-eager']


def server_ready():
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(BASE_URL + '/models', timeout=2) as response:
            listing = json.load(response)
    except OSError:
        return False
    if MODEL not in {item.get('id') for item in listing.get('data', [])}:
        raise RuntimeError('Port 8000 already serves an unexpected model; refusing to replace it')
    return True


def preflight():
    if not PYTHON.is_file():
        raise RuntimeError(f'vLLM environment missing: {PYTHON}')
    if not shutil.which(os.environ.get('CC', 'cc')):
        raise RuntimeError('Triton requires a C compiler; install gcc, g++ and libc6-dev before loading GPUs')
    provenance = CHECKPOINT / 'bemrs_checkpoint_provenance.json'
    if not provenance.is_file():
        raise RuntimeError('Verified official checkpoint missing; run deployment/vllm/download_model.py first')
    metadata = json.loads(provenance.read_text())
    if metadata.get('model') != MODEL or set(metadata.get('weight_dtypes', {})) != {'BF16'} or metadata.get('quantization') is not None:
        raise RuntimeError('Expected the verified official, unquantized BF16 checkpoint')
    rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.total,memory.free',
                                     '--format=csv,noheader,nounits'], text=True)
    memory = {fields[0].strip(): tuple(float(value) for value in fields[1:])
              for line in rows.splitlines() if (fields := line.split(','))}
    for gpu in GPU_IDS:
        if gpu not in memory:
            raise RuntimeError(f'Required GPU {gpu} unavailable')
        total, free = memory[gpu]
        if free < total * GPU_MEMORY_UTILIZATION + 1024:
            raise RuntimeError(f'GPU {gpu} has only {free:.0f} MiB free; wait for other jobs, do not stop them')


def stop_owned_server(process):
    """Signal only the process group created by this context, never other jobs."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


@contextmanager
def managed_server(enabled=True, external_service=False):
    if not enabled:
        yield
        return
    logs = ROOT / 'deployment/vllm/logs'
    logs.mkdir(parents=True, exist_ok=True)
    with (logs / 'managed_server.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('Another managed vLLM experiment is running; run commands sequentially') from exc
        if server_ready():
            print(f'[LocalVLLM] Reusing external server {BASE_URL}; it will not be stopped by this command', flush=True)
            yield
            return
        preflight()
        log = logs / ('server_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '.log')
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=','.join(GPU_IDS), NCCL_P2P_DISABLE='1', NCCL_IB_DISABLE='1',
                   HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', VLLM_NO_USAGE_STATS='1',
                   VLLM_USE_FLASHINFER_SAMPLER='0',
                   OMP_NUM_THREADS='4', VLLM_WORKER_MULTIPROC_METHOD='spawn')
        with log.open('a') as output:
            process = subprocess.Popen(server_command(), env=env, stdout=output, stderr=subprocess.STDOUT,
                                       cwd=ROOT, start_new_session=True)
            previous_handler = signal.getsignal(signal.SIGTERM)
            def terminate(signum, frame):
                raise KeyboardInterrupt('Experiment terminated; releasing owned vLLM server')
            signal.signal(signal.SIGTERM, terminate)
            try:
                print(f'[LocalVLLM] Starting official BF16 checkpoint on GPUs {",".join(GPU_IDS)}; log={log}', flush=True)
                deadline = time.monotonic() + 900
                last_report = time.monotonic()
                while not server_ready():
                    if process.poll() is not None:
                        raise RuntimeError(f'vLLM exited with {process.returncode}; inspect {log}')
                    if time.monotonic() > deadline:
                        raise RuntimeError(f'vLLM startup timed out; inspect {log}')
                    if time.monotonic() - last_report >= 30:
                        print(f'[LocalVLLM] Still loading; inspect {log}', flush=True)
                        last_report = time.monotonic()
                    time.sleep(1)
                print(f'[LocalVLLM] Ready: {BASE_URL} model={MODEL} dtype=BF16 TP=4', flush=True)
                if external_service:
                    # A deliberately foreground service can be borrowed by an
                    # experiment; its lifetime stays under this terminal's control.
                    fcntl.flock(lock, fcntl.LOCK_UN)
                yield
            finally:
                signal.signal(signal.SIGTERM, previous_handler)
                stop_owned_server(process)
                print('[LocalVLLM] Owned server stopped; GPU memory released', flush=True)


if __name__ == '__main__':
    with managed_server(external_service=True):
        print('Foreground service ready; Ctrl+C stops only this vLLM service.', flush=True)
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
