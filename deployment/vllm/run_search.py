"""Run the formal BeMRS entry point with on-demand official BF16 vLLM."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from deployment.vllm.runtime import managed_server
from deployment.vllm.transport import API_KEY, BASE_URL, MODEL, install_local_adapter


if __name__ == '__main__':
    arguments = sys.argv[1:]
    checking = any(arg.lower() == 'check=true' for arg in arguments)
    inspecting = any(arg in ('--help', '-h', '--cfg', '--info') for arg in arguments)
    sys.argv = [str(ROOT/'main.py'), *arguments,
                f'model={MODEL}', f'base_url={BASE_URL}', f'api_key={API_KEY}']
    install_local_adapter()
    from main import main
    with managed_server(enabled=not checking and not inspecting):
        main()
