"""CVRP5D, stratified random probes and the existing local Ollama transport."""
import argparse
from datetime import datetime
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', nargs='+', type=int, default=[1111, 2222, 3333])
    parser.add_argument('--check', action='store_true', help='Fixed seed extraction only; no LLM/real evaluator')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    for seed in args.seeds:
        mode = 'checks_local_ollama' if args.check else 'runs_local_ollama'
        output = ROOT/'experiments/cvrp_entropy_cost'/mode/'5d'/f'seed_{seed}'/stamp
        cmd = [sys.executable, str(Path(__file__).resolve()), '--child',
               '--config-path', str(ROOT/'cfg'), 'problem=cvrp_aco', f'seed={seed}',
               'model=qwen3:8b', 'base_url=http://127.0.0.1:11434/v1', 'api_key=ollama',
               'problem.behavior.feature_group=entropy_time_preferences5',
               'problem.behavior.probe_mode=random_stratified',
               f'check={str(args.check).lower()}', f'hydra.run.dir={output}']
        print(' '.join(cmd), flush=True)
        if not args.dry_run:
            subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--child':
        sys.path.insert(0, str(ROOT))
        # Same loopback-only, think=False, keep_alive=5m adapter as OP.
        from experiments.op_entropy_cost.run import install_local_adapter
        install_local_adapter()
        sys.argv = [str(ROOT/'main.py'), *sys.argv[2:]]
        from main import main as run_experiment
        run_experiment()
    else:
        main()
