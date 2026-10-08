"""CVRP2D then CVRP5D, sharing stratified probes and official BF16 vLLM."""
import argparse
from datetime import datetime
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
GROUPS = {'2d': 'entropy_time2', '5d': 'entropy_time_preferences5'}
sys.path.insert(0, str(ROOT))
from deployment.vllm.runtime import managed_server
from deployment.vllm.transport import API_KEY, BASE_URL, MODEL, install_local_adapter


def build_jobs(groups, seeds, check, stamp):
    """Plan independent runs; only feature projection differs between groups."""
    jobs = []
    mode = 'checks_local_vllm' if check else 'runs_local_vllm'
    for group in groups:
        for seed in seeds:
            output = ROOT/'experiments/cvrp_entropy_cost'/mode/group/f'seed_{seed}'/stamp
            cmd = [sys.executable, str(Path(__file__).resolve()), '--child',
                   '--config-path', str(ROOT/'cfg'), 'problem=cvrp_aco', f'seed={seed}',
                   f'model={MODEL}', f'base_url={BASE_URL}', f'api_key={API_KEY}',
                   f'problem.behavior.feature_group={GROUPS[group]}',
                   'problem.behavior.probe_mode=random_stratified',
                   'problem.behavior.matrices=5', 'problem.behavior.probes_per_stage=4',
                   f'check={str(check).lower()}', f'hydra.run.dir={output}']
            jobs.append((group, seed, output, cmd))
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--groups', nargs='+', choices=GROUPS, default=['2d', '5d'])
    parser.add_argument('--seeds', nargs='+', type=int, default=[1111, 2222, 3333])
    parser.add_argument('--check', action='store_true', help='Fixed seed extraction only; no LLM/real evaluator')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if len(set(args.groups)) != len(args.groups) or len(set(args.seeds)) != len(args.seeds):
        parser.error('Groups and seeds must not contain duplicates')
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    jobs = build_jobs(args.groups, args.seeds, args.check, stamp)
    print('CVRP official Qwen3-8B BF16 vLLM jobs (5 instances x 12 probes):', flush=True)
    for group, seed, output, _ in jobs:
        print(f'  {group} seed={seed} -> {output}', flush=True)
    with managed_server(enabled=not args.check and not args.dry_run):
        for group, seed, output, cmd in jobs:
            print(' '.join(cmd), flush=True)
            if not args.dry_run:
                subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--child':
        sys.path.insert(0, str(ROOT))
        # Same loopback-only official BF16 vLLM adapter as OP.
        install_local_adapter()
        sys.argv = [str(ROOT/'main.py'), *sys.argv[2:]]
        from main import main as run_experiment
        run_experiment()
    else:
        main()
