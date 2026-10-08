"""Sequential OP feature comparison: all 2D seeds, then all 5D seeds."""
import argparse
from datetime import datetime
import logging
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
GROUPS = {'2d':'entropy_time2','5d':'entropy_time_preferences5'}
sys.path.insert(0, str(ROOT))
from deployment.vllm.runtime import managed_server
from deployment.vllm.transport import API_KEY, BASE_URL, MODEL, install_local_adapter


def build_jobs(groups, seeds, check, stamp, random_states=False, stratified=False):
    jobs = []
    suffix = '_stratified_random_states_local_vllm' if stratified else ('_random_states_local_vllm' if random_states else '_local_vllm')
    mode = ('checks' if check else 'runs') + suffix
    child = '--child-stratified' if stratified else ('--child-random' if random_states else '--child')
    for group in groups:
        for seed in seeds:
            output = ROOT/'experiments/op_entropy_cost'/mode/group/f'seed_{seed}'/stamp
            cmd = [sys.executable, str(Path(__file__).resolve()), child,
                   '--config-path', str(ROOT/'cfg'), 'problem=op_aco', f'seed={seed}',
                   f'model={MODEL}', f'base_url={BASE_URL}', f'api_key={API_KEY}',
                   f'problem.behavior.feature_group={GROUPS[group]}',
                   f'check={str(check).lower()}', f'hydra.run.dir={output}']
            jobs.append((group, seed, output, cmd))
    return jobs


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--groups',nargs='+',choices=GROUPS,default=['2d','5d'])
    parser.add_argument('--seeds',nargs='+',type=int,default=[1111,2222,3333])
    parser.add_argument('--check',action='store_true',help='Seed extraction only; no LLM or objective evaluations')
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--random-states',action='store_true',help='Uniform feasible random paths and random state sampling; separate results')
    parser.add_argument('--stratified-random-states',action='store_true',help='Random paths, equal early/middle/late state counts')
    parser.add_argument('--llm-check',action='store_true',help='Six short concurrent vLLM requests via the real InterfaceAPI; no search')
    args=parser.parse_args()
    if len(set(args.groups)) != len(args.groups) or len(set(args.seeds)) != len(args.seeds):
        parser.error('Groups and seeds must not contain duplicates')
    if args.random_states and args.stratified_random_states:
        parser.error('Choose either random or stratified-random states')
    if args.llm_check:
        with managed_server():
            install_local_adapter()
            logging.basicConfig(level=logging.INFO)
            from bemrs.core.llm import InterfaceAPI
            replies=InterfaceAPI(BASE_URL,API_KEY,MODEL,False).get_responses(
                ['Reply with exactly OK.']*6,max_workers=6)
            assert all(reply.strip() for reply in replies)
            print('LOCAL VLLM CHECK OK:',replies)
        return
    stamp=datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    jobs = build_jobs(args.groups, args.seeds, args.check, stamp,
                      random_states=args.random_states, stratified=args.stratified_random_states)
    with managed_server(enabled=not args.check and not args.dry_run):
        for group, seed, output, cmd in jobs:
            print(' '.join(cmd),flush=True)
            if not args.dry_run:
                subprocess.run(cmd,cwd=ROOT,check=True)


if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1] in ('--child','--child-random','--child-stratified'):
        random_states=sys.argv[1] in ('--child-random','--child-stratified')
        stratified=sys.argv[1]=='--child-stratified'
        sys.argv=[str(ROOT/'main.py'),*sys.argv[2:]]
        install_local_adapter()
        if random_states:
            from random_states import install
            install(stratified=stratified)
        from main import main as run_experiment
        run_experiment()
    else:
        main()
