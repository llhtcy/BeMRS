"""Run only entropy + extraction seconds + joint-input stability on OP.

All non-feature parameters come from the installed server configuration.
No monkey patches, no in-place configuration edits, no baseline reruns.
"""
import argparse
from datetime import datetime
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', nargs='+', type=int, default=[1111,2222,3333])
    parser.add_argument('--check', action='store_true', help='Extract seed features only; no LLM or real evaluations')
    parser.add_argument('--dry-run', action='store_true', help='Print commands only')
    args = parser.parse_args()
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    for seed in args.seeds:
        output = ROOT/'experiments'/'op_response3'/('checks' if args.check else 'runs')/f'seed_{seed}'/stamp
        overrides = ['problem=op_aco', f'seed={seed}', 'problem.behavior.feature_group=response3',
                     f'check={str(args.check).lower()}', f'hydra.run.dir={output}']
        cmd = [sys.executable, str(ROOT/'main.py'), *overrides]
        print(' '.join(cmd), flush=True)
        if not args.dry_run:
            subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
