"""Evaluate one supplied algorithm using an isolated copy of the problem runner."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
from main import ROOT, isolated_workspace

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--problem',required=True,choices=sorted(p.name for p in (ROOT/'problems').iterdir() if p.is_dir() and not p.name.startswith('.')))
    parser.add_argument('--code',required=True,type=Path)
    parser.add_argument('--size',required=True,type=int)
    parser.add_argument('--split',choices=('train','val'),default='train')
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--timeout',type=int,default=300)
    args=parser.parse_args()
    code=args.code.resolve().read_text()
    # Seed templates share the same NumPy scaffolding as search initialization.
    if args.code.name == 'seed_func.txt':
        code='import numpy as np\n\n'+code
    out=args.output.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Refusing to overwrite {out}')
    out.mkdir(parents=True,exist_ok=True)
    workspace=isolated_workspace(out)
    problem=workspace/'problems'/args.problem
    (problem/'gpt.py').write_text(code+'\n')
    env=os.environ.copy();env['PYTHONPATH']=str(ROOT)+os.pathsep+env.get('PYTHONPATH','')
    with (out/'evaluation.log').open('w') as log:
        subprocess.run([sys.executable,str(problem/'eval.py'),str(args.size),str(workspace),args.split],cwd=workspace,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=args.timeout,check=True)
    print((out/'evaluation.log').read_text())

if __name__=='__main__':main()
