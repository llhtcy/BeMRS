"""Sequential OP feature comparison: all 2D seeds, then all 5D seeds."""
import argparse
from datetime import datetime
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
GROUPS = {'2d':'entropy_time2','5d':'entropy_time_preferences5'}
OLLAMA_URL = 'http://127.0.0.1:11434'
MODEL = 'qwen3:8b'


def install_local_adapter():
    """Experiment-only transport, installed before InterfaceAPI is imported."""
    sys.path.insert(0,str(ROOT))
    # Never inherit cloud credentials or route local requests through a proxy.
    for name in ('OPENAI_API_KEY','LITELLM_API_KEY'):
        os.environ[name]='ollama'
    for name in ('OPENAI_BASE_URL','OPENAI_API_BASE','LITELLM_API_BASE'):
        os.environ[name]=OLLAMA_URL+'/v1'
    for name in ('NO_PROXY','no_proxy'):
        os.environ[name]=','.join(filter(None,[os.environ.get(name,''),'127.0.0.1','localhost']))
    import utils.utils as shared

    def local_completion(n, messages, model, temperature, api_base=None, api_key=None):
        if model not in (MODEL,'openai/'+MODEL):
            raise ValueError(f'Local experiment refuses unexpected model {model}')
        choices=[]
        for _ in range(n):
            payload=dict(model=MODEL,messages=messages,stream=False,think=False,
                         options=dict(temperature=temperature),keep_alive='5m')
            for attempt in range(3):
                try:
                    request=urllib.request.Request(OLLAMA_URL+'/api/chat',
                        data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
                    # Explicit proxy bypass for the loopback-only transport.
                    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
                    with opener.open(request,timeout=1800) as response:
                        backend=response.headers.get('X-Ollama-Backend','unknown')
                        result=json.load(response)
                    content=result.get('message',{}).get('content','')
                    if not result.get('done') or not content.strip():
                        raise RuntimeError('Ollama returned incomplete or empty content')
                    if result.get('done_reason')=='length':
                        raise RuntimeError('Ollama truncated the generated response')
                    logging.info('[LocalOllama] model=%s backend=%s think=False prompt_tokens=%s output_tokens=%s',
                                 MODEL,backend,result.get('prompt_eval_count'),result.get('eval_count'))
                    choices.append(SimpleNamespace(message=SimpleNamespace(content=content)))
                    break
                except Exception:
                    if attempt==2:
                        raise
                    time.sleep(2)
        return choices

    shared.chat_completion=local_completion


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--groups',nargs='+',choices=GROUPS,default=['2d','5d'])
    parser.add_argument('--seeds',nargs='+',type=int,default=[1111,2222,3333])
    parser.add_argument('--check',action='store_true',help='Seed extraction only; no LLM or objective evaluations')
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--random-states',action='store_true',help='Uniform feasible random paths and random state sampling; separate results')
    parser.add_argument('--stratified-random-states',action='store_true',help='Random paths, equal early/middle/late state counts')
    parser.add_argument('--llm-check',action='store_true',help='Six short concurrent local requests via the real InterfaceAPI; no search')
    args=parser.parse_args()
    if args.llm_check:
        install_local_adapter()
        logging.basicConfig(level=logging.INFO)
        from bemrs.core.llm import InterfaceAPI
        replies=InterfaceAPI(OLLAMA_URL+'/v1','ollama',MODEL,False).get_responses(
            ['Reply with exactly OK.']*6,max_workers=6)
        assert all(reply.strip() for reply in replies)
        print('LOCAL LLM CHECK OK:',replies)
        return
    stamp=datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    for group in args.groups:
        for seed in args.seeds:
            suffix = '_stratified_random_states_local_ollama' if args.stratified_random_states else ('_random_states_local_ollama' if args.random_states else '_local_ollama')
            mode=('checks' if args.check else 'runs')+suffix
            output=ROOT/'experiments'/'op_entropy_cost'/mode/group/f'seed_{seed}'/stamp
            child='--child-stratified' if args.stratified_random_states else ('--child-random' if args.random_states else '--child')
            cmd=[sys.executable,str(Path(__file__).resolve()),child,'--config-path',str(ROOT/'cfg'),
                 'problem=op_aco',f'seed={seed}',
                 f'model={MODEL}',f'base_url={OLLAMA_URL}/v1','api_key=ollama',
                 f'problem.behavior.feature_group={GROUPS[group]}',f'check={str(args.check).lower()}',
                 f'hydra.run.dir={output}']
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
