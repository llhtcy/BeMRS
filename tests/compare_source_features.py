"""Compare real probe descriptors with the archived server implementation."""
import json, os, subprocess, sys
from pathlib import Path
import numpy as np
from hydra import initialize_config_dir,compose
from bemrs import BeMRS
from bemrs.core.engine import SearchEngine

root=Path(__file__).resolve().parents[1]
source=Path(sys.argv[1]).resolve()
for task in ('op_aco','mkp_aco','cvrp_aco'):
    with initialize_config_dir(version_base=None,config_dir=str(root/'cfg')):
        cfg=compose(config_name='config',overrides=[f'problem={task}'])
    runner=BeMRS(cfg,root)
    engine=SearchEngine(cfg,runner.problem)
    code='import numpy as np\n'+runner.problem.prompts.seed_func
    actual=engine.embedder.encode_single(code)
    env=os.environ.copy()
    for key,val in list(env.items()):
        if key.startswith('BEMRS_'):env['EOH_'+key[6:]]=val
    env['PYTHONPATH']=str(source)
    snippet='''import importlib,json,sys
from pathlib import Path
m=importlib.import_module('behavior_guided_simplified_stage8_behavior_filter.'+sys.argv[1]+'_behavior')
cls=getattr(m,{'op_aco':'OPACOBehaviorEmbedder','mkp_aco':'MKPACOBehaviorEmbedder','cvrp_aco':'CVRPACOBehaviorEmbedder'}[sys.argv[1]])
enc=cls(Path(sys.argv[2])/sys.argv[3]);code=sys.stdin.read()
indices=[i for i,n in enumerate(m.TREND_FEATURE_NAMES) if n.endswith('_mean')]
print('RESULT='+json.dumps(enc.encode_single(code)[indices].tolist()))'''
    r=subprocess.run([sys.executable,'-c',snippet,task,str(source),cfg.problem.behavior.dataset],input=code,text=True,capture_output=True,env=env,cwd=source,check=True)
    expected=json.loads(next(l[7:] for l in r.stdout.splitlines() if l.startswith('RESULT=')))
    np.testing.assert_array_equal(actual,np.asarray(expected,dtype=np.float32))
    print('EXACT MATCH',task,len(actual),flush=True)
