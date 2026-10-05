"""Lightweight behavioral tests. No LLM calls and no expensive objective runs."""
import hashlib
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from hydra import initialize_config_dir, compose
from bemrs.core.engine import SearchEngine
from bemrs.regional_allocation import annealed_winner_take_most_quotas
from bemrs import BeMRS
from main import ROOT, isolated_workspace

class RuntimeTests(unittest.TestCase):
    def test_allocation(self):
        for k in (1,2,3,5):
            q,_,_=annealed_winner_take_most_quotas(list(range(k)),{r:8 for r in range(k)},4,210,210,{},None)
            self.assertEqual(sum(q.values()),4)
            self.assertEqual(q[0],4)
        counts={};cursor=None
        for _ in range(5):
            q,d,cursor=annealed_winner_take_most_quotas(list(range(5)),{r:8 for r in range(5)},4,0,210,counts,cursor)
            counts=d['regional_allocated_count']
        self.assertEqual(set(counts.values()),{4})
        q,_,_=annealed_winner_take_most_quotas([0,1,2],{0:1,1:5,2:4},4,210,210,{},None)
        self.assertEqual(sum(q.values()),4);self.assertEqual(q[0],1)

    def test_parent_rank(self):
        distance=np.array([[0,1,2,3],[1,0,1,2],[2,1,0,1],[3,2,1,0]],float)
        def sample(near):
            rng=np.random.default_rng(1111)
            return [SearchEngine._rank_softmax_maxmin_parent_indices(distance,2,0,1.5,rng,prefer_near=near)[1] for _ in range(1000)]
        self.assertGreater(np.mean(sample(False)),np.mean(sample(True)))

    def test_full_search_mock_io(self):
        from bemrs.core.llm import InterfaceAPI
        from bemrs.problem_adapter import Problem
        counter=[0]
        def responses(self,prompts,max_workers=None):
            result=[]
            for _ in prompts:
                counter[0]+=1
                result.append('{Unique synthetic mechanism '+str(counter[0])+'}\n```python\nimport numpy as np\ndef heuristics_v2(prize, distance, maxlen):\n    return np.ones_like(distance) * '+str(counter[0])+'\n```')
            return result
        class Encoder:
            output_dim=6;extractor_version='test-only';extractor_description='test-only'
            def encode_single(self,code):
                seed=int(hashlib.sha256(code.encode()).hexdigest()[:8],16)
                return np.random.default_rng(seed).random(6).astype(np.float32)
            def encode(self,codes,**kwargs):return np.array([self.encode_single(c) for c in codes])
        def evaluate(self,codes,iteration):
            return [-10-float(int(hashlib.sha256(c.encode()).hexdigest()[:8],16))/2**32 for c in codes]
        old=Path.cwd()
        with tempfile.TemporaryDirectory() as tmp:
            os.chdir(tmp)
            try:
                with initialize_config_dir(version_base=None,config_dir=str(ROOT/'cfg')):
                    cfg=compose(config_name='config',overrides=['max_fe=65','method.visualization.enabled=false','method.cold_start.visualize=false','method.generation.evaluation_ratio=0.1','method.region.archive_target_distinct=24'])
                workspace=isolated_workspace(Path(tmp))
                with patch('bemrs.core.engine.build_behavior_embedder',return_value=Encoder()),patch.object(InterfaceAPI,'get_responses',responses),patch.object(Problem,'batch_evaluate',evaluate):
                    code,path=BeMRS(cfg,workspace).evolve()
                self.assertTrue(code);self.assertTrue(Path(path).exists())
                import json
                rows=[json.loads(l) for l in Path('algorithm_lineage.jsonl').read_text().splitlines()]
                successful=[r for r in rows if r['evaluation']['success'] and not r['evaluation'].get('shared')]
                self.assertEqual(len(successful),65)
                self.assertTrue(any(r['pipeline'].get('behavior_novelty_slot') for r in rows))
                self.assertTrue(any(r['pipeline'].get('selection_method') == 'advantage_region_xgboost_direct' for r in rows))
                from collections import defaultdict
                import math
                mixed = defaultdict(list)
                for row in rows:
                    if row['pipeline'].get('generation_kind'):
                        mixed[row['generation']].append(row)
                self.assertTrue(mixed)
                remaining = 65 - 30  # initialization and two BE rounds unchanged
                selected_counts = []
                for generation in sorted(mixed):
                    batch = mixed[generation]
                    self.assertEqual({r['operator'] for r in batch}, {'bx','br'})
                    self.assertTrue({r['pipeline']['generation_kind'] for r in batch}
                                    <= {'intra_bx','intra_br','inter_bx'})
                    selected = [r for r in batch if r['pipeline']['selected_for_evaluation']]
                    self.assertEqual(len(selected), min(math.ceil(len(batch)*.1), remaining))
                    self.assertEqual(len({r['child']['code'] for r in selected}),len(selected))
                    novelty = [r for r in selected if r['pipeline'].get('behavior_novelty_slot')]
                    self.assertEqual(len(novelty), int(len(selected)>=2 and len(batch)>len(selected)))
                    remaining -= len(selected)
                    selected_counts.append(len(selected))
                self.assertEqual(remaining,0)
                self.assertGreater(len(selected_counts),1)
            finally:os.chdir(old)

if __name__=='__main__':unittest.main()
