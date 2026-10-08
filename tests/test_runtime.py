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
    def test_predictor_uses_one_configurable_threshold(self):
        from unittest.mock import Mock
        from bemrs.runtime_config import apply_runtime_config
        with initialize_config_dir(config_dir=str(ROOT/'cfg'), version_base=None):
            cfg = compose(config_name='config', overrides=['method.predictor.min_samples=7'])
        self.assertNotIn('cold_start', cfg.method)
        self.assertNotIn('behavior_explore', cfg.method)
        self.assertNotIn('start_successful_evals', cfg.method.predictor)
        with patch.dict(os.environ, {}, clear=True):
            applied = apply_runtime_config(cfg)
            self.assertEqual(applied['BEMRS_PREDICTOR_MIN_SAMPLES'], '7')
            self.assertFalse(any('COLD_START' in k or 'BEHAVIOR_EXPLORE' in k or
                                 'SURROGATE_START' in k for k in applied))
        engine = SearchEngine.__new__(SearchEngine)
        engine.surrogate_selection_enabled = True
        engine.behavior_filter_start_with_predictor = False
        engine.behavior_predictor = Mock(min_samples=7)
        engine.get_successful_real_eval_count = Mock(return_value=6)
        engine.behavior_predictor.get_sample_count.return_value = 7
        engine._train_predictor_if_needed = Mock(return_value=True)
        self.assertFalse(engine._ensure_surrogate())
        engine.get_successful_real_eval_count.return_value = 7
        engine.behavior_predictor.get_sample_count.return_value = 6
        self.assertFalse(engine._ensure_surrogate())
        engine._train_predictor_if_needed.assert_not_called()
        engine.behavior_predictor.get_sample_count.return_value = 7
        self.assertTrue(engine._ensure_surrogate())
        engine._train_predictor_if_needed.assert_called_once()
        engine.behavior_filter_start_with_predictor = True
        engine.behavior_predictor.surrogate_model = 'xgboost_direct'
        engine.behavior_predictor.is_ready.return_value = True
        engine._train_predictor_if_needed.reset_mock()
        engine.get_successful_real_eval_count.return_value = 6
        self.assertFalse(engine._behavior_filter_predictor_ready())
        engine.get_successful_real_eval_count.return_value = 7
        engine.behavior_predictor.get_sample_count.return_value = 6
        self.assertFalse(engine._behavior_filter_predictor_ready())
        engine._train_predictor_if_needed.assert_not_called()
        engine.behavior_predictor.get_sample_count.return_value = 7
        self.assertTrue(engine._behavior_filter_predictor_ready())


    def test_initialization_fixed_slots_and_successful_context(self):
        from unittest.mock import Mock
        engine = SearchEngine.__new__(SearchEngine)
        engine.is_real_eval_budget_exhausted = Mock(return_value=False)
        engine.get_remaining_generated_algorithm_slots = Mock(return_value=100)
        rows = [dict(code=c, algorithm=idea, objective=None) for c, idea in
                [('', 'empty'), ('bad', 'invalid'), ('ok', 'accepted'),
                 ('ok', 'duplicate'), ('other', 'same_score')]]
        engine.get_offspring = Mock(side_effect=[(None,r) for r in rows])
        def attach(candidates):
            off = candidates[0][1]
            off['behavior_feature_failed'] = off['code'] == 'bad'
        engine._attach_behavior_features = attach
        def evaluate(candidates, **kwargs):
            candidates[0][1]['objective'] = 1.0
            return candidates
        engine._evaluate_candidates = evaluate
        engine._finalize_lineage_candidates = Mock()
        engine._append_timing_record = Mock()
        result = engine._sequential_initialization(5)
        self.assertEqual([r['algorithm'] for r in result], ['accepted','same_score'])
        self.assertEqual(engine.get_offspring.call_count, 5)
        contexts = [call.kwargs['generation_context']['previous_ideas']
                    for call in engine.get_offspring.call_args_list]
        self.assertEqual(contexts, [[],[],[],['accepted'],['accepted']])

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
            return [SearchEngine._reciprocal_rank_parent_indices(distance,2,0,rng,prefer_near=near)[1] for _ in range(1000)]
        self.assertGreater(np.mean(sample(False)),np.mean(sample(True)))

    def test_full_search_mock_io(self):
        self._run_search_mock_io()

    def test_archive_fallback_without_regions(self):
        self._run_search_mock_io(regions=False, budget=32)

    def test_budget_exhausted_during_initialization(self):
        self._run_search_mock_io(budget=12)

    def _run_search_mock_io(self, regions=True, budget=65):
        from bemrs.core.llm import InterfaceAPI
        from bemrs.problem_adapter import Problem
        counter=[0]
        init_prompts=[]
        fit_events=[]
        original_update=SearchEngine._train_predictor_if_needed
        def update(engine,predictor,attr_name,label):
            previous=getattr(engine,attr_name+'_generation',None)
            ready=original_update(engine,predictor,attr_name,label)
            attempt=getattr(engine,attr_name+'_generation',None)
            if attempt != previous:
                fit_events.append((engine.current_generation,predictor.get_sample_count()))
            return ready
        def responses(self,prompts,max_workers=None):
            result=[]
            for prompt in prompts:
                if 'initialization candidate No.' in prompt:
                    init_prompts.append(prompt)
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
                    cfg=compose(config_name='config',overrides=[f'max_fe={budget}',f'method.region.enabled={str(regions).lower()}','method.visualization.enabled=false','method.generation.evaluation_ratio=0.1','method.region.archive_target_distinct=24'])
                self.assertNotIn('pop_size', cfg)
                workspace=isolated_workspace(Path(tmp))
                with patch('bemrs.core.engine.build_behavior_embedder',return_value=Encoder()),patch.object(InterfaceAPI,'get_responses',responses),patch.object(Problem,'batch_evaluate',evaluate),patch.object(SearchEngine,'_train_predictor_if_needed',update):
                    code,path=BeMRS(cfg,workspace).evolve()
                self.assertTrue(code);self.assertTrue(Path(path).exists())
                import json
                rows=[json.loads(l) for l in Path('algorithm_lineage.jsonl').read_text().splitlines()]
                successful=[r for r in rows if r['evaluation']['success'] and not r['evaluation'].get('shared')]
                self.assertEqual(len(successful),budget)
                self.assertEqual(len(fit_events),len({g for g,_ in fit_events}))
                if budget > 50:
                    self.assertTrue(fit_events)
                    self.assertTrue(all(samples >= 50 for _,samples in fit_events))
                self.assertFalse(any(r['operator']=='be' for r in rows))
                for row in rows:
                    if row['operator'] in ('bx', 'br'):
                        self.assertEqual(len(row['parents']), 2)
                self.assertEqual(sum(r['generation']==0 for r in successful),min(24,budget))
                self.assertEqual(len(init_prompts),min(24,budget)-1)
                snapshots = list(Path('.').glob('population_generation_*.json'))
                self.assertTrue(snapshots)
                for snapshot in snapshots:
                    archive = json.loads(snapshot.read_text())
                    self.assertLessEqual(len(archive),24)
                    self.assertEqual(len(archive),len({r['objective'] for r in archive}))
                    self.assertEqual(archive,sorted(archive,key=lambda r:r['objective']))
                best=json.loads(Path(path).read_text())
                self.assertEqual(best['objective'], min(r['child']['objective'] for r in successful))
                if not regions:
                    formal=[r for r in rows if r['generation']>0]
                    self.assertTrue(formal)
                    self.assertEqual({r['pipeline']['generation_parent_scope'] for r in formal},{'advantage_archive'})
                    self.assertEqual(sum(r['generation']==1 and r['operator']=='bx' for r in formal),24)
                    return
                if budget <= 24:
                    self.assertFalse(any(r['generation']>0 for r in rows))
                    return
                last_context=init_prompts[-1].split('Most recent successfully evaluated algorithm ideas')[1]
                self.assertIn('Unique synthetic mechanism 22\n',last_context)
                self.assertIn('Unique synthetic mechanism 13\n',last_context)
                self.assertNotIn('Unique synthetic mechanism 12\n',last_context)
                self.assertTrue(any(r['pipeline'].get('behavior_novelty_slot') for r in rows))
                self.assertTrue(any(r['pipeline'].get('selection_method') == 'advantage_region_xgboost_direct' for r in rows))
                from collections import defaultdict
                import math
                mixed = defaultdict(list)
                for row in rows:
                    if row['pipeline'].get('generation_kind'):
                        mixed[row['generation']].append(row)
                self.assertTrue(mixed)
                remaining = budget - 24  # One archive-sized initialization, no BE.
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
