"""Launcher plans only; no model calls, real evaluations or generated outputs."""
import contextlib
import io
import unittest
from unittest.mock import patch

from experiments.cvrp_entropy_cost import run


class CVRPLauncherTests(unittest.TestCase):
    def test_six_jobs_group_order_and_output_isolation(self):
        jobs = run.build_jobs(['2d', '5d'], [1111, 2222, 3333], False, 'test_stamp')
        self.assertEqual([(g, s) for g, s, _, _ in jobs],
                         [('2d', 1111), ('2d', 2222), ('2d', 3333),
                          ('5d', 1111), ('5d', 2222), ('5d', 3333)])
        self.assertEqual(len({str(output) for _, _, output, _ in jobs}), 6)
        for group, seed, output, cmd in jobs:
            self.assertIn('runs_local_ollama', output.parts)
            self.assertIn(group, output.parts)
            self.assertIn(f'seed_{seed}', output.parts)
            self.assertIn(f'problem.behavior.feature_group={run.GROUPS[group]}', cmd)
            self.assertIn('model=qwen3:8b', cmd)
            self.assertIn('base_url=http://127.0.0.1:11434/v1', cmd)
            self.assertIn('api_key=ollama', cmd)
            self.assertIn('problem.behavior.matrices=5', cmd)
            self.assertIn('problem.behavior.probes_per_stage=4', cmd)
            self.assertIn('problem.behavior.probe_mode=random_stratified', cmd)
            self.assertIn('check=false', cmd)
            self.assertEqual(cmd[cmd.index('--config-path')+1], str(run.ROOT/'cfg'))

    def test_group_comparison_differs_only_in_projection_and_output(self):
        jobs = run.build_jobs(['2d', '5d'], [1111], False, 'same_stamp')
        shared = lambda cmd: [arg for arg in cmd if not arg.startswith(
            ('problem.behavior.feature_group=', 'hydra.run.dir='))]
        self.assertEqual(shared(jobs[0][3]), shared(jobs[1][3]))

    def test_check_jobs_have_separate_outputs(self):
        jobs = run.build_jobs(['2d'], [2222], True, 'test_stamp')
        _, _, output, cmd = jobs[0]
        self.assertIn('checks_local_ollama', output.parts)
        self.assertIn('check=true', cmd)

    def test_dry_run_never_starts_a_child(self):
        with patch('sys.argv', ['run.py', '--dry-run']), patch.object(run.subprocess, 'run') as child:
            with contextlib.redirect_stdout(io.StringIO()):
                run.main()
        child.assert_not_called()

    def test_explicit_group_seed_launch_and_duplicate_rejection(self):
        with patch('sys.argv', ['run.py', '--groups', '5d', '--seeds', '3333']), \
             patch.object(run.subprocess, 'run') as child:
            with contextlib.redirect_stdout(io.StringIO()):
                run.main()
        child.assert_called_once()
        self.assertIn('seed=3333', child.call_args.args[0])
        self.assertIn('problem.behavior.feature_group=entropy_time_preferences5', child.call_args.args[0])
        self.assertEqual(child.call_args.kwargs, dict(cwd=run.ROOT, check=True))
        with patch('sys.argv', ['run.py', '--seeds', '1111', '1111']), \
             contextlib.redirect_stderr(io.StringIO()), patch.object(run.subprocess, 'run') as child:
            with self.assertRaises(SystemExit):
                run.main()
        child.assert_not_called()


if __name__ == '__main__':
    unittest.main()
