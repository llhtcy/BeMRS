"""Transport/lifecycle tests, entirely mocked: no downloads, LLM or GPU work."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from deployment.vllm import runtime, transport
from experiments.op_entropy_cost import run as op


def response(content='OK', finish='stop'):
    handle = io.BytesIO(json.dumps({'choices': [{'message': {'content': content},
        'finish_reason': finish}], 'usage': {'prompt_tokens': 4, 'completion_tokens': 1}}).encode())
    return handle


class TransportTests(unittest.TestCase):
    def test_local_protocol_preserves_non_thinking_and_sampling(self):
        opener = MagicMock()
        opener.open.side_effect = lambda *a, **k: response()
        with patch.object(transport.urllib.request, 'build_opener', return_value=opener):
            result = transport.local_completion(2, [{'role': 'user', 'content': 'test'}], transport.MODEL, 1.0)
        self.assertEqual([item.message.content for item in result], ['OK', 'OK'])
        self.assertEqual(opener.open.call_count, 2)
        request = opener.open.call_args.args[0]
        self.assertEqual(request.full_url, 'http://127.0.0.1:8000/v1/chat/completions')
        data = json.loads(request.data)
        self.assertEqual(data['chat_template_kwargs'], {'enable_thinking': False})
        self.assertEqual((data['temperature'], data['top_p'], data['top_k']), (1.0, 0.95, 20))
        self.assertEqual(data['repetition_penalty'], 1.0)
        self.assertFalse(data['stream'])
        self.assertNotIn('think', data)
        self.assertNotIn('keep_alive', data)
        self.assertEqual(opener.open.call_args.kwargs['timeout'], 1800)

    def test_rejects_cloud_models_and_endpoints_without_request(self):
        with patch.object(transport.urllib.request, 'build_opener') as network:
            with self.assertRaises(ValueError):
                transport.local_completion(1, [], 'other-model', 1.0)
            with self.assertRaises(ValueError):
                transport.local_completion(1, [], transport.MODEL, 1.0, api_base='https://example.com/v1')
        network.assert_not_called()

    def test_truncated_and_empty_responses_fail_after_bounded_retries(self):
        for content, finish in [('partial', 'length'), ('', 'stop')]:
            opener = MagicMock()
            opener.open.side_effect = lambda *a, **k: response(content, finish)
            with patch.object(transport.urllib.request, 'build_opener', return_value=opener), \
                 patch.object(transport.time, 'sleep'):
                with self.assertRaises(RuntimeError):
                    transport.local_completion(1, [], transport.MODEL, 1.0)
            self.assertEqual(opener.open.call_count, 3)


class LifecycleTests(unittest.TestCase):
    def test_missing_compiler_fails_before_gpu_allocation(self):
        with patch.object(runtime, 'PYTHON', Path(__file__)), \
             patch.object(runtime.shutil, 'which', return_value=None), \
             patch.object(runtime.subprocess, 'check_output') as gpu:
            with self.assertRaisesRegex(RuntimeError, 'C compiler'):
                runtime.preflight()
        gpu.assert_not_called()

    def test_bf16_four_gpu_command_has_no_quantization(self):
        cmd = runtime.server_command()
        self.assertEqual(cmd[cmd.index('--dtype') + 1], 'bfloat16')
        self.assertEqual(cmd[cmd.index('--tensor-parallel-size') + 1], '4')
        self.assertEqual(cmd[cmd.index('--host') + 1], '127.0.0.1')
        self.assertNotIn('--quantization', cmd)
        self.assertEqual(runtime.GPU_IDS, ('0', '1', '2', '3'))

    def test_disabled_lifecycle_never_checks_or_loads_server(self):
        with patch.object(runtime, 'server_ready') as ready, patch.object(runtime, 'preflight') as preflight:
            with runtime.managed_server(enabled=False):
                pass
        ready.assert_not_called()
        preflight.assert_not_called()

    def test_external_server_is_not_stopped(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(runtime, 'ROOT', Path(folder)), \
             patch.object(runtime, 'server_ready', return_value=True), \
             patch.object(runtime.subprocess, 'Popen') as spawn, \
             patch.object(runtime, 'stop_owned_server') as stop, contextlib.redirect_stdout(io.StringIO()):
            with runtime.managed_server():
                pass
        spawn.assert_not_called()
        stop.assert_not_called()

    def test_owned_server_is_stopped_even_when_experiment_fails(self):
        process = MagicMock()
        with tempfile.TemporaryDirectory() as folder, patch.object(runtime, 'ROOT', Path(folder)), \
             patch.object(runtime, 'server_ready', side_effect=[False, True]), \
             patch.object(runtime, 'preflight'), \
             patch.object(runtime.subprocess, 'Popen', return_value=process) as spawn, \
             patch.object(runtime, 'stop_owned_server') as stop, contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, 'experiment failed'):
                with runtime.managed_server():
                    raise RuntimeError('experiment failed')
        stop.assert_called_once_with(process)
        self.assertTrue(spawn.call_args.kwargs['start_new_session'])
        self.assertEqual(spawn.call_args.kwargs['env']['CUDA_VISIBLE_DEVICES'], '0,1,2,3')
        self.assertEqual(spawn.call_args.kwargs['env']['HF_HUB_OFFLINE'], '1')
        self.assertEqual(spawn.call_args.kwargs['env']['VLLM_USE_FLASHINFER_SAMPLER'], '0')


class OPLauncherTests(unittest.TestCase):
    def test_six_jobs_and_old_results_remain_separate(self):
        jobs = op.build_jobs(['2d', '5d'], [1111, 2222, 3333], False, 'stamp', stratified=True)
        self.assertEqual(len(jobs), 6)
        for _, _, output, cmd in jobs:
            self.assertIn('runs_stratified_random_states_local_vllm', output.parts)
            self.assertIn('model=Qwen/Qwen3-8B', cmd)
            self.assertIn('--child-stratified', cmd)
        strip = lambda cmd: [value for value in cmd if not value.startswith(
            ('problem.behavior.feature_group=', 'hydra.run.dir='))]
        self.assertEqual(strip(jobs[0][3]), strip(jobs[3][3]))

    def test_dry_run_and_feature_checks_do_not_load_model(self):
        for arguments in (['--dry-run'], ['--check', '--groups', '5d', '--seeds', '1111']):
            with patch('sys.argv', ['run.py', *arguments]), \
                 patch.object(op.subprocess, 'run'), \
                 patch.object(op, 'managed_server', return_value=contextlib.nullcontext()) as manager, \
                 contextlib.redirect_stdout(io.StringIO()):
                op.main()
            manager.assert_called_once_with(enabled=False)


if __name__ == '__main__':
    unittest.main()
