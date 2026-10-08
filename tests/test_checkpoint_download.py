"""Small in-memory tests for resuming checkpoint chunks; no network or model."""
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from deployment.vllm.download_model import fetch_ranges, resumable_ranges


class CheckpointResumeTests(unittest.TestCase):
    def test_reuses_old_chunks_and_downloads_only_missing_gap(self):
        data = b'abcdefghijkl'
        with tempfile.TemporaryDirectory() as folder:
            partial = Path(folder) / 'model.part'
            partial.write_bytes(data[:3])
            partial.with_name('model.part.range_3_8').write_bytes(data[3:8])
            partial.with_name('model.part.range_10_12').write_bytes(data[10:])
            self.assertEqual(resumable_ranges(partial, len(data)), [(3, 8), (8, 10), (10, 12)])
            def open_range(request, **kwargs):
                self.assertEqual(request.get_header('Range'), 'bytes=8-9')
                response = io.BytesIO(data[8:10])
                response.status = 206
                response.headers = {'Content-Range': 'bytes 8-9/12'}
                return response
            with patch('deployment.vllm.download_model.urllib.request.urlopen', side_effect=open_range) as network:
                fetch_ranges(partial, 'https://example.com/model', len(data))
            network.assert_called_once()
            self.assertEqual(partial.read_bytes(), data)
            self.assertEqual(list(Path(folder).glob('*.range_*')), [])

    def test_incomplete_cached_chunk_resumes_at_its_actual_length(self):
        with tempfile.TemporaryDirectory() as folder:
            partial = Path(folder) / 'model.part'
            partial.write_bytes(b'abc')
            partial.with_name('model.part.range_3_6').write_bytes(b'd')
            response = io.BytesIO(b'ef')
            response.status = 206
            response.headers = {'Content-Range': 'bytes 4-5/6'}
            with patch('deployment.vllm.download_model.urllib.request.urlopen', return_value=response) as network:
                fetch_ranges(partial, 'https://example.com/model', 6)
            self.assertEqual(network.call_args.args[0].get_header('Range'), 'bytes=4-5')
            self.assertEqual(partial.read_bytes(), b'abcdef')

    def test_oversized_prefix_is_not_modified(self):
        with tempfile.TemporaryDirectory() as folder:
            partial = Path(folder) / 'model.part'
            partial.write_bytes(b'oversized')
            with self.assertRaisesRegex(RuntimeError, 'Oversized partial'):
                resumable_ranges(partial, 3)
            self.assertEqual(partial.read_bytes(), b'oversized')


if __name__ == '__main__':
    unittest.main()
