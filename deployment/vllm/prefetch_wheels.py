"""Parallel-download a pip resolution report, checking PyPI archive SHA256."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
from urllib.parse import unquote, urlsplit
import urllib.request

from download_model import digest, fetch_ranges


def fetch(item, root):
    download = item['download_info']
    url = download['url']
    parsed = urlsplit(url)
    if parsed.scheme == 'file':
        return
    if parsed.scheme != 'https' or parsed.hostname != 'files.pythonhosted.org':
        raise RuntimeError('Only public PyPI archive downloads are allowed')
    filename = Path(unquote(parsed.path)).name
    if not filename.endswith('.whl'):
        raise RuntimeError('Expected a prebuilt wheel, not source builds')
    expected = download['archive_info']['hashes']['sha256']
    target = root / filename
    if target.exists():
        if digest(target) != expected:
            raise RuntimeError(f'Existing wheel checksum mismatch: {filename}')
        print(f'WHEEL verified cache {filename}', flush=True)
        return
    with urllib.request.urlopen(urllib.request.Request(url, method='HEAD'), timeout=30) as response:
        size = int(response.headers['Content-Length'])
    partial = target.with_name(target.name + '.part')
    if not partial.exists():
        candidates = [folder / filename for folder in Path('/tmp').glob('pip-unpack-*')
                      if (folder / filename).is_file() and (folder / filename).stat().st_size <= size]
        if candidates:
            shutil.copyfile(max(candidates, key=lambda path: path.stat().st_size), partial)
    fetch_ranges(partial, url, size)
    if digest(partial) != expected:
        raise RuntimeError(f'PyPI SHA256 mismatch: {filename}')
    partial.replace(target)
    print(f'WHEEL verified download {filename} bytes={size}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', type=Path)
    parser.add_argument('--directory', type=Path, default=Path('/root/lhc/bemrs-vllm-wheelhouse'))
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    plan = json.loads(args.report.read_text())
    with ThreadPoolExecutor(max_workers=6) as executor:
        list(executor.map(lambda item: fetch(item, args.directory), plan['install']))
    print('ALL RESOLVED WHEELS READY', flush=True)
