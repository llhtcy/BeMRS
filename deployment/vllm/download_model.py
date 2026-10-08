"""Download Qwen's official BF16 checkpoint from its ModelScope repository.

No code conversion or quantization. Every file is checked against the official
repository's SHA256; interrupted downloads resume when the server supports it.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import struct
import time
from urllib.parse import urlencode
import urllib.request

MODEL = 'Qwen/Qwen3-8B'
BASE = 'https://modelscope.cn/api/v1/models/' + MODEL


def digest(path):
    hasher = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def resumable_ranges(partial, size, block_size=16 * 1024 * 1024):
    """Reuse cached ranges even if a previous downloader used another block size."""
    prefix = partial.stat().st_size if partial.exists() else 0
    if prefix > size:
        raise RuntimeError(f'Oversized partial file: {partial}')
    cached = {}
    for path in partial.parent.glob(partial.name + '.range_*'):
        suffix = path.name.removeprefix(partial.name + '.range_')
        start, end = (int(value) for value in suffix.split('_'))
        if not 0 <= start < end <= size or path.stat().st_size > end - start:
            raise RuntimeError(f'Invalid cached range: {path}')
        if start >= prefix:
            cached[start] = max(end, cached.get(start, end))
    ranges = []
    cursor = prefix
    while cursor < size:
        end = cached.get(cursor)
        if end is None:
            following = min((start for start in cached if start > cursor), default=size)
            end = min(cursor + block_size, following, size)
        ranges.append((cursor, end))
        cursor = end
    return ranges


def fetch_ranges(partial, url, size):
    """Bounded range concurrency; preserve the prefix and old cached chunks."""
    ranges = resumable_ranges(partial, size)
    def fetch(bounds):
        start, end = bounds
        chunk_file = partial.with_name(partial.name + f'.range_{start}_{end}')
        for attempt in range(6):
            downloaded = chunk_file.stat().st_size if chunk_file.exists() else 0
            if downloaded == end - start:
                return chunk_file
            if downloaded > end - start:
                raise RuntimeError('Oversized range partial')
            request = urllib.request.Request(url, headers={'Range': f'bytes={start + downloaded}-{end - 1}'})
            try:
                with urllib.request.urlopen(request, timeout=60) as response:
                    expected = f'bytes {start + downloaded}-{end - 1}/{size}'
                    if response.status != 206 or response.headers.get('Content-Range') != expected:
                        raise RuntimeError('Checkpoint endpoint must honor exact byte ranges')
                    with chunk_file.open('ab') as handle:
                        while data := response.read(4 * 1024 * 1024):
                            handle.write(data)
                if chunk_file.stat().st_size != end - start:
                    raise RuntimeError('Incomplete range')
                return chunk_file
            except Exception:
                if attempt == 5:
                    raise
                time.sleep(2)
    with ThreadPoolExecutor(max_workers=4) as executor:
        last_report = time.monotonic()
        # Ordered appends preserve a resumable contiguous prefix on interruption.
        with partial.open('ab') as handle:
            for bounds, chunk_file in zip(ranges, executor.map(fetch, ranges)):
                with chunk_file.open('rb') as source:
                    for data in iter(lambda: source.read(8 * 1024 * 1024), b''):
                        handle.write(data)
                handle.flush()
                chunk_file.unlink()
                if time.monotonic() - last_report >= 25:
                    print(f'DOWNLOAD {partial.name}: {bounds[1] / size:.1%}', flush=True)
                    last_report = time.monotonic()


def fetch_file(root, item):
    relative = Path(item['Path'])
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Unsafe repository path')
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    size, sha = int(item['Size']), item['Sha256']
    if target.exists():
        if target.stat().st_size != size or digest(target) != sha:
            raise RuntimeError(f'Refusing to overwrite an unverified file: {target}')
        print(f'VERIFIED existing {relative}', flush=True)
        return
    partial = target.with_name(target.name + '.part')
    url = BASE + '/repo?' + urlencode({'Revision': 'master', 'FilePath': str(relative)})
    parallel = size > 256 * 1024 * 1024
    if parallel:
        fetch_ranges(partial, url, size)
    for attempt in (() if parallel else range(6)):
        offset = partial.stat().st_size if partial.exists() else 0
        if offset == size:
            break
        if offset > size:
            raise RuntimeError(f'Oversized partial file: {partial}')
        try:
            headers = {'User-Agent': 'BeMRS-official-checkpoint-downloader'}
            if offset:
                headers['Range'] = f'bytes={offset}-'
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=60) as response:
                resume = offset > 0 and response.status == 206
                if resume and not response.headers.get('Content-Range', '').startswith(f'bytes {offset}-'):
                    raise RuntimeError('Unexpected range response')
                written = offset if resume else 0
                last_report = time.monotonic()
                with partial.open('ab' if resume else 'wb') as handle:
                    while chunk := response.read(8 * 1024 * 1024):
                        handle.write(chunk)
                        written += len(chunk)
                        if written > size:
                            raise RuntimeError('Response exceeded the official file size')
                        if time.monotonic() - last_report >= 25:
                            print(f'DOWNLOAD {relative}: {written / size:.1%}', flush=True)
                            last_report = time.monotonic()
            if partial.stat().st_size != size:
                raise RuntimeError('Incomplete download')
            break
        except Exception as exc:
            print(f'RETRY {relative} attempt={attempt + 1}: {type(exc).__name__}', flush=True)
            if attempt == 5:
                raise
            time.sleep(2)
    if digest(partial) != sha:
        raise RuntimeError(f'SHA256 mismatch; partial preserved for inspection: {partial}')
    partial.replace(target)
    print(f'VERIFIED {relative} bytes={size} sha256={sha}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=Path('/root/lhc/models/Qwen3-8B'))
    args = parser.parse_args()
    root = args.directory.resolve()
    root.mkdir(parents=True, exist_ok=True)
    url = BASE + '/repo/files?' + urlencode({'Revision': 'master', 'Recursive': 'true'})
    with urllib.request.urlopen(url, timeout=30) as response:
        listing = json.load(response)
    if not listing.get('Success'):
        raise RuntimeError('Official ModelScope repository listing failed')
    files = [item for item in listing['Data']['Files'] if item.get('Type') != 'tree']
    print(f'OFFICIAL CHECKPOINT {MODEL}: files={len(files)} bytes={sum(int(f["Size"]) for f in files)}', flush=True)
    # A detached resumable download must never race a second invocation.
    with (root / '.checkpoint-download.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('This checkpoint is already being downloaded') from exc
        with ThreadPoolExecutor(max_workers=4) as executor:
            list(executor.map(lambda item: fetch_file(root, item), files))
    config = json.loads((root / 'config.json').read_text())
    if config.get('torch_dtype', config.get('dtype')) != 'bfloat16' or config.get('quantization_config'):
        raise RuntimeError('Expected the unquantized official BF16 checkpoint')
    dtype_counts = {}
    for file in sorted(root.glob('model-*.safetensors')):
        with file.open('rb') as handle:
            length = struct.unpack('<Q', handle.read(8))[0]
            if length > 16 * 1024 * 1024:
                raise RuntimeError('Unexpected safetensors header size')
            header = json.loads(handle.read(length))
        for name, tensor in header.items():
            if name != '__metadata__':
                dtype = tensor['dtype']
                dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1
    if set(dtype_counts) != {'BF16'}:
        raise RuntimeError(f'Unexpected weight dtypes: {dtype_counts}')
    provenance = {'model': MODEL, 'source': BASE, 'revision': 'master',
                  'verified_at_utc': datetime.now(timezone.utc).isoformat(),
                  'quantization': None, 'weight_dtypes': dtype_counts,
                  'files': [{key: item[key] for key in ('Path', 'Size', 'Sha256')} for item in files]}
    (root / 'bemrs_checkpoint_provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(f'CHECKPOINT READY {root}: dtypes={dtype_counts}', flush=True)


if __name__ == '__main__':
    main()
