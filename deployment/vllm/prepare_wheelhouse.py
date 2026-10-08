"""Reuse existing verified pip wheel-cache bodies without editing their source."""
import email
import os
from pathlib import Path
import zipfile

ROOT = Path('/root/lhc/bemrs-vllm-wheelhouse')
CACHE = Path('/root/.cache/pip/http-v2')


if __name__ == '__main__':
    ROOT.mkdir(parents=True, exist_ok=True)
    count = 0
    for path in CACHE.rglob('*.body'):
        try:
            with zipfile.ZipFile(path) as archive:
                record = next((name for name in archive.namelist() if name.endswith('.dist-info/WHEEL')), None)
                if not record:
                    continue
                wheel = email.message_from_bytes(archive.read(record))
                info = email.message_from_bytes(archive.read(record.rsplit('/', 1)[0] + '/METADATA'))
                tag = wheel.get('Tag')
                name = info['Name'].replace('-', '_')
                target = ROOT / f'{name}-{info["Version"]}-{tag}.whl'
                if tag and not target.exists():
                    os.link(path, target)
                    count += 1
        except zipfile.BadZipFile:
            pass
    print(f'Cached wheels linked: {count}; wheelhouse={ROOT}')
