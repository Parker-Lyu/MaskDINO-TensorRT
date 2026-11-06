#!/usr/bin/env python3
"""Fetch pinned public sources or checksum-verified assets (stdlib only)."""
import argparse
import hashlib
import json
import subprocess
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    with path.open('rb') as stream:
        result = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
        return result.hexdigest()


def fetch_sources(lock):
    for name, spec in lock['sources'].items():
        dest = ROOT / 'third_party' / name
        if not dest.exists():
            dest.mkdir(parents=True)
            subprocess.run(['git', 'init', '-q', str(dest)], check=True)
            subprocess.run(['git', '-C', str(dest), 'fetch', '--depth=1', spec['url'], spec['commit']], check=True)
            subprocess.run(['git', '-C', str(dest), 'checkout', '--detach', 'FETCH_HEAD'], check=True)
        actual = subprocess.check_output(['git', '-C', str(dest), 'rev-parse', 'HEAD'], text=True).strip()
        if actual != spec['commit']:
            raise RuntimeError(f'{name}: expected {spec["commit"]}, got {actual}')
        print(f'{name}: {actual}', flush=True)


def fetch_assets(lock, directory):
    for spec in lock['assets']:
        dest = directory / spec['path']
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() and sha256(dest) == spec['sha256']:
            print(f'Verified {spec["path"]}', flush=True)
            continue
        print(f'Downloading {spec["url"]}', flush=True)
        temporary = dest.with_suffix(dest.suffix + '.partial')
        request = urllib.request.Request(spec['url'], headers={'User-Agent': 'MaskDINO-TensorRT/0.1'})
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open('wb') as output:
            while data := response.read(1024 * 1024):
                output.write(data)
        if sha256(temporary) != spec['sha256']:
            temporary.unlink()
            raise RuntimeError(f'Checksum mismatch: {spec["path"]}')
        temporary.replace(dest)
        print(f'Verified {spec["path"]}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('kind', choices=['sources', 'assets'])
    parser.add_argument('--directory', type=Path, default=Path.cwd())
    args = parser.parse_args()
    lock = json.loads((ROOT / 'dependencies.lock.json').read_text())
    if args.kind == 'sources':
        fetch_sources(lock)
    else:
        fetch_assets(lock, args.directory)
