#!/usr/bin/env python3
"""Read-only source provenance, usable over SSH stdin without remote installation."""
import base64
import hashlib
import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path

SOURCE_SUFFIXES = {'.py', '.bash', '.sh', '.yaml', '.yml', '.json', '.toml', '.lock', '.md', '.txt', '.xml', '.cpp', '.h', '.tex', '.bib', '.sty', '.cls'}


def snapshot(root):
    root = Path(root)
    def git(*args):
        return subprocess.check_output(['git', '-C', str(root), *args])
    tracked = git('ls-files', '-z').decode().split('\0')
    dirty = set((git('diff', '--name-only', 'HEAD', '-z') + git('ls-files', '--others', '--exclude-standard', '-z')).decode().split('\0'))
    submodules = {}
    for line in git('ls-files','--stage').decode().splitlines():
        if line.startswith('160000 '):
            name=line.split('\t',1)[1]
            if (root/name/'.git').exists():submodules[name]=snapshot(root/name)
    hashes = {}
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode='w:gz') as archive:
        for name in sorted(set(tracked) | dirty):
            p = root/name
            if not name or not p.is_file() or '.git' in p.parts:
                continue
            if p.suffix not in SOURCE_SUFFIXES and p.suffix and p.name not in ('Dockerfile', 'Makefile') and not p.name.startswith('.env'):
                continue
            if name.startswith(('logs/', 'videos/', 'benchmark_results/')):
                continue
            hashes[name] = hashlib.sha256(p.read_bytes()).hexdigest()
            if name in dirty:
                archive.add(p, arcname=name, recursive=False)
        # Submodules can contain dirty source without appearing in parent ls-files.
    return dict(root=str(root), head=git('rev-parse', 'HEAD').decode().strip(),
                branch=git('branch', '--show-current').decode().strip(),
                status=git('status', '--short').decode(),
                diff_base64=base64.b64encode(git('diff', '--binary', 'HEAD')).decode(),
                source_hashes=hashes, submodules=submodules, dirty_sources_tar_gz_base64=base64.b64encode(output.getvalue()).decode())


if __name__ == '__main__':
    result=snapshot(sys.argv[1])
    if len(sys.argv)>2:
        result['capture_exists']=(Path(sys.argv[1])/'DynaNav/benchmark_results'/sys.argv[2]).exists()
    print(json.dumps(result))
