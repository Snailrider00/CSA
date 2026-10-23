#!/usr/bin/env python3
"""Install CSA trainers into an isolated nnU-Net 2.8.1 environment."""
import argparse
import hashlib
import importlib.metadata
import importlib.util
from pathlib import Path
import shutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target', type=Path, help='Root containing nnunetv2 (default: installed package parent)')
    parser.add_argument('--replace', action='store_true', help='Replace differing existing files, retaining .csa-backup copies')
    args = parser.parse_args()
    version = importlib.metadata.version('nnunetv2')
    if version != '2.8.1':
        raise RuntimeError(f'Expected nnunetv2 2.8.1, found {version}')
    spec = importlib.util.find_spec('nnunetv2')
    target = args.target or Path(next(iter(spec.submodule_search_locations))).parent
    source = Path(__file__).resolve().parents[1] / 'nnunet_overlay'
    files = sorted(source.rglob('*.py'))
    for src in files:
        dst = target / src.relative_to(source)
        if dst.exists() and src.read_bytes() != dst.read_bytes() and not args.replace:
            raise RuntimeError(f'Refusing to replace {dst}; use a fresh environment or --replace')
    for src in files:
        dst = target / src.relative_to(source)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() and dst.read_bytes() != src.read_bytes():
            backup = dst.with_suffix('.py.csa-backup')
            if not backup.exists():
                shutil.copy2(dst, backup)
        shutil.copy2(src, dst)
        print(src.relative_to(source))
    print(f'Installed {len(files)} CSA source files into {target}')


if __name__ == '__main__':
    main()
