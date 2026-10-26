"""Maintainer-only: seal the assembled checkout, without reading its parent tree."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IGNORED = {'.git', '.venv', '__pycache__', 'runs', 'models', '.cache', 'build', 'dist'}


def files():
    for path in sorted(ROOT.rglob('*')):
        relative = path.relative_to(ROOT)
        if any(p in IGNORED or p.endswith('.egg-info') for p in relative.parts):
            continue
        if path.is_file() and relative.as_posix() != 'provenance/release_manifest.json':
            yield path, relative.as_posix()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write', action='store_true', required=True,
                        help='Explicitly regenerate hashes after reviewing edits')
    parser.parse_args()
    exported = ROOT / 'provenance/export_manifest.json'
    source_rows = {row['path']: row for row in json.loads(exported.read_bytes())['files']}
    rows = []
    for path, name in files():
        row = dict(path=name, bytes=path.stat().st_size,
                   sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        if name in source_rows:
            row['export_sha256'] = source_rows[name]['sha256']
            row['source_sha256'] = source_rows[name].get('source_sha256')
            row['release_modified'] = row['sha256'] != row['export_sha256']
        rows.append(row)
    output = ROOT / 'provenance/release_manifest.json'
    output.write_text(json.dumps(dict(schema=1, files=rows,
        note='Packaging integrity, not proof of GPU execution. Excludes only the manifest itself and local build/run/cache directories.'),
        indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(dict(files=len(rows))))


if __name__ == '__main__':
    main()
