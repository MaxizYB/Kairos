"""Read-only integrity checks for a source checkout and the pinned checkpoint."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def verify_checkout():
    manifest = json.loads((ROOT / 'provenance/release_manifest.json').read_bytes())
    for row in manifest['files']:
        path = ROOT / row['path']
        if not path.resolve().is_relative_to(ROOT):
            raise ValueError('Manifest escapes checkout')
        if not path.is_file() or digest(path) != row['sha256']:
            raise ValueError('Missing or modified release file: ' + row['path'])
    return len(manifest['files'])


def verify_workload():
    """Verify current trace token signatures against its causal message sidecar."""
    protocol=json.loads((ROOT/'data/workload/visible_protocol.json').read_bytes())
    if (protocol['visible_messages_sha256'] != digest(ROOT/'data/workload/visible_messages.json')
            or protocol['current_assistant_included'] is not False
            or protocol['future_outcomes_included'] is not False):
        raise ValueError('Invalid visible-message protocol')
    signatures={(r['program'],r['stage']):r for r in protocol['request_signatures']}
    programs=json.loads((ROOT/'data/workload/trace.json').read_bytes())['programs']
    seen=set()
    for program in programs:
        tokens=[]
        for stage,turn in enumerate(program['turns']):
            tokens=tokens[:turn.get('keep_prefix_tokens',len(tokens))]+turn['added_token_ids']
            identity=(program['id'],stage)
            raw=json.dumps(tokens,separators=(',',':')).encode()
            if hashlib.sha256(raw).hexdigest()!=signatures[identity]['prompt_sha256']:
                raise ValueError('Prompt/visible-message signature differs')
            seen.add(identity)
    if seen != set(signatures):
        raise ValueError('Visible-message protocol has a different call set')
    return len(seen)


def verify_model(model, *, include_weights=True):
    """Check recorded bytes, not merely an HF directory name. No downloads."""
    root = Path(model).resolve()
    record = json.loads((ROOT / 'results/environment.json').read_bytes())['model_verification']
    checked = []
    for row in record['files']:
        if not include_weights and row['file'].endswith('.safetensors'):
            continue
        path = root / row['file']
        if not path.is_file() or path.stat().st_size != row['bytes'] or digest(path) != row['sha256']:
            raise ValueError('Missing or different checkpoint file: ' + row['file'])
        checked.append(row['file'])
    return dict(revision=record['revision'], files_verified=checked,
                weights_verified=include_weights)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path)
    parser.add_argument('--metadata-only', action='store_true',
                        help='Skip the five large weight hashes; not full model verification')
    args = parser.parse_args()
    result = dict(release_files_verified=verify_checkout(), calls_verified=verify_workload())
    if args.model:
        result['checkpoint'] = verify_model(args.model, include_weights=not args.metadata_only)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
