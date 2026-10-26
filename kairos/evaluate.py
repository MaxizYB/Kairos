"""Evaluate the frozen suffix distribution without GPU or ML dependencies."""
import argparse
import gzip
import json
from pathlib import Path

from kairos.forest import SuffixForest
from kairos.metrics import evaluate, macro_average

ROOT = Path(__file__).resolve().parents[1]


def evaluate_states(data, forest):
    results = {}
    for label, split in (('heldout36', 'blind'), ('swe_complete14', 'development')):
        rows = [dict(r, id=p['id']) for p in data['programs']
                if p['split'] == split and p['complete'] and p['within_16k']
                for r in p['states'] if not r['after']]
        if {r['id'] for r in rows} & forest.training_ids:
            raise ValueError('Training/evaluation identity overlap')
        scored = [dict(id=r['id'], stage=r['stage'], target=r['target'],
                       metrics=evaluate(forest.predict_features(
                           r['old_features'] + r['features']), r['target']))
                  for r in rows]
        groups = dict(all=scored,
                      cold=[r for r in scored if r['stage'] == 0],
                      warm=[r for r in scored if r['stage'] > 0],
                      warm_positive=[r for r in scored if r['stage'] > 0 and r['target'] > 0])
        results[label] = {name: dict(programs=len({r['id'] for r in group}),
                                      states=len(group), **macro_average(group))
                          for name, group in groups.items()}
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--forest', type=Path, default=ROOT/'assets/forest.json')
    parser.add_argument('--states', type=Path, default=ROOT/'data/training/causal_states.json.gz')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    data = json.loads(gzip.decompress(args.states.read_bytes()))
    result = evaluate_states(data, SuffixForest(args.forest, None))
    if args.output:
        with args.output.open('x') as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write('\n')
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
