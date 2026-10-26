"""Rebuild the frozen forest or reproduce its group-disjoint OOF metrics."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

from kairos.forest import SuffixForest
from kairos.metrics import evaluate, macro_average

ROOT = Path(__file__).resolve().parents[1]


def training_rows(data):
    programs = [p for p in data['programs'] if p['split']=='train' and
                p['complete'] and p['within_16k']]
    return [dict(r,id=p['id']) for p in programs for r in p['states'] if not r['after']]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--states',type=Path,default=ROOT/'data/training/causal_states.json.gz')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--cross-validate',action='store_true')
    args=parser.parse_args()
    from kairos.training_algorithm import fit_forest, group, features
    from threadpoolctl import threadpool_limits
    data=json.loads(gzip.decompress(args.states.read_bytes()))
    rows=training_rows(data)
    if (len({r['id'] for r in rows}),len(rows)) != (157,2106):
        raise ValueError('Unexpected frozen training domain')
    args.output.mkdir(parents=True,exist_ok=False)
    with threadpool_limits(limits=1):
        if args.cross_validate:
            scored=[]
            for fold in range(3):
                fitting=[r for r in rows if group(r['id'])!=fold]
                checking=[r for r in rows if group(r['id'])==fold]
                assert not ({r['id'] for r in fitting}&{r['id'] for r in checking})
                path=args.output/f'fold-{fold}.json'
                path.write_text(json.dumps(fit_forest(fitting,'honest_cdf'),separators=(',',':'))+'\n')
                forest=SuffixForest(path,None)
                for row in checking:
                    scored.append(dict(id=row['id'],metrics=evaluate(
                        forest.predict_features(features(row)),row['target'])))
            (args.output/'oof_metrics.json').write_text(json.dumps(macro_average(scored),indent=2)+'\n')
        artifact=fit_forest(rows,'honest_cdf')
        path=args.output/'forest.json'
        path.write_text(json.dumps(artifact,separators=(',',':'),allow_nan=False)+'\n')
        original=ROOT/'assets/forest.json'
        report=dict(programs=157,states=len(rows),trees=48,seed=20261020,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            identical_to_bundled=json.loads(path.read_bytes())==json.loads(original.read_bytes()))
        (args.output/'training_report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report))


if __name__=='__main__':main()
