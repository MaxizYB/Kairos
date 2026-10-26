"""Frozen program-honest CDF forest training algorithm."""
from collections import Counter
import hashlib
import numpy as np
from kairos.metrics import distribution

def group(name):
    return int(hashlib.sha256(('r-distribution-cv/' + name).encode()).hexdigest()[:8], 16) % 3


def weights(rows):
    counts = Counter(r['id'] for r in rows)
    return np.asarray([1 / counts[r['id']] for r in rows])


def features(row):
    # Exactly the existing 127 prefix-causal numerical features. No endpoint,
    # current output contract, observed future tool, identity or target.
    return row['old_features'] + row['features']


def cdf_targets(y, w, intervals=32):
    """Training-only quadrature of the integrated CDF squared-error loss."""
    order = np.argsort(y)
    cumulative = np.cumsum(w[order]) / w.sum()
    knots = np.interp(np.linspace(0, 1, intervals + 1), cumulative, y[order])
    knots = np.unique(np.r_[0., knots, max(y)])
    width = np.diff(knots)
    midpoints = (knots[:-1] + knots[1:]) / 2
    return (y[:, None] <= midpoints[None, :]) * np.sqrt(width), knots


def tree_structure(est):
    t = est.tree_
    return dict(feature=t.feature.tolist(), threshold=t.threshold.tolist(),
                left=t.children_left.tolist(), right=t.children_right.tolist())


def fit_forest(rows, kind='honest_r', seed=20261020):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.tree import ExtraTreeRegressor
    X = np.asarray([features(r) for r in rows], dtype=np.float32)
    y = np.asarray([r['target'] for r in rows])
    w = weights(rows)
    ids = sorted({r['id'] for r in rows})
    quadrature_knots = []
    trees = []
    if kind == 'qrf':
        model = RandomForestRegressor(n_estimators=48, max_depth=7, min_samples_leaf=24,
            max_features=.8, random_state=seed, n_jobs=1)
        model.fit(X, y, sample_weight=w)
        estimators = model.estimators_
    else:
        estimators = [None] * 48
    for k, est in enumerate(estimators):
        if kind == 'qrf':
            # Standard QRF: estimate with each tree's bootstrap population.
            estimation = np.asarray(model.estimators_samples_[k])
        else:
            structure_ids = {name for name in ids if
                int(hashlib.sha256(('r5-honesty/' + name + '/' + str(k)).encode()).hexdigest()[:8], 16) % 2}
            structure = np.asarray([j for j, r in enumerate(rows) if r['id'] in structure_ids])
            estimation = np.asarray([j for j, r in enumerate(rows) if r['id'] not in structure_ids])
            assert not ({rows[j]['id'] for j in structure} & {rows[j]['id'] for j in estimation})
            if kind == 'honest_cdf':
                # Even the response-grid construction uses structure labels
                # only; estimation Programs cannot affect the partition.
                target, knots = cdf_targets(y[structure], w[structure])
                quadrature_knots.append(knots.tolist())
            else:
                target = y[structure]
            est = ExtraTreeRegressor(max_depth=7, min_samples_leaf=24, max_features=.8,
                                     random_state=seed + k)
            est.fit(X[structure], target, sample_weight=w[structure])
        leaves = est.apply(X[estimation])
        tree = tree_structure(est)
        leaf_values = {}
        for node in np.flatnonzero(est.tree_.children_left < 0):
            selected = estimation[leaves == node]
            if not len(selected):
                selected = estimation
            atoms, mass = distribution([round(float(y[j]), 2) for j in selected],
                                       [float(w[j]) for j in selected])
            leaf_values[str(node)] = list(map(list, zip(atoms, mass)))
        tree['value'] = leaf_values
        trees.append(tree)
    return dict(schema='r-only-distribution-v1', kind=kind, features=127,
                trees=trees, training_ids=ids, training_states=len(rows),
                group_honesty=kind != 'qrf', crps_quadrature_knots=quadrature_knots or None,
                fitting='pre-call only; complete Programs; Program-balanced; frozen 48/depth7/leaf24/.8',
                seed=seed)


