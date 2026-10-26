import ast
from concurrent.futures import Future
import gzip
import importlib.util
import json
import math
import os
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from kairos.arrival import ArrivalPredictor
from kairos.causal import causal_features
from kairos.cost import distribution_benefit, recovery_seconds
from kairos.evaluate import evaluate_states
from kairos.features import features
from kairos.forest import SuffixForest
from kairos.metrics import crps, distribution, macro_average
from kairos.protection import CommitmentHoldPolicy
from kairos.run import affinity, expand, prepare, recorded_environment
from kairos.score import current_service, service_rank

ROOT = Path(__file__).resolve().parents[1]


class ScoreTests(unittest.TestCase):
    def test_no_suffix_is_current_cost(self):
        self.assertEqual(service_rank(2, 3, [0], [1], 'distribution')['score'], 3)

    def test_deterministic_equal_units(self):
        self.assertAlmostEqual(service_rank(2, 2, [8], [1], 'distribution')['score'], 10)

    def test_distribution_not_mean_plug_in(self):
        mixture = service_rank(1, 1, [0, 18], [.5, .5], 'distribution')
        mean = service_rank(1, 1, [0, 18], [.5, .5], 'mean')
        self.assertAlmostEqual(mean['score'], 10)
        self.assertAlmostEqual(mixture['score'], 1.9)

    def test_probabilities_normalize(self):
        self.assertAlmostEqual(distribution_benefit(2, [0, 4], [2, 2]), 2/3)

    def test_invalid_inputs(self):
        for atoms, mass in (([], []), ([0], [0]), ([-1], [1]), ([math.nan], [1])):
            with self.assertRaises(ValueError):
                service_rank(1, 1, atoms, mass, 'distribution')
        for standard, actual in ((0, 1), (1, -1), (math.inf, 1)):
            with self.assertRaises(ValueError):
                service_rank(standard, actual, [0], [1], 'distribution')


class LayoutTests(unittest.TestCase):
    profile = dict(beta=[0, .001, 0, 0], reload_fixed_seconds=.01,
                   reload_seconds_per_token=.0001)

    def test_cpu_hole_stops_restore(self):
        layout = dict(blocks=[], tokens=1024, chunks=4, cpu_chunks=[1, 2, 3])
        self.assertAlmostEqual(recovery_seconds(layout, (), self.profile), 1.024)

    def test_contiguous_gpu_cpu(self):
        layout = dict(blocks=list(range(16)), tokens=1024, chunks=4, cpu_chunks=[1, 2, 3])
        self.assertAlmostEqual(recovery_seconds(layout, (), self.profile), .01 + 768*.0001)

    def test_restored_not_double_charged(self):
        actual, restore = current_service(prompt=1024, output=8, computed=0,
            gpu_pages=0, host={}, profile=self.profile, page=16, step=.02, restored=True)
        self.assertEqual(restore, 0)
        self.assertAlmostEqual(actual, .16)

    def test_computed_prompt_not_double_charged(self):
        _, restore = current_service(prompt=1024, output=8, computed=1024,
            gpu_pages=0, host={}, profile=self.profile, page=16, step=.02)
        self.assertEqual(restore, 0)


class DistributionTests(unittest.TestCase):
    def test_crps_exact(self):
        self.assertEqual(crps([0], [1], 2), 2)
        self.assertAlmostEqual(crps([0, 2], [.5, .5], 1), .5)

    def test_duplicate_atoms(self):
        self.assertEqual(distribution([2, 0, 2], [1, 2, 1]), ([0., 2.], [.5, .5]))

    def test_program_macro_not_state_micro(self):
        rows = [dict(id='a', metrics=dict(x=0))]*9 + [dict(id='b', metrics=dict(x=2))]
        self.assertEqual(macro_average(rows)['x'], 1)

    def test_forest_matches_measured_golden(self):
        forest = SuffixForest(ROOT/'assets/forest.json', None)
        for probe in json.loads((ROOT/'tests/fixtures/forest_predictions.json').read_bytes()):
            self.assertEqual(forest.predict_features(probe['features']), probe['expected'])

    def test_forest_split_guard(self):
        forest = SuffixForest(ROOT/'assets/forest.json', None)
        with self.assertRaises(ValueError):
            SuffixForest(ROOT/'assets/forest.json', None, [next(iter(forest.training_ids))])

    def test_127_feature_schema(self):
        prompts, outputs = [10, 20], [5]
        base = features('visible error test', prompts, outputs)
        causal, _ = causal_features([dict(role='user', content='visible issue')],
                                    prompts, outputs, (1.,), False)
        self.assertEqual((len(base), len(causal), len(base)+len(causal)), (81, 46, 127))

    def test_frozen_predictor_metrics_reproduce(self):
        data = json.loads(gzip.decompress((ROOT/'data/training/causal_states.json.gz').read_bytes()))
        got = evaluate_states(data, SuffixForest(ROOT/'assets/forest.json', None))
        expected = json.loads((ROOT/'results/predictor.json').read_bytes())['metrics']
        for cohort in got:
            for subgroup in got[cohort]:
                for key, value in got[cohort][subgroup].items():
                    self.assertAlmostEqual(value, expected[cohort][subgroup][key], places=9)


class ProtectionTests(unittest.TestCase):
    def test_event_sweep_matches_bruteforce(self):
        samples = [.1, .2, .2, .8, 2]
        policy = CommitmentHoldPolicy(training={'tool': samples})
        for benefit in (0., .1, .5, 3.):
            for rate in (0., .3, 1.):
                candidates = sorted({0., *samples})
                utility = lambda t: sum(d<=t for d in samples)/len(samples)*benefit-rate*sum(min(d,t) for d in samples)/len(samples)
                best = max(candidates, key=utility)
                value, _ = policy.ttl('tool', benefit, opportunity_rate=rate)
                self.assertAlmostEqual(utility(value), utility(best))

    def test_no_support_no_lease(self):
        self.assertEqual(CommitmentHoldPolicy().ttl('tool', 2)[0], 0.)

    def test_live_tail_does_not_invent_deadline(self):
        policy = CommitmentHoldPolicy(training={'tool': [.1, .2]})
        policy.observe('tool', 50, 100)
        self.assertLessEqual(policy.ttl('tool', 100, opportunity_rate=0)[0], .2)

    def test_cold_observation_never_predicts(self):
        worker = ArrivalPredictor(None, None, 16)
        result = worker.observe(dict(p='p', request_id='r', ids=(1, 2), outputs=()))
        self.assertIsNone(result['prediction'])


class DispatchBoundaryTests(unittest.TestCase):
    """Run the exact exported method with fake engine state, not a fake GPU test."""
    @classmethod
    def setUpClass(cls):
        tree = ast.parse((ROOT/'kairos/dispatch.py').read_text())
        definition = next(n for n in tree.body if isinstance(n, ast.ClassDef))
        method = next(n for n in definition.body if isinstance(n, ast.FunctionDef)
                      and n.name == '_r5_dispatch_refinement')
        minimal = ast.ClassDef(name='DispatchHarness', bases=[], keywords=[], body=[method], decorator_list=[])
        module = ast.fix_missing_locations(ast.Module(body=[minimal], type_ignores=[]))
        scope = dict(time=time)
        exec(compile(module, 'kairos/dispatch.py', 'exec'), scope)
        cls.harness = scope['DispatchHarness']

    def make_harness(self, valid):
        obj = self.harness()
        obj.forecast_enabled = True
        obj.forecast_predictions = {p: dict(request=rid) for p, rid in valid.items()}
        obj.order = dict(a=0, b=1, c=2, d=3, e=4)
        obj.online_rank = dict(obj.order)
        obj.online_victim_rank = dict(obj.order)
        obj._forecast_collect = lambda: None
        obj._emit = lambda *args, **kwargs: None
        obj._forecast_quotes = lambda by, eligible: {p: dict(score={'b': 8, 'd': 2}[p]) for p in eligible}
        return obj

    def test_only_matching_mature_slots_move(self):
        obj = self.make_harness(dict(b='rb', d='rd'))
        by = {p: SimpleNamespace(request_id='r'+p) for p in 'bcd'}
        scores = {p: (i, i) for i,p in enumerate('bcd')}
        _, order = obj._r5_dispatch_refinement(by, list('abcde'), {}, {'b', 'd'}, scores, list('bcd'))
        self.assertEqual(order, list('dcb'))
        self.assertEqual(sorted(obj.online_rank, key=obj.online_rank.get), list('adcbe'))
        self.assertEqual(obj.online_victim_rank, obj.order)

    def test_missing_or_stale_prediction_retains_slots(self):
        obj = self.make_harness(dict(b='old-request', d='rd'))
        obj._forecast_quotes = lambda *args: self.fail('No quote RPC for fewer than two matches')
        by = {p: SimpleNamespace(request_id='r'+p) for p in 'bcd'}
        scores = {p: (i, i) for i,p in enumerate('bcd')}
        _, order = obj._r5_dispatch_refinement(by, list('abcde'), {}, {'b', 'd'}, scores, list('bcd'))
        self.assertEqual(order, list('bcd'))
        self.assertEqual(obj.online_rank, obj.order)


class ReleaseTests(unittest.TestCase):
    def test_intra_package_import_closure(self):
        for path in (ROOT/'kairos').rglob('*.py'):
            if 'vendor' in path.parts:
                continue
            package = '.'.join(path.relative_to(ROOT).with_suffix('').parts[:-1])
            for node in ast.walk(ast.parse(path.read_text())):
                names = []
                if isinstance(node, ast.Import):
                    names = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [importlib.util.resolve_name('.'*node.level+(node.module or ''), package)
                             if node.level else node.module or '']
                for name in names:
                    self.assertFalse(name.startswith(('continuum_compat', 'gapped_compat', 'Phase_continuum')))
                    if name.startswith('kairos.'):
                        base = ROOT.joinpath(*name.split('.'))
                        self.assertTrue(base.with_suffix('.py').exists() or (base/'__init__.py').exists(), (path,name))

    def test_two_configs_expand_independently(self):
        args = SimpleNamespace(method='kairos', model='/model', profile=ROOT/'assets/deployment_profile.json',
            reload_profile=ROOT/'assets/reload_profile.json', engine_cpus='0-1', helper_cpus='2-3', gpu=0)
        for method in ('kairos', 'continuum'):
            args.method = method
            cfg, engine, helper, env = prepare(args, Path('/tmp/out'), '/tmp/control.sock')
            self.assertIn('/model', engine)
            self.assertNotIn('--kv-cache-memory-bytes', engine)
            self.assertFalse(any('$' in value for value in env.values() if isinstance(value, str)))
            self.assertEqual(cfg['cpu_kv_gib'], 8)

    def test_inherited_secrets_not_serialized(self):
        env = dict(HF_TOKEN='must-not-be-recorded', SSH_AUTH_SOCK='/private', CUDA_VISIBLE_DEVICES='0')
        self.assertEqual(recorded_environment(dict(environment={}), env), dict(CUDA_VISIBLE_DEVICES='0'))

    def test_affinity_parser(self):
        self.assertEqual(affinity('0-2, 4'), {0, 1, 2, 4})
        with self.assertRaises(ValueError):
            affinity('4-2')


if __name__ == '__main__':
    unittest.main()
