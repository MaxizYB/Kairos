"""Release-only checks; no torch, sklearn or vLLM import required."""
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from kairos.run import RUNTIME_ACTIVATION, prepare
from kairos._runtime.activation import RUNTIME_ENVIRONMENT
from kairos.verify import verify_model, verify_workload
from kairos._runtime.teacher_replay import replay

ROOT = Path(__file__).resolve().parents[1]


class ReleaseChecks(unittest.TestCase):
    def test_docs_have_resolvable_relative_links(self):
        for path in [ROOT/'README.md', *sorted((ROOT/'docs').glob('*.md'))]:
            for target in re.findall(r'\]\(([^)]+)\)', path.read_text()):
                if '://' not in target and not target.startswith('#'):
                    self.assertTrue((path.parent/target.split('#')[0]).exists(), (path, target))

    def test_public_narrative_has_no_historical_methods(self):
        for path in [ROOT/'README.md', *sorted((ROOT/'docs').glob('*.md'))]:
            self.assertIsNone(re.search(r'(?i)\b(?:r[5-8](?:-c)?|v(?:4|9|10|14|15|21))\b',
                                        path.read_text()), path)

    def test_runtime_activation_is_exact_and_non_oracle(self):
        self.assertEqual(set(RUNTIME_ACTIVATION), {'kairos', 'continuum'})
        tags = set(RUNTIME_ACTIVATION['kairos'].split('~'))
        self.assertIn('knownoutput', tags)
        self.assertIn('finishlogical', tags)
        self.assertNotIn('truth', RUNTIME_ACTIVATION['kairos'])
        self.assertEqual(set(RUNTIME_ENVIRONMENT), {'kairos', 'continuum'})

    def test_public_configs_do_not_expose_internal_activation_tags(self):
        for name in ('kairos', 'continuum'):
            config = json.loads((ROOT/'configs'/f'{name}.json').read_bytes())
            encoded = json.dumps(config)
            self.assertIsNone(re.search(r'CONTINUUM_R(?:5|82|83)(?:_|\"|$)', encoded))
        self.assertEqual(RUNTIME_ENVIRONMENT['kairos']['CONTINUUM_R82_FOREST'],
                         '${ROOT}/assets/forest.json')

    def test_wrong_checkpoint_fails_before_serving(self):
        with self.assertRaises(ValueError):
            verify_model(ROOT/'assets', include_weights=False)

    def test_all_model_files_listed(self):
        record=json.loads((ROOT/'results/environment.json').read_bytes())['model_verification']
        self.assertEqual(sum(x['file'].endswith('.safetensors') for x in record['files']), 5)
        for row in record['files']:
            self.assertEqual(len(row['sha256']),64)
            self.assertGreater(row['bytes'],0)

    def test_policy_environment_is_not_inherited(self):
        args=SimpleNamespace(method='kairos',model='/model',gpu=0,
            profile=ROOT/'assets/deployment_profile.json',reload_profile=ROOT/'assets/reload_profile.json',
            engine_cpus='0',helper_cpus='1')
        with patch.dict('os.environ',dict(CONTINUUM_SCALAR_RANK_TRUTH='/private/truth',
            CONTINUITY_KERNEL_DIAGNOSTIC='1',LMCACHE_ENABLE_ASYNC_LOADING='True',
            PYTHONPATH='/old-research-checkout')):
            _,_,_,env=prepare(args,ROOT/'runs/test','/tmp/kairos-test.sock')
        self.assertNotIn('CONTINUUM_SCALAR_RANK_TRUTH',env)
        self.assertNotIn('CONTINUITY_KERNEL_DIAGNOSTIC',env)
        self.assertEqual(env['LMCACHE_ENABLE_ASYNC_LOADING'],'False')
        self.assertEqual(env['PYTHONPATH'],str(ROOT))
        self.assertEqual(env['CONTINUUM_R5'], '1')
        self.assertEqual(env['CONTINUUM_R82'], '1')
        self.assertEqual(env['CONTINUUM_R83'], '1')

    def test_dynamic_vendor_paths_are_local_and_complete(self):
        directory=ROOT/'kairos/_runtime/vendor'
        for filename in ('lmcache-lazy-68a9c6b/lazy_offload_manager.py',
            'hicache-3dc9136/unified_tree_core.py','efficientagent-b21b12a/connector.py'):
            self.assertTrue((directory/filename).is_file())
        for path in (ROOT/'kairos').rglob('*.py'):
            if 'vendor' in path.parts:
                continue
            self.assertNotIn('/home/caohuilin',path.read_text())
            self.assertNotIn('Phase_continuum',path.read_text())

    def test_frozen_assets_have_expected_hashes(self):
        for file,expected in {
            'assets/forest.json':'cb3d0859f7c837bb4aa719cc2aaa3ca5e8ceec40fa593cd14ea55a286bab1274',
            'data/workload/trace.json':'4c896b93b14eff39bcc6e2fd0700ba59a1bfcfd785bd15e645d5e1a2ffeee2e8',
        }.items():
            self.assertEqual(hashlib.sha256((ROOT/file).read_bytes()).hexdigest(),expected)

    def test_visible_protocol_matches_all_replay_prompts(self):
        self.assertEqual(verify_workload(), 178)


class TeacherProtocolChecks(unittest.IsolatedAsyncioTestCase):
    async def test_exact_cache_identity_and_causal_current_request(self):
        payloads=[]
        class Response:
            status_code=200
            async def __aenter__(self):return self
            async def __aexit__(self,*args):return False
            async def aiter_lines(self):
                yield 'data: '+json.dumps(dict(id='call',choices=[dict(token_ids=[100])]))
                yield 'data: [DONE]'
        class Client:
            def __init__(self,*args,**kwargs):pass
            async def __aenter__(self):return self
            async def __aexit__(self,*args):return False
            def stream(self,*args,**kwargs):
                payloads.append(kwargs['json'])
                return Response()
        httpx=SimpleNamespace(AsyncClient=Client,Timeout=lambda *a,**k:None,
                              Limits=lambda **k:None)
        def turn(ids):
            return dict(added_token_ids=ids,output_tokens=1,gap_after_s=0.,
                source_gap_s=0.,source_cached_tokens=0,source_tool_failures=0,tool='test')
        row=dict(id='p',cache_salt='frontier:p',session_id='session',synthetic_user='user',
            cohort='cohort',source_turn_id='source',arrival_s=0.,predecessor=None,
            human_gap_original_s=0.,human_gap_replay_s=0.,turns=[turn([1,2]),turn([3])])
        visible={'p':[[dict(role='user',content='current')],
                      [dict(role='user',content='observed continuation')]]}
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            (root/'visible.json').write_text(json.dumps(visible))
            args=SimpleNamespace(model='model',port=123,request_timeout=1,run_timeout=2,
                                 reuse_engine=True,current_method='kairos')
            with patch.dict(sys.modules,dict(httpx=httpx)),patch.dict(os.environ,
                    CONTINUUM_WORK_MESSAGES=str(root/'visible.json')):
                got=await replay(args,[row],root)
            self.assertEqual(got['completed'],2)
            records=[json.loads(x) for x in (root/'requests.jsonl').read_text().splitlines()]
            self.assertEqual([r['session'] for r in records],['session','session'])
        self.assertEqual([p['cache_salt'] for p in payloads],['frontier:p','frontier:p'])
        self.assertEqual(payloads[1]['prompt'],[1,2,3])  # not generated output [100]
        for stage,payload in enumerate(payloads):
            meta=payload['vllm_xargs']
            self.assertEqual(meta['cont_oracle_gap'],0)
            self.assertEqual(json.loads(meta['cont_visible_messages']),visible['p'][stage])
            self.assertEqual(meta['cont_final'],stage==1)


if __name__=='__main__':
    unittest.main()
