"""Kairos correctness implementation."""
import json
import os
import random
import time
import urllib.request


def validate(args, helper, directory):
    from kairos._runtime.replay import dump, drain_owned_lmcache, reset_owned_lmcache, reset_prefix_cache
    if helper is None or not args.gapped_reuse:
        raise ValueError('Correctness fixture requires the private gapped helper')
    generator = random.Random(88291)
    prompt = [31000] + [generator.randrange(100, 30000) for _ in range(6143)]
    layouts = [
        dict(name='connected_gpu_cpu_frontier', gpu_keep_chunks=list(range(8)),
             cpu_keep_chunks=list(range(8, 23))),
        dict(name='cpu_after_leading_hole', gpu_keep_chunks=[],
             cpu_keep_chunks=list(range(1, 23))),
        dict(name='gpu_disjoint', gpu_keep_chunks=[2, 5, 9, 15, 20],
             cpu_keep_chunks=[]),
        dict(name='interleaved_two_tier', gpu_keep_chunks=[0, 2, 4, 6, 12, 18],
             cpu_keep_chunks=[1, 3, 7, 9, 10, 11, 15, 16, 20, 21, 22]),
        dict(name='partial_gpu_inside_cpu_restore', gpu_keep_chunks=[],
             gpu_keep_pages=list(range(32, 40)) + list(range(144, 152)),
             cpu_keep_chunks=[2, 9, 15, 16, 20, 21, 22]),
    ]
    results = []

    # logprobs invokes a separately compiled sampler path even when the model
    # compilation mode is zero. Warm precisely that path outside correctness
    # and performance timing; do not hide a cold compile as a restore timeout.
    warm = dict(model=args.model, prompt=[31002, 101, 102, 103], max_tokens=1,
        ignore_eos=True, temperature=0, seed=1, return_token_ids=True,
        logprobs=20, cache_salt='gapped-logprob-warmup',
        vllm_xargs=dict(cont_program_id='gapped-logprob-warmup', cont_final=True,
                        cont_experiment='v21-eager-a~logprob-warmup'))
    print(json.dumps(dict(event='gapped_logprob_warmup_start')), flush=True)
    req = urllib.request.Request(f'http://127.0.0.1:{args.port}/v1/completions',
        data=json.dumps(warm).encode(), headers={'Content-Type': 'application/json'})
    began = time.monotonic()
    with urllib.request.urlopen(req, timeout=600) as response:
        warm_result = json.load(response)
    if 'error' in warm_result or not warm_result['choices'][0].get('token_ids'):
        raise RuntimeError('Greedy/logprob warmup failed')
    dump(directory/'gapped-logprob-warmup.json', dict(elapsed_s=time.monotonic()-began,
        response=warm_result, included_in_benchmark=False))
    drain_owned_lmcache(helper)
    print(json.dumps(dict(event='gapped_logprob_warmup_done',
                         elapsed_s=time.monotonic()-began)), flush=True)

    def request(experiment, program, fixture=None):
        meta = dict(cont_program_id=program, cont_final=True,
                    cont_experiment=experiment)
        if fixture:
            # vLLM's public vllm_xargs schema accepts scalar/simple-list
            # values, not nested objects. Keep the fixture as a JSON string.
            meta['cont_gapped_fixture'] = json.dumps(fixture)
        payload = dict(model=args.model, prompt=prompt, max_tokens=8,
            ignore_eos=True, temperature=0, seed=1, return_token_ids=True,
            logprobs=20, cache_salt=program, vllm_xargs=meta)
        req = urllib.request.Request(f'http://127.0.0.1:{args.port}/v1/completions',
            data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
        started = time.monotonic()
        with urllib.request.urlopen(req, timeout=120) as response:
            result = json.load(response)
        if 'error' in result:
            raise RuntimeError(result['error'])
        choice = result['choices'][0]
        if len(choice.get('token_ids', ())) != 8:
            raise RuntimeError('Fixture requires actual output token IDs')
        return dict(choice=choice, elapsed_s=time.monotonic()-started,
                    usage=result['usage'])

    from kairos._runtime.compat.layered_policy import LAYERS
    registered=[m.split('~',1)[0] for group in getattr(args,'suite_groups',())
                for m in group['methods']]
    hybrid=[m for m in dict.fromkeys(registered) if any(
        m.startswith(name+'-partial-') and layer.recovery_route=='layout'
        for name,layer in LAYERS.items())]
    for layout in layouts:
        methods = (('v21-eager-a',) if os.environ.get('CONTINUUM_PREFIX_ONLY_CORRECTNESS') == '1'
                   else ('v21-eager-a', 'gapped-v21-partial-eager-a', *hybrid))
        for method in methods:
            experiment = method+'~fixture-'+layout['name']
            program = 'fixture-'+layout['name']
            reset_owned_lmcache(helper, directory, label=method+'-'+layout['name'])
            reset_prefix_cache(args.port)
            cold = request(experiment, program)
            drain_owned_lmcache(helper)
            restored = request(experiment, program, layout)
            drain_owned_lmcache(helper)
            expected, actual = cold['choice'], restored['choice']
            same_tokens = expected['token_ids'] == actual['token_ids']
            differences = [abs(a-b) for a, b in zip(
                expected['logprobs']['token_logprobs'],
                actual['logprobs']['token_logprobs'], strict=True)]
            events = [json.loads(line) for line in (directory/'scheduler.jsonl').read_text().splitlines()
                      if line.strip()]
            events = [event for event in events if event.get('experiment') == experiment]
            arrivals = [event for event in events if event['event'] == 'arrival']
            if len(arrivals) != 2:
                raise RuntimeError('Cannot attribute fixture events to its two requests')
            rid = arrivals[-1]['request']
            segments = [event for event in events if event['event'] == 'gapped_segments'
                        and event['request'] == rid]
            saved = sum(event['logical_tokens']-event['actual_queries'] for event in segments)
            row = dict(layout=layout, method=method, cold=cold, restored=restored,
                same_greedy_tokens=same_tokens, max_logprob_delta=max(differences),
                saved_queries=saved, segments=segments,
                synthetic=True, included_in_benchmark=False,
                numerical_tolerance='same greedy IDs; sampled logprob delta <= 0.1 in BF16')
            routes=[e for e in events if e['event']=='recovery_execution_route'
                    and e['request']==rid]
            if method in hybrid:
                expected_route=('prefix' if layout['name']=='connected_gpu_cpu_frontier'
                                else 'segmented')
                if len(routes)!=1 or routes[0]['route']!=expected_route:
                    raise RuntimeError('Controlled hybrid fixture used the wrong restore protocol')
                row['verified_route']=expected_route
            results.append(row)
            dump(directory/'gapped-correctness.json', results)
            if not same_tokens or max(differences) > .1:
                raise RuntimeError('Segmented restore differs from full recompute beyond tolerance')
            if method.startswith('gapped-') and saved <= 0 and not (
                    method in hybrid and row['verified_route']=='prefix'):
                raise RuntimeError('No actual gapped query skipping in serving fixture')
            print(json.dumps(dict(event='gapped_correctness_pass', method=method,
                layout=layout['name'], saved_queries=saved,
                max_logprob_delta=max(differences))), flush=True)
    if any('grantalloc' in m.split('~') for group in getattr(args, 'suite_groups', ())
           for m in group['methods']):
        from kairos._runtime.compat.grant_allocation_correctness import validate as validate_allocation
        validate_allocation(args, helper, directory)
    return results
