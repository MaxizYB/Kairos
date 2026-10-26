"""Kairos grant allocation correctness implementation."""
import json
import random
import urllib.request


def validate(args, helper, directory):
    from kairos._runtime.replay import dump, drain_owned_lmcache, reset_owned_lmcache, reset_prefix_cache

    rng = random.Random(88291)
    prompt = [31000] + [rng.randrange(100, 30000) for _ in range(6143)]
    layouts = [None, dict(name='connected_gpu_cpu_frontier',
        gpu_keep_chunks=list(range(8)), cpu_keep_chunks=list(range(8, 23)))]
    baseline = 'v21-eager-a~onlinefitresidual~knownoutput~finishlogical'
    candidate = baseline + '~grantalloc'
    rows = []

    def request(method, program, salt, output, fixture=None):
        meta = dict(cont_experiment=method, cont_program_id=program, cont_final=True)
        if fixture is not None:
            meta['cont_gapped_fixture'] = json.dumps(fixture)
        payload = dict(model=args.model, prompt=prompt, max_tokens=output,
            ignore_eos=True, temperature=0, seed=1, return_token_ids=True,
            logprobs=20, cache_salt=salt, vllm_xargs=meta)
        req = urllib.request.Request(f'http://127.0.0.1:{args.port}/v1/completions',
            data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=120) as response:
            result = json.load(response)
        if 'error' in result:
            raise RuntimeError(result['error'])
        choice = result['choices'][0]
        if len(choice.get('token_ids', ())) != output:
            raise RuntimeError('Allocation fixture did not finish the full output')
        drain_owned_lmcache(helper)
        return choice

    for layout in layouts:
        name = layout['name'] if layout else 'cold'
        choices = []
        for arm, method in [('final', baseline), ('grantalloc', candidate)]:
            experiment = method + '~allocation-check-' + name
            program = 'allocation-check-' + arm + '-' + name
            salt = 'allocation-check-' + name
            reset_owned_lmcache(helper, directory, label=program)
            reset_prefix_cache(args.port)
            if layout:
                # Method switches intentionally reset APC. Seed inside the
                # same experiment, with a separate completed Program so the
                # online terminal marker is not reused by the actual check.
                request(experiment, program + '-seed', salt, 1)
            choices.append(request(experiment, program, salt, 33, layout))
            events = [json.loads(line) for line in
                      (directory/'scheduler.jsonl').read_text().splitlines()]
            events = [e for e in events if e.get('experiment') == experiment]
            allocations = [e for e in events if e['event'] == 'grant_allocation']
            if arm == 'grantalloc':
                if len(allocations) != 1 or allocations[0]['allocated_pages'] < 2:
                    raise RuntimeError('Real allocation check did not allocate multiple future pages')
                if allocations[0]['scheduled_tokens'] != 1:
                    raise RuntimeError('Preallocation changed the actual decode work')
            elif allocations:
                raise RuntimeError('Baseline unexpectedly enabled grant allocation')
        same = choices[0]['token_ids'] == choices[1]['token_ids']
        delta = max(abs(a-b) for a, b in zip(
            choices[0]['logprobs']['token_logprobs'],
            choices[1]['logprobs']['token_logprobs'], strict=True))
        row = dict(layout=name, same_greedy_tokens=same, max_logprob_delta=delta,
            output_tokens=33, synthetic=True, included_in_benchmark=False)
        rows.append(row)
        dump(directory/'grant-allocation-correctness.json', rows)
        if not same or delta > .1:
            raise RuntimeError('Grant allocation changed real-model results beyond BF16 tolerance')
        print(json.dumps(dict(event='grant_allocation_correctness_pass', **row)), flush=True)
    return rows
