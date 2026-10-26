"""Kairos arrival implementation."""
import json
import time


class ArrivalPredictor:
    def __init__(self, forest, service, page):
        self.forest, self.service, self.page = forest, service, page
        self.previous, self.keep, self.seen = {}, {}, {}

    @staticmethod
    def common(a, b):
        n = 0
        for x, y in zip(a, b):
            if x != y:
                break
            n += 1
        return n

    def observe(self, observation):
        # The executor is FIFO and has one worker. Never read live engine
        # state, requests, KV managers, mutable histories or future outputs.
        started = time.perf_counter()
        p = observation['p']
        ids, outputs = observation['ids'], observation['outputs']
        previous = self.previous.get(p, ())
        keep, post, inference_s = 0, None, 0.
        if outputs:
            raw = observation['raw']
            messages = json.loads(raw) if isinstance(raw, str) else raw
            if (not isinstance(messages, list) or not messages or
                    messages[-1].get('role') == 'assistant' or not previous):
                raise ValueError('Missing causal pre-call history')
            positions = [i for i, m in enumerate(messages) if m.get('role') == 'assistant']
            if not positions:
                raise ValueError('Missing observed previous assistant')
            from kairos.messages import model_message, token_ids
            full = token_ids(self.forest.tokenizer.apply_chat_template(
                [model_message(m) for m in messages[:positions[-1]+1] if m['role'] != 'exit'],
                tokenize=True, add_generation_prompt=False, enable_thinking=False))
            overlap = self.common(previous, full)
            self.seen[p] = self.seen.get(p, 0.) + self.service(
                len(previous), max(1, len(full)-overlap), self.keep.get(p, 0))
            previous_kv = (list(previous) + full[overlap:])[:-1]
            keep = min(self.common(previous_kv, ids), len(ids)-1)//self.page*self.page
            inference = time.perf_counter()
            post = self.forest.predict(
                text=self.forest.tokenizer.decode(ids, skip_special_tokens=False),
                messages=messages, prompt_lengths=list(observation['inputs']),
                outputs=list(outputs), services=self.seen[p])
            inference_s = time.perf_counter()-inference
        self.previous[p], self.keep[p] = ids, keep
        return dict(observation=observation, keep=keep, services=self.seen.get(p, 0.),
                    prediction=post, seconds=time.perf_counter()-started,
                    inference_s=inference_s, worker_finished=time.perf_counter())
