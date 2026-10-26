"""Kairos plan implementation."""
from dataclasses import dataclass


def prefill_query_budget(normal, decode_count, threshold, adaptive, fixed=None):
    """An explicit execution control, independent of cache placement."""
    if normal <= 0 or adaptive <= 0 or (fixed is not None and fixed <= 0):
        raise ValueError('Query budgets must be positive')
    if fixed is not None:
        return min(normal, fixed)
    return min(normal, adaptive) if threshold and decode_count > threshold else normal


def runs(indices):
    answer = []
    for i in sorted(set(indices)):
        if answer and answer[-1][1] == i:
            answer[-1] = (answer[-1][0], i + 1)
        else:
            answer.append((i, i + 1))
    return answer


def query_segments(begin, end, valid_pages, page=16, slots=8):
    """Missing intervals, with a final logits/progress anchor.

    Combining intervals recomputes the intervening hits, never drops missing
    queries. The anchor avoids zero-token worker requests and accounts for
    the native runner's contiguous-progress/sampling invariant.
    """
    if not 0 <= begin < end or slots < 1:
        raise ValueError("Invalid segmented budget")
    # Walk page intervals rather than individual tokens. This path also runs
    # inside bounded hypothetical recovery plans on the CPU control plane.
    anchor=max(begin,end-page)
    segments=[]
    for i in range(begin//page,(end+page-1)//page):
        a,b=max(begin,i*page),min(end,(i+1)*page)
        if i in valid_pages:a=max(a,anchor)
        if a>=b:continue
        if segments and segments[-1][1]==a:segments[-1]=(segments[-1][0],b)
        else:segments.append((a,b))
    if len(segments) > slots:
        # Merging a gap does not change any other gap's length. Select the
        # same smallest original gaps once; preserve the old leftmost tie.
        joined = set(sorted(range(len(segments) - 1),
            key=lambda j: (segments[j + 1][0] - segments[j][1], j))[:len(segments) - slots])
        merged = [segments[0]]
        for i, segment in enumerate(segments[1:]):
            if i in joined:
                merged[-1] = (merged[-1][0], segment[1])
            else:
                merged.append(segment)
        segments = merged
    return segments


@dataclass(frozen=True)
class Costs:
    query: float = 0.00003
    attention_pair: float = 0.000000005
    segment: float = 0.001
    load_token: float = 0.00001
    load_call: float = 0.001
    chunk: int = 256

    def compute(self, segments):
        return sum(self.query * (b - a) + self.attention_pair *
                   ((a + b + 1) * (b - a) / 2) + self.segment
                   for a, b in segments)

    def chunk_compute(self, index):
        return self.compute([(index * self.chunk, (index + 1) * self.chunk)])

    def estimate(self, gpu, cpu, count, selective=True):
        """Small dynamic program including segment startup and load calls.

        States indicate whether the previous chunk is being recomputed.
        A CPU hit may be loaded or recomputed. DP chooses a legal SERIAL
        plan, so this is a calibrated proxy, not a global JCT optimum.
        """
        states = {False: (0., ())}
        for i in range(count):
            options = ("gpu",) if i in gpu else (("load", "compute")
                if i in cpu and selective else ("load",) if i in cpu else ("compute",))
            next_states = {}
            for previous, (value, choices) in states.items():
                for action in options:
                    computing = action == "compute"
                    if action == "gpu":
                        delta = 0.
                    elif action == "load":
                        delta = self.load_token * self.chunk + self.load_call
                    else:
                        delta = self.chunk_compute(i) - (self.segment if previous else 0.)
                    candidate = (value + delta, choices + (action,))
                    if computing not in next_states or candidate[0] < next_states[computing][0]:
                        next_states[computing] = candidate
            states = next_states
        value, actions = min(states.values())
        return value, {i for i, action in enumerate(actions) if action == "load"}

    def loss(self, tier, index, gpu, cpu, count):
        before = self.estimate(gpu, cpu, count)[0]
        after = self.estimate(gpu - {index} if tier == "gpu" else gpu,
                              cpu - {index} if tier == "cpu" else cpu, count)[0]
        return max(0., after - before)

    def estimate_pages(self, gpu_pages, cpu_chunks, count, page=16):
        """Account for partially resident chunks without discarding GPU hits."""
        per_chunk = self.chunk // page
        states = {False: (0., ())}
        for i in range(count):
            missing = [j for j in range(i * per_chunk, (i + 1) * per_chunk)
                       if j not in gpu_pages]
            segments = [(a * page, b * page) for a, b in runs(missing)]
            next_states = {}
            for previous, (value, choices) in states.items():
                cost = self.compute(segments)
                if previous and segments and segments[0][0] == i * self.chunk:
                    cost -= self.segment
                end_compute = bool(segments and segments[-1][1] == (i + 1) * self.chunk)
                candidate = (value + cost, choices + ('compute',))
                if end_compute not in next_states or candidate[0] < next_states[end_compute][0]:
                    next_states[end_compute] = candidate
                if missing and i in cpu_chunks:
                    loaded = (value + self.load_token * self.chunk + self.load_call,
                              choices + ('load',))
                    if False not in next_states or loaded[0] < next_states[False][0]:
                        next_states[False] = loaded
            states = next_states
        value, choices = min(states.values())
        return value, {i for i, action in enumerate(choices) if action == 'load'}
