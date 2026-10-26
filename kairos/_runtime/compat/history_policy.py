"""Kairos history policy implementation."""
from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class History:
    keys: tuple = ()
    state: str = "running"
    tool: str = ""
    departed: float = 0.0
    gpu_chunks: int = 0
    gpu_observed_at: float = -1.0
    gpu_event_tracked: bool = False
    retained_chunks: int = 0
    retain_until: float = 0.0


class HistoryPolicy:
    def __init__(self):
        self.programs: dict[str, History] = {}
        self.tool_history = {}
        self.beta = (0.0, 1e-5, 0.0, 0.0)
        self.reload_per_token = 0.0
        self.last_decisions = []

    def retain_prefix(self, program, resident_keys, now=None, budget=40, timeout=5.0):
        """Bounded eviction preference for a committed, useful CPU prefix.

        This is not a reader lock or a GPU reference. Arrivals release the
        preference. Total preferred payload never exceeds budget chunks, and
        a finite deadline prevents a stalled tool from retaining it forever.
        """
        now = time.monotonic() if now is None else now
        row = self.programs.get(program)
        if row is None or row.state != "tool":
            return 0
        resident = {key.base() for key in resident_keys}
        end = self.end(row.keys, resident)
        other = sum(h.retained_chunks for p, h in self.programs.items()
                    if p != program and h.state == "tool" and h.retain_until > now)
        row.retained_chunks = min(end, max(0, budget-other))
        row.retain_until = now+timeout if row.retained_chunks else 0.0
        return row.retained_chunks

    def configure(self, tool_history, beta, reload_per_token=0.0):
        self.tool_history = {name: [float(row[0]) for row in rows][-512:]
                             for name, rows in tool_history.items()}
        self.beta = tuple(float(x) for x in beta)
        self.reload_per_token = max(0.0, float(reload_per_token))

    def update(self, program, state, keys=None, tool="", observed_gap=None, now=None):
        if state not in ("running", "tool", "finished"):
            raise ValueError("Invalid CPU history lifecycle")
        row = self.programs.setdefault(program, History())
        if observed_gap is not None and row.tool:
            samples = self.tool_history.setdefault(row.tool, [])
            samples.append(max(0.0, float(observed_gap)))
            del samples[:-512]
        if keys is not None:
            # A replacement manifest may describe a different causal history
            # even when its length is unchanged. Do not carry event advice
            # across that boundary before the scheduler republishes it.
            row.gpu_event_tracked = False
            row.gpu_observed_at = -1.
            row.keys = tuple(keys)
        row.state = state
        if state != "tool":
            row.retained_chunks = 0
            row.retain_until = 0.0
        if state == "tool":
            row.tool = tool
            row.departed = time.monotonic() if now is None else now

    def observe_gpu(self, prefixes, now=None, event_tracked=False):
        """Advisory APC metadata, never a pin or a restoration certificate.

        Native request admission still revalidates GPU pages. Unreferenced
        cached pages may disappear immediately after this observation.
        """
        now = time.monotonic() if now is None else now
        for program, chunks in prefixes.items():
            row = self.programs.get(program)
            if row is not None:
                row.gpu_chunks = min(len(row.keys), max(0, int(chunks)))
                row.gpu_observed_at = now
                row.gpu_event_tracked = event_tracked

    def observed_chunks(self, row, now):
        # An explicitly registered native-index publisher withdraws coverage
        # synchronously before allocator reuse. Its unchanged prefix needs
        # no heartbeat; the ordinary sampler retains its original freshness
        # rule. Neither mode is permission to consume KV.
        return row.gpu_chunks if row.gpu_event_tracked or 0 <= now-row.gpu_observed_at <= .15 else 0

    def withdraw_gpu_pages(self, programs):
        for program, pages in programs.items():
            row = self.programs.get(program)
            if row is not None and row.gpu_event_tracked and pages:
                row.gpu_chunks = min(row.gpu_chunks, min(pages)//16)

    def timing(self, row, remaining_lease, now):
        if row.state != "tool":
            return 1.0, 0.0
        elapsed = max(0.0, now-row.departed)
        values = self.tool_history.get(row.tool, [])
        if len(values) <= 100:
            values = [x for samples in self.tool_history.values() for x in samples]
        residuals = [x-elapsed for x in values if x > elapsed]
        if not residuals:
            # Conservative extrapolation after the empirical tail; it is not
            # an oracle timeout and cannot extend any GPU reference.
            residuals = [max(0.05, elapsed)]
        need_after_lease = sum(x > remaining_lease for x in residuals)/len(residuals)
        return need_after_lease, sum(residuals)/len(residuals)

    @staticmethod
    def end(keys, resident, start=0):
        while start < len(keys) and keys[start].base() in resident:
            start += 1
        return start

    def saved_cost(self, chunks):
        tokens = chunks*256
        return max(0.0, (self.beta[1]-self.reload_per_token)*tokens
                   + self.beta[2]*tokens*(tokens+1)/2)

    def admission_order(self, keys, grants, now=None):
        """Use finite STORE slots for a CPU interval complementary to GPU.

        Keys already present are handled separately by the native wrapper.
        Beyond an observed GPU prefix admit in forward order; inside it admit
        backward from that boundary. No allocation, pin, copy, or promise of
        future coverage is made here. Unknown/stale layouts stay native.
        """
        now = time.monotonic() if now is None else now
        known = {key.base(): (program, i, row)
                 for program, row in self.programs.items()
                 for i, key in enumerate(row.keys)}
        groups, ranks = {}, {}
        for order, key in enumerate(keys):
            match = known.get(key.base())
            if match is None:
                group, phase, position = ("unknown",), 0, order
            else:
                program, index, row = match
                group = ("program", program)
                grant = grants.get(program)
                certified = min(len(row.keys), len(grant.keys)) if grant else 0
                if grant:
                    certified = next((i for i in range(certified)
                        if grant.keys[i].base() != row.keys[i].base()), certified)
                observed = self.observed_chunks(row, now)
                gpu = max(certified, observed)
                phase, position = ((0, index) if not gpu or index >= gpu else (1, -index))
            groups.setdefault(group, len(groups))
            ranks[key] = (groups[group], phase, position, order)
        return sorted(keys, key=ranks.__getitem__)

    def select(self, ordered, target, resident_keys, grants, mode, now=None, zero_loss_only=False,
               shelter_arrived_tail=False, reference_first=False):
        now = time.monotonic() if now is None else now
        resident = {key.base() for key in resident_keys}
        eligible = {key.base(): key for key in ordered}
        recency = {key.base(): n for n, key in enumerate(ordered)}
        known = {key.base(): (program, index)
                 for program, row in self.programs.items()
                 for index, key in enumerate(row.keys)}
        chosen, decisions = [], []
        while eligible and len(chosen) < max(0, target):
            candidates = []
            # Unknown warmup/output objects cannot be assigned a program
            # benefit. Keep their native order and log that classification.
            for base, key in eligible.items():
                if base not in known:
                    candidates.append(((0, 0, recency[base]), key,
                                       dict(program=None, kind="untracked")))
            for program, row in self.programs.items():
                positions = [i for i, key in enumerate(row.keys) if key.base() in resident]
                if not positions:
                    continue
                prefix_before = self.end(row.keys, resident)
                grant = grants.get(program)
                gpu_chunks = min(len(row.keys), len(grant.keys)) if grant else 0
                gpu_chunks = min(gpu_chunks, next((i for i in range(gpu_chunks)
                    if grant.keys[i].base() != row.keys[i].base()), gpu_chunks)) if grant else 0
                certified_chunks = gpu_chunks
                observed = self.observed_chunks(row, now)
                if mode in ("history-cover", "history-cover-wide", "history-horizon"):
                    gpu_chunks = max(gpu_chunks, observed)
                joined_before = self.end(row.keys, resident, gpu_chunks)
                lease = max(0.0, grant.deadline-now) if grant else 0.0
                late, residual = self.timing(row, lease, now)
                native_rank = min((recency[k.base()] for k in row.keys if k.base() in eligible),
                                  default=len(ordered))
                # Baseline removes only the CPU tail. Coverage policy may
                # remove a leading duplicate covered by recent GPU APC,
                # preserving the complementary CPU suffix. Never skip a
                # locked leading object, or create an interior CPU hole.
                indices = [max(positions)]
                covered_eligible = [i for i in positions
                                    if i < gpu_chunks and row.keys[i].base() in eligible]
                if mode in ("history-cover", "history-cover-wide", "history-horizon") and min(positions) < gpu_chunks:
                    # The wide variant may pass a reader-locked CPU head,
                    # but only inside the currently GPU-covered prefix.
                    # GPU/CPU joined reachability is unchanged by that hole;
                    # the next GPU victim decision still uses actual CPU keys.
                    indices.insert(0, min(covered_eligible) if
                                   mode == "history-cover-wide" and covered_eligible
                                   else min(positions))
                for index in dict.fromkeys(indices):
                    base = row.keys[index].base()
                    if base not in eligible:
                        continue
                    if (mode == "history-active" and row.state == "tool"
                            and now < row.retain_until and index < row.retained_chunks):
                        continue
                    key = eligible[base]
                    after = resident-{base}
                    prefix_after = self.end(row.keys, after)
                    joined_after = self.end(row.keys, after, gpu_chunks)
                    cpu_loss = self.saved_cost(prefix_before)-self.saved_cost(prefix_after)
                    joined_loss = self.saved_cost(joined_before)-self.saved_cost(joined_after)
                    value = (late*cpu_loss+(1-late)*joined_loss)/(0.05+residual)
                    if mode in ("history-cover", "history-cover-wide"):
                        # Measured current coverage, not a prediction that the
                        # APC prefix will survive until the next tool return.
                        value = joined_loss/(0.05+residual)
                    # history-horizon retains the expected loss above: a
                    # current duplicate can cease to cover a CPU gap after
                    # its acknowledged protection ends. Unpinned APC is not
                    # a future residency promise; no data or references change.
                    if row.state == "finished":
                        rank = (0, 0, native_rank)
                    elif mode == "history-active":
                        # A running label alone is not evidence. Only the
                        # acknowledged native-reference certificate counts.
                        # Retain positional metadata after GPU unpin; all
                        # removals stay at the CPU tail. Cap new backup debt
                        # to the bounded copy opportunity, including old gaps.
                        missing = sum(key.base() not in resident
                                      for key in row.keys[:certified_chunks])
                        if grant and index < certified_chunks and missing >= 8:
                            continue
                        if grant and index < certified_chunks and missing < 8:
                            rank = (1, -lease, native_rank)
                        elif cpu_loss <= 0:
                            rank = (2, 0, native_rank)
                        else:
                            rank = (3, cpu_loss, native_rank)
                    elif mode == "history-tail":
                        rank = (1, native_rank, -index)
                    else:
                        rank = (1, value, native_rank)
                        if reference_first:
                            # A native request reference and a TTL reference
                            # are stronger evidence than a sampled APC hit.
                            # This only breaks equal current-loss choices;
                            # it never authorizes a positive-loss deletion.
                            backed = index < certified_chunks
                            proof = (0 if row.state == "running" else 1) if backed else 2
                            horizon = -lease if backed and row.state == "tool" else 0.0
                            rank = (1, value, proof, horizon, native_rank)
                    candidates.append((rank, key, dict(program=program,
                        kind="finished" if row.state == "finished" else mode,
                        chunk=index, prefix_before=prefix_before, prefix_after=prefix_after,
                        joined_before=joined_before, joined_after=joined_after,
                        certified_gpu_chunks=certified_chunks, observed_gpu_chunks=observed,
                        gpu_observation_age_s=now-row.gpu_observed_at,
                        remaining_gpu_lease_s=lease, estimated_post_lease_probability=late,
                        estimated_residual_s=residual, value=value,
                        reference_backed=index < certified_chunks,
                        reference_first=reference_first,
                        covered_eligible_chunks=len(covered_eligible),
                        cpu_head_locked=row.keys[min(positions)].base() not in eligible)))
            if zero_loss_only:
                candidates = [item for item in candidates if item[2].get('kind') in ('untracked', 'finished')
                              or item[2]['joined_before'] == item[2]['joined_after']]
            if shelter_arrived_tail:
                # An arrived request can recompute the missing bridge shortly.
                # Its disconnected CPU tail is not worthless merely because
                # current lookup stops earlier. Never turn this preference
                # into a native lock, GPU pin, or knowledge of future turns.
                candidates = [item for item in candidates
                    if item[2].get('kind') in ('untracked', 'finished')
                    or self.programs[item[2]['program']].state != 'running'
                    or item[2]['chunk'] < max(item[2]['certified_gpu_chunks'],
                                             item[2]['observed_gpu_chunks'])]
            if not candidates:
                break
            _, key, detail = min(candidates, key=lambda x: x[0])
            chosen.append(key)
            decisions.append(detail)
            resident.discard(key.base())
            eligible.pop(key.base())
        self.last_decisions = decisions
        return chosen
