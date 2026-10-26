"""Kairos gpu index implementation."""
from __future__ import annotations


class GPUHistoryIndex:
    def __init__(self, lookup):
        self.lookup = lookup
        self.rows = {}
        self.watch = {}
        self.by_block = {}
        # Revision of tracked history coverage. Publishing a decode page
        # outside all watched histories does not change this revision.
        self.revision = 0

    def _forget_block(self, bid, pair):
        if bid is None:
            return
        owners = self.by_block.get(bid)
        if owners is not None:
            owners.discard(pair)
            if not owners:
                self.by_block.pop(bid, None)

    def remove(self, program):
        old = self.rows.pop(program, None)
        if old:
            self.revision += 1
            for i, bid in enumerate(old['blocks']):
                self._forget_block(bid, (program, i))
            for key in old['keys']:
                owners = self.watch.get(key, set())
                owners = {pair for pair in owners if pair[0] != program}
                if owners:
                    self.watch[key] = owners
                else:
                    self.watch.pop(key, None)

    def replace(self, program, keys):
        self.remove(program)
        keys = tuple(keys)
        ids = [self.lookup(key) for key in keys]
        masks = [0] * ((len(ids) + 15) // 16)
        for i, bid in enumerate(ids):
            if bid is not None:
                masks[i // 16] |= 1 << (i % 16)
        self.rows[program] = dict(keys=keys, blocks=ids,
            masks=masks,
            manifest_revision=self.revision + 1,
            frontier=next((i for i, bid in enumerate(ids) if bid is None), len(ids)))
        self.revision += 1
        for i, key in enumerate(keys):
            self.watch.setdefault(key, set()).add((program, i))
            if ids[i] is not None:
                self.by_block.setdefault(ids[i], set()).add((program, i))

    def changed(self, key):
        owners = self.watch.get(key)
        if not owners:
            return
        bid = self.lookup(key)
        for program, i in owners:
            row = self.rows[program]
            if row['blocks'][i] == bid:
                continue
            self.revision += 1
            self._forget_block(row['blocks'][i], (program, i))
            row['blocks'][i] = bid
            bit = 1 << (i % 16)
            if bid is None:
                row['masks'][i // 16] &= ~bit
            else:
                row['masks'][i // 16] |= bit
            if bid is not None:
                self.by_block.setdefault(bid, set()).add((program, i))
            if bid is None:
                row['frontier'] = min(row['frontier'], i)
            elif i == row['frontier']:
                while (row['frontier'] < len(row['blocks'])
                       and row['blocks'][row['frontier']] is not None):
                    row['frontier'] += 1

    def prefixes(self):
        return {p: row['frontier']//16 for p, row in self.rows.items()}

    def shrinks(self, victims):
        """A zero result proves no *GPU prefix* loss, independent of CPU state."""
        return any(i < self.rows[program]['frontier']
                   for bid in victims for program, i in self.by_block.get(bid, ()))

    def layouts(self, cpu):
        return [dict(cpu[p], program=p, blocks=list(row['blocks']))
                for p, row in self.rows.items() if p in cpu]

    def zero_restore_victims(self, candidates, cpu, count):
        """First LRU zero-loss pages, or None if the bounded scan is insufficient.

        The prefix objective has a nonnegative, monotone loss. Removing a
        zero-loss page leaves each owner's restore endpoint unchanged, so
        subsequent positive candidates cannot become zero through that
        removal. Thus the first ``count`` zero-loss pages are exactly the
        legacy joint greedy result. All shared owners must certify zero.
        This is a decision-equivalent fast path, not a new victim policy.
        CPU is the same committed, advisory snapshot used by legacy greedy.
        """
        if count < 0:
            raise ValueError('Negative victim demand')
        if not count:
            return []
        reaches = {}
        chosen = []
        for bid in candidates:
            zero = True
            for program, position in self.by_block.get(bid, ()):
                host = cpu.get(program)
                if host is None or host.get('state') == 'finished':
                    continue
                row = self.rows[program]
                frontier = row['frontier']
                if position >= frontier:
                    continue
                if program not in reaches:
                    chunks = host['chunks']
                    present = set(host['cpu_chunks'])
                    ends = [0] * (chunks+1)
                    ends[-1] = chunks*16
                    for i in range(chunks-1, -1, -1):
                        ends[i] = ends[i+1] if i in present else i*16
                    reaches[program] = (ends, max(frontier, ends[min(frontier//16, chunks)]))
                ends, before = reaches[program]
                after = max(position, ends[min(position//16, len(ends)-1)])
                if after < before:
                    zero = False
                    break
            if zero:
                chosen.append(bid)
                if len(chosen) == count:
                    return chosen
        return None

    def prefix_changes(self, victims, cpu):
        """Post-reuse hint without copying/scanning complete GPU histories."""
        ends = {p: row['frontier'] for p, row in self.rows.items() if p in cpu}
        for bid in victims:
            for program, position in self.by_block.get(bid, ()):
                if program in ends:
                    ends[program] = min(ends[program], position)
        return {p: dict(before_pages=self.rows[p]['frontier'], after_pages=end)
                for p, end in ends.items()}
