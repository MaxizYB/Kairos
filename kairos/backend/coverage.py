"""backend.coverage."""

import time
from dataclasses import dataclass
from kairos._runtime.compat.history_policy import HistoryPolicy


@dataclass(frozen=True)
class CoverageView:
    owners: dict
    gpu: dict
    certified: dict


class CoverageHistory(HistoryPolicy):
    def coverage(self, grants, now):
        owners, gpu, certified = ({}, {}, {})
        for p, row in self.programs.items():
            for i, key in enumerate(row.keys):
                owners.setdefault(key.base(), []).append((p, i))
            grant = grants.get(p)
            n = min(len(row.keys), len(grant.keys)) if grant else 0
            n = next(
                (i for i in range(n) if row.keys[i].base() != grant.keys[i].base()), n
            )
            certified[p] = n
            gpu[p] = min(len(row.keys), max(n, self.observed_chunks(row, now)))
        return CoverageView(owners, gpu, certified)

    def write_budget(self, keys, present, free_slots, grants, now):
        view = self.coverage(grants, now)
        present = {key.base() for key in present}
        absent = [key for key in keys if key.base() not in present]
        if len(absent) <= free_slots:
            return (list(keys), [], dict(pressure=False))
        essential, duplicates = (set(), {})
        for key in absent:
            owners = view.owners.get(key.base(), ())
            if not owners or any((i >= view.gpu[p] for p, i in owners)):
                essential.add(key.base())
            else:
                for p, i in owners:
                    duplicates.setdefault(p, []).append((i, key))
        extras = []
        for group in duplicates.values():
            group.sort(reverse=True, key=lambda item: item[0])
            essential.add(group[0][1].base())
            extras.extend((key for _, key in group[1:]))
        extras = list(
            dict.fromkeys((k.base() for k in extras if k.base() not in essential))
        )
        essential.update(extras[: max(0, free_slots - len(essential))])
        kept = [key for key in keys if key.base() in present or key.base() in essential]
        skipped = [
            key
            for key in keys
            if key.base() not in present and key.base() not in essential
        ]
        return (
            kept,
            skipped,
            dict(
                pressure=True,
                free_slots=free_slots,
                essential_chunks=len(essential),
                missing_chunks=len(absent),
                bridge_chunks_per_program=1,
                all_owners_checked=True,
            ),
        )

    def select(
        self,
        ordered,
        target,
        resident_keys,
        grants,
        mode,
        now=None,
        zero_loss_only=False,
        shelter_arrived_tail=False,
        reference_first=False,
    ):
        if (
            mode != "history-cover"
            or not zero_loss_only
            or shelter_arrived_tail
            or reference_first
        ):
            return super().select(
                ordered,
                target,
                resident_keys,
                grants,
                mode,
                now,
                zero_loss_only,
                shelter_arrived_tail,
                reference_first,
            )
        now = time.monotonic() if now is None else now
        resident = {k.base() for k in resident_keys}
        eligible = {k.base(): k for k in ordered}
        recency = {k.base(): i for i, k in enumerate(ordered)}
        view = self.coverage(grants, now)
        owners, gpu, certified = (view.owners, view.gpu, view.certified)
        chosen, decisions = ([], [])
        while eligible and len(chosen) < max(0, target):
            candidates = []
            positions = {
                p: [i for i, k in enumerate(row.keys) if k.base() in resident]
                for p, row in self.programs.items()
            }
            before = {
                p: self.end(row.keys, resident, gpu[p])
                for p, row in self.programs.items()
            }
            for base, key in eligible.items():
                if base not in owners:
                    candidates.append(
                        (
                            (0, 0, recency[base]),
                            key,
                            dict(program=None, kind="untracked"),
                        )
                    )
            for p, row in self.programs.items():
                pos = positions[p]
                if not pos:
                    continue
                indices = [max(pos)]
                if min(pos) < gpu[p]:
                    indices.insert(0, min(pos))
                native_rank = min(
                    (recency[k.base()] for k in row.keys if k.base() in eligible),
                    default=len(ordered),
                )
                for i in dict.fromkeys(indices):
                    base = row.keys[i].base()
                    if base not in eligible:
                        continue
                    after = resident - {base}
                    losses = {
                        other: before[other]
                        - self.end(self.programs[other].keys, after, gpu[other])
                        for other, _ in owners[base]
                        if self.programs[other].state != "finished"
                    }
                    if any(losses.values()):
                        continue
                    joined_after = self.end(row.keys, after, gpu[p])
                    detail = dict(
                        program=p,
                        kind="finished" if row.state == "finished" else mode,
                        chunk=i,
                        prefix_before=self.end(row.keys, resident),
                        prefix_after=self.end(row.keys, after),
                        joined_before=before[p],
                        joined_after=joined_after,
                        certified_gpu_chunks=certified[p],
                        observed_gpu_chunks=self.observed_chunks(row, now),
                        value=0.0,
                        all_live_owner_losses=losses,
                        owner_count=len(owners[base]),
                        decision_basis="current joint coverage; native recency on equal loss",
                        timing_evaluated=False,
                    )
                    candidates.append(
                        (
                            (0 if row.state == "finished" else 1, 0, native_rank),
                            eligible[base],
                            detail,
                        )
                    )
            if not candidates:
                break
            _, key, detail = min(candidates, key=lambda item: item[0])
            chosen.append(key)
            decisions.append(detail)
            resident.discard(key.base())
            eligible.pop(key.base())
        self.last_decisions = decisions
        return chosen
