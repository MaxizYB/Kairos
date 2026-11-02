"""Intern immutable posterior vectors and expand them after serving exits."""

import json
from pathlib import Path

DEFINITION = "forecast_distribution_definition"
SUFFIX = "_kairos_suffix_distribution_ref"
OUTPUT = "_kairos_output_distribution_ref"


class Interner:
    def __init__(self):
        self.objects = {}
        self.next_id = 0

    def intern(self, atoms, mass):
        key = id(atoms), id(mass)
        saved = self.objects.get(key)
        if saved is not None and saved[0] is atoms and saved[1] is mass:
            return saved[2], None
        ref = self.next_id
        self.next_id += 1
        self.objects[key] = atoms, mass, ref
        return ref, dict(ref=ref, atoms=atoms, mass=mass)


def expand_rows(events):
    definitions = {}

    def expand(row, scope):
        row = dict(row)
        for ref, atoms, mass in (
            (SUFFIX, "atoms_s", "mass"),
            (OUTPUT, "output_atoms", "output_mass"),
        ):
            if ref in row:
                key = scope, row.pop(ref)
                if key not in definitions:
                    raise ValueError("Unknown posterior reference")
                row[atoms], row[mass] = definitions[key]
        return row

    for original in events:
        event = dict(original)
        scope = event.get("experiment")
        if event["event"] == DEFINITION:
            key = scope, event["ref"]
            if key in definitions:
                raise ValueError("Duplicate posterior definition")
            definitions[key] = event["atoms"], event["mass"]
            continue
        if event["event"] in ("program_service", "forecast_service"):
            for section in ("rows", "quotes"):
                event[section] = {
                    p: expand(row, scope) for p, row in event[section].items()
                }
        proof = event.get("continuity_j")
        if proof is not None:
            held = {
                p: dict(q, quote=expand(q["quote"], scope))
                if q["kind"] == "running-current-call"
                else q
                for p, q in proof["held"].items()
            }
            event["continuity_j"] = dict(proof, held=held)
        yield event


def expand_file(path):
    path = Path(path)
    compact = path.with_name(path.stem + "-compact" + path.suffix)
    if compact.exists():
        raise FileExistsError("Preserve original compact evidence")
    path.rename(compact)
    count = 0
    with compact.open() as source, path.open("x") as target:
        for row in expand_rows(json.loads(line) for line in source):
            target.write(json.dumps(row) + "\n")
            count += 1
    return dict(path=str(path), compact=str(compact), expanded_events=count)
