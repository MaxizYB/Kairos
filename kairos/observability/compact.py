"""observability.distributions."""

from kairos.observability.distributions import Interner, DEFINITION, OUTPUT, SUFFIX


class Interner:
    def __init__(self):
        self.objects = {}
        self.next_id = 0

    def intern(self, atoms, mass):
        key = (id(atoms), id(mass))
        saved = self.objects.get(key)
        if saved is not None and saved[0] is atoms and (saved[1] is mass):
            return (saved[2], None)
        ref = self.next_id
        self.next_id += 1
        self.objects[key] = (atoms, mass, ref)
        return (ref, dict(ref=ref, atoms=atoms, mass=mass))


DEFINITION = "forecast_distribution_definition"
SUFFIX = "_kairos_suffix_distribution_ref"
OUTPUT = "_kairos_output_distribution_ref"


class CompactDistributions:
    def _emit(self, event, **fields):
        emit = super()._emit
        if event not in ("program_service", "forecast_service"):
            if event in ("program_registry", "forecast_registry"):
                self._posterior_log_interner = Interner()
                fields["distribution_log"] = (
                    "lossless reference records; expanded after serving"
                )
            return super()._emit(event, **fields)
        if not hasattr(self, "_posterior_log_interner"):
            self._posterior_log_interner = Interner()

        def record(value, atoms_key, mass_key, ref_key):
            result = dict(value)
            ref, definition = self._posterior_log_interner.intern(
                result.pop(atoms_key), result.pop(mass_key)
            )
            if definition is not None:
                emit(DEFINITION, **definition)
            result[ref_key] = ref
            return result

        fields["rows"] = {
            p: record(r, "atoms_s", "mass", SUFFIX) for p, r in fields["rows"].items()
        }
        fields["quotes"] = {
            p: record(
                record(q, "atoms_s", "mass", SUFFIX),
                "output_atoms",
                "output_mass",
                OUTPUT,
            )
            for p, q in fields["quotes"].items()
        }
        return super()._emit(event, **fields)
