"""One resource quotation shared by ready and running owners."""

from kairos.scheduling.priority import PACKAGE
from kairos.observability.distributions import DEFINITION, Interner, OUTPUT, SUFFIX

IMPLEMENTATION = "kairos-continuity-compiled-exact-v2"


class ContinuityQuotes:
    def _switch_experiment(self, name):
        changed = name != getattr(self, "experiment", None)
        result = super()._switch_experiment(name)
        if changed:
            self._continuity_quantum_cache = {}
            self._continuity_index_cache = {}
        return result

    def _quotes(self, ready, scored, forecasts, calls):
        scope = getattr(self, "_continuity_quote_scope", None)
        if scope is None:
            return super()._quotes(ready, scored, forecasts, calls)
        active = scope["running"]
        merged = dict(ready, **active)
        all_quotes = super()._quotes(
            merged, set(scored) | set(active), forecasts, calls
        )
        scope["quotes"] = {p: all_quotes[p] for p in active}
        return {p: all_quotes[p] for p in scored}

    def _emit(self, event, **fields):
        if event in ("program_registry", "forecast_registry"):
            fields.update(
                continuity_ordering=PACKAGE,
                continuity_formula="J=t/g; idle suffix probe with disjoint training quantum",
                running_dispatch_reordered=False,
            )
        if event in ("program_service", "forecast_service"):
            proof = self._continuity_j_proof
            if getattr(self, "_forecast_compact_log_installed", False):
                proof = dict(
                    proof,
                    held={
                        p: dict(q, quote=self._compact_quote(q["quote"]))
                        if q["kind"] == "running-current-call"
                        else q
                        for p, q in proof["held"].items()
                    },
                )
            fields["continuity_j"] = proof
        return super()._emit(event, **fields)

    def _compact_quote(self, quote):
        result = dict(quote)
        interner = self.__dict__.setdefault("_posterior_log_interner", Interner())
        for atoms, mass, key in (
            ("atoms_s", "mass", SUFFIX),
            ("output_atoms", "output_mass", OUTPUT),
        ):
            ref, definition = interner.intern(result.pop(atoms), result.pop(mass))
            if definition is not None:
                super()._emit(DEFINITION, **definition)
            result[key] = ref
        return result
