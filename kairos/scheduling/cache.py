"""Request-scoped immutable law reductions and live resource quotations."""

from collections import OrderedDict
from kairos.scheduling.score import current_service, service_rank

PACKAGE = "cached-service"


class ServiceCache:
    def _switch_experiment(self, name):
        changed = name != getattr(self, "experiment", None)
        result = super()._switch_experiment(name)
        if changed:
            self.__dict__.pop("_compiled_stats", None)
            self._service_cache_state()
        return result

    def _current_call(self, request):
        stats = self._service_cache_state()
        p = self.program(request)
        post = getattr(self, "forecast_output_cache", {}).get(p, {}).get("post")
        key = (
            request.request_id,
            post,
            len(request.output_token_ids),
            int(request.max_tokens),
            request.num_prompt_tokens,
        )
        saved = self._compiled_calls.get(p)
        if (
            saved is not None
            and saved[0][0] == key[0]
            and (saved[0][1] is post)
            and (saved[0][2:] == key[2:])
        ):
            stats["call_hits"] += 1
            return saved[1]
        result = super()._current_call(request)
        post = getattr(self, "forecast_output_cache", {}).get(p, {}).get("post")
        self._compiled_calls[p] = ((key[0], post, *key[2:]), result)
        stats["call_misses"] += 1
        return result

    def _service(self, prompt, output, keep):
        stats = self._service_cache_state()
        key = (prompt, output, keep)
        cache = self._compiled_service
        try:
            value = cache[key]
        except KeyError:
            pass
        else:
            stats["service_hits"] += 1
            return value
        result = super()._service(prompt, output, keep)
        cache[key] = result
        if len(cache) > 4096:
            cache.popitem(last=False)
        stats["service_misses"] += 1
        return result

    def _quotes(self, ready, mature, forecasts, calls):
        stats = self._service_cache_state()
        host = (self._replica_call(dict(op="cpu_residency")) or {}).get("programs", {})
        profile = dict(self.profile, beta=self.beta)
        step = float(
            self.profile.get("pipeline_service", {})
            .get("decode_step_by_seqs", {})
            .get("1", self.reference_profile["decode_step_by_seqs"]["1"])
        )
        result = {}
        for p in sorted(mature, key=self.order.__getitem__):
            request, forecast, call = (ready[p], forecasts[p], calls[p])
            standard = self._service(
                request.num_prompt_tokens, call.total_tokens, forecast["keep"]
            )
            blocks, _, _ = self.kv_cache_manager.get_computed_blocks(request)
            pages = len(blocks.blocks[0])
            restored = request.request_id in self.finished_recving_kv_req_ids
            actual, restore = current_service(
                prompt=request.num_prompt_tokens,
                output=max(1.0, call.remaining_tokens),
                computed=request.num_computed_tokens,
                gpu_pages=pages,
                host=host.get(p, {}),
                profile=profile,
                page=self.block_size,
                step=step,
                restored=restored,
            )
            post = forecast["post"]
            saved = self._compiled_progress.get(p)
            if (
                saved is not None
                and saved[0] == request.request_id
                and (saved[1] is post)
                and (saved[2] == standard)
            ):
                quote = dict(saved[3])
                quote.update(actual_s=actual, score=actual / quote["progress"])
                stats["index_hits"] += 1
            else:
                quote = service_rank(
                    standard, actual, post["atoms"], post["mass"], "distribution"
                )
                self._compiled_progress[p] = (
                    request.request_id,
                    post,
                    standard,
                    dict(quote),
                )
                stats["index_misses"] += 1
            quote.update(
                request=request.request_id,
                current_s=standard,
                restore_s=restore,
                prompt_tokens=request.num_prompt_tokens,
                remaining_output=max(1.0, call.remaining_tokens),
                generated_output=len(request.output_token_ids),
                output_mean=call.total_tokens,
                output_cap=call.cap_tokens,
                output_exact=call.exact,
                output_source=call.source,
                output_atoms=call.atoms,
                output_mass=call.mass,
                computed_tokens=request.num_computed_tokens,
                gpu_hit_pages=pages,
                pin_present=p in self.pins,
                host=dict(host.get(p, {})),
                page=self.block_size,
                decode_step_s=step,
                restored=restored,
                standard_keep=forecast["keep"],
                rank_source=forecast["source"],
                predictor_used=forecast["source"] == "honest_cdf157",
                atoms_s=post["atoms"],
                mass=post["mass"],
            )
            result[p] = quote
        return result

    def _service_cache_state(self):
        if not hasattr(self, "_compiled_stats"):
            self._compiled_calls = {}
            self._compiled_service = OrderedDict()
            self._compiled_progress = {}
            self._compiled_means = {}
            self._compiled_stats = dict(
                call_hits=0,
                call_misses=0,
                service_hits=0,
                service_misses=0,
                index_hits=0,
                index_misses=0,
                mean_hits=0,
                mean_misses=0,
            )
        return self._compiled_stats
