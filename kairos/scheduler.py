"""Kairos scheduler implementation."""
from kairos._runtime.compat.online_path import known_request_output
from kairos.score import current_service, service_rank
from kairos.dispatch import ForecastDispatchScheduler


class KairosScheduler(ForecastDispatchScheduler):
    VERSION = 'kairos'
    SCORE = 'actual current service t / E[s/(s+R+)]'
    RANKING = 'mature ready slots: distributional progress per current recovery/service'

    def _forecast_quotes(self, by, eligible):
        # A single advisory read for the entire event, only when at least two
        # current-request forecasts can actually compete. No victim planning,
        # waiting prediction, layout mutation or native/grant order override.
        cpu = (self._replica_call(dict(op='cpu_residency')) or {}).get('programs', {})
        profile = dict(self.profile, beta=self.beta)
        step = float(self.profile.get('pipeline_service', {}).get(
            'decode_step_by_seqs', {}).get('1', self.suffix_reference['decode_step_by_seqs']['1']))
        details = {}
        for p in eligible:
            row, request = self.forecast_predictions[p], by[p]
            output = known_request_output(request)
            if output is None:
                raise ValueError('Missing common fixed-output contract')
            standard = self._suffix_service(request.num_prompt_tokens, output, row['keep'])
            blocks, _, _ = self.kv_cache_manager.get_computed_blocks(request)
            gpu_pages = len(blocks.blocks[0])
            restored = request.request_id in self.finished_recving_kv_req_ids
            remaining_output = max(1, output-len(request.output_token_ids))
            actual, restore = current_service(prompt=request.num_prompt_tokens,
                output=remaining_output, computed=request.num_computed_tokens,
                gpu_pages=gpu_pages, host=cpu.get(p, {}), profile=profile,
                page=self.block_size, step=step, restored=restored)
            post = row['post']
            quote = service_rank(standard, actual, post['atoms'], post['mass'], 'distribution')
            quote.update(request=request.request_id, current_s=standard,
                restore_s=restore, gpu_hit_pages=gpu_pages, pin_present=p in self.pins,
                prompt_tokens=request.num_prompt_tokens, remaining_output=remaining_output,
                computed_tokens=request.num_computed_tokens, host=dict(cpu.get(p, {})),
                page=self.block_size, decode_step_s=step, restored=restored)
            details[p] = quote
        return details
