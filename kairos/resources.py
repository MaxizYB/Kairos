import hashlib
import json
import os
from pathlib import Path
import time
from kairos._runtime.compat.online_path import known_request_output
from kairos.forest import SuffixForest


class PredictorResources:
    def _suffix_prepare_resources(self, phase='first-request'):
        """Load static inference resources, without observing a request.

        A production service may call this before announcing readiness. The
        per-experiment causal histories are still initialized by the switch;
        no workload prefix, output or evaluation outcome is used here.
        """
        if getattr(self, 'suffix_resources', None) is not None:
            return self.suffix_resources
        started = time.perf_counter()
        raw = Path(os.environ['CONTINUUM_R82_PROFILE']).read_bytes()
        reference = json.loads(raw)
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(os.environ['CONTINUUM_MODEL'],
                                                 local_files_only=True)
        forest = SuffixForest(os.environ['CONTINUUM_R82_FOREST'], tokenizer)
        self.suffix_resources = (reference, forest, hashlib.sha256(raw).hexdigest())
        self.suffix_resource_phase = phase
        self._emit('suffix_resources_ready', phase=phase,
            seconds=time.perf_counter()-started, forest_sha256=forest.sha256,
            profile_sha256=self.suffix_resources[2], request_observations=0,
            future_information=False)
        return self.suffix_resources

    def _suffix_service(self, prompt, output, keep):
        from kairos.causal import service
        return service(prompt, output, keep, self.suffix_reference)

    def _suffix_observation(self, request):
        """Freeze only information visible at this arrival, never a live history."""
        p = self.program(request)
        if p in self.suffix_forest.training_ids:
            raise ValueError('Kairos request identity overlaps frozen forest training')
        if known_request_output(request) is None:
            raise ValueError('Kairos replay requires an identical fixed-output contract')
        raw = self.metadata(request).get('cont_visible_messages')
        # Most metadata is already serialized. Freeze list-valued callers as
        # well: a deferred consumer must not see later mutations/outcomes.
        if isinstance(raw, list):
            raw = json.dumps(raw)
        return dict(p=p, request_id=request.request_id, ids=tuple(request.prompt_token_ids),
                    raw=raw, inputs=tuple(self.online_inputs[p]),
                    outputs=tuple(self.online_outputs[p]))

