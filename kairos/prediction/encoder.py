"""Server-only arrived-state encoder. No training scripts or unused heads.

Token IDs and the trained encoder are unchanged. Bounded text caches avoid
retokenizing identical arrived fields; packed pinned inputs need two copies
and one final synchronization instead of four blocking input copies.
"""

from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from transformers import AutoConfig, AutoModel, AutoTokenizer

from .joint import JointDensity


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def signature(visible):
    return hashlib.sha256(
        json.dumps(visible, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


class VisibleTokenizer:
    LIMITS = dict(issue=126, previous=94, feedback=158)

    def __init__(self, tokenizer, mean, scale):
        self.tokenizer = tokenizer
        self.mean, self.scale = np.asarray(mean), np.asarray(scale)
        self.cache = {key: OrderedDict() for key in self.LIMITS}
        self.hits = self.misses = 0

    def one(self, visible):
        parts = {}
        for key, limit in self.LIMITS.items():
            text, cache = visible[key], self.cache[key]
            try:
                ids = cache[text]
                cache.move_to_end(text)
                self.hits += 1
            except KeyError:
                ids = self.tokenizer.encode(text, add_special_tokens=False)
                ids = tuple(ids[:limit] if key == "issue" else ids[-limit:])
                cache[text] = ids
                self.misses += 1
                if len(cache) > 64:
                    cache.popitem(last=False)
            parts[key] = list(ids)
        issue, previous, feedback = (parts[k] for k in self.LIMITS)
        tok = self.tokenizer
        ids = (
            [tok.cls_token_id]
            + issue
            + [tok.sep_token_id]
            + previous
            + [tok.sep_token_id]
            + feedback
            + [tok.sep_token_id]
        )
        types = [0] * (len(issue) + 2) + [1] * (len(previous) + len(feedback) + 2)
        numeric = (np.asarray(visible["numeric"], float) - self.mean) / self.scale
        if (
            len(ids) != len(types)
            or len(ids) > 384
            or numeric.shape != (127,)
            or not np.isfinite(numeric).all()
        ):
            raise ValueError("Invalid arrived feature schema")
        return dict(
            input_ids=ids, token_type_ids=types, numeric=numeric.astype(np.float32)
        )

    def reset(self):
        for cache in self.cache.values():
            cache.clear()
        self.hits = self.misses = 0


class FeatureEncoder(nn.Module):
    def __init__(self, path):
        super().__init__()
        # The trained checkpoint contains every encoder tensor. Construct from
        # config rather than downloading/loading a second, unused base model.
        # Preserve the host RNG so initialization cannot affect serving.
        config = AutoConfig.from_pretrained(path, local_files_only=True)
        with torch.random.fork_rng(devices=[]):
            self.encoder = AutoModel.from_config(config)
        self.numeric = nn.Sequential(nn.Linear(127, 64), nn.GELU(), nn.LayerNorm(64))

    def forward(self, batch):
        with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
            hidden = self.encoder(
                **{k: v for k, v in batch.items() if k != "numeric"}
            ).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            return torch.cat(
                [(hidden * mask).sum(1) / mask.sum(1), self.numeric(batch["numeric"])],
                dim=1,
            ).float()


class ContinuousRuntime:
    def __init__(self, encoder, checkpoint, study, density_path):
        started = time.perf_counter()
        torch.set_num_threads(1)
        checkpoint, study = Path(checkpoint), Path(study)
        selection = json.loads((checkpoint / "SELECTION_LOCK.json").read_bytes())
        self.hashes = {
            key: sha(checkpoint / (key + ".pt")) for key in ("expanded", "distribution")
        }
        if (
            selection["primary"] != "distribution"
            or self.hashes != selection["checkpoints"]
        ):
            raise ValueError("Preselected encoder/checkpoint changed")
        artifact = torch.load(
            checkpoint / "expanded.pt", map_location="cpu", weights_only=True
        )
        self.training_ids = set(artifact["training_ids"])
        selected = json.loads((study / "SELECTION_LOCK.json").read_bytes())
        if sha(study / "coupled.pt") != selected["checkpoint_sha256"]:
            raise ValueError("Training provenance changed")
        metadata = torch.load(
            study / "coupled.pt", map_location="cpu", weights_only=True
        )
        if (
            metadata["encoder_sha256"] != self.hashes["expanded"]
            or set(metadata["training_ids"]) != self.training_ids
            or (
                metadata.get("marginal") and not metadata.get("public_cap_conditioning")
            )
        ):
            raise ValueError("Training identities/feature provenance changed")
        self.reference_step = metadata["reference_step"]
        self.hashes.update(
            coupled=sha(study / "coupled.pt"),
            coupled_lock=sha(study / "SELECTION_LOCK.json"),
            density=sha(density_path),
        )
        tokenizer = AutoTokenizer.from_pretrained(encoder, local_files_only=True)
        self.inputs = VisibleTokenizer(tokenizer, artifact["mean"], artifact["scale"])
        self.model = FeatureEncoder(encoder).eval()
        self.model.load_state_dict(
            {k: v for k, v in artifact["state"].items() if not k.startswith("head.")},
            strict=True,
        )
        self.model.to("cuda:0")
        with np.load(density_path) as f:
            self.density = JointDensity({k: f[k] for k in f.files})
        self.tokens_host = torch.empty((3, 384), dtype=torch.long, pin_memory=True)
        self.numeric_host = torch.empty((1, 127), dtype=torch.float32, pin_memory=True)
        self.tokens_gpu = torch.empty_like(self.tokens_host, device="cuda:0")
        self.numeric_gpu = torch.empty_like(self.numeric_host, device="cuda:0")
        self.static = dict(
            input_ids=self.tokens_gpu[0:1],
            token_type_ids=self.tokens_gpu[1:2],
            attention_mask=self.tokens_gpu[2:3],
            numeric=self.numeric_gpu,
        )
        self.token_array, self.numeric_array = (
            self.tokens_host.numpy(),
            self.numeric_host.numpy(),
        )
        self.graph = None
        with torch.inference_mode():
            dummy = self.inputs.one(
                dict(issue="", previous="", feedback="", numeric=[0.0] * 127)
            )
            self._stage(dummy)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    self.model(self.static)
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                self.graph_output = self.model(self.static)
            self.graph = graph
        self.arrivals = {}
        self.calls = 0
        self.inputs.reset()
        self.load_seconds = time.perf_counter() - started

    def _stage(self, encoded):
        n = len(encoded["input_ids"])
        self.token_array.fill(0)
        self.token_array[0].fill(self.inputs.tokenizer.pad_token_id)
        self.token_array[0, :n] = encoded["input_ids"]
        self.token_array[1, :n] = encoded["token_type_ids"]
        self.token_array[2, :n] = 1
        self.numeric_array[0] = encoded["numeric"]
        self.tokens_gpu.copy_(self.tokens_host, non_blocking=True)
        self.numeric_gpu.copy_(self.numeric_host, non_blocking=True)

    def reset(self):
        self.arrivals.clear()
        self.calls = 0
        self.inputs.reset()

    @torch.inference_mode()
    def predict(self, *, key, stage, visible):
        if key in self.training_ids:
            raise ValueError("Training/evaluation identity overlap")
        before = self.arrivals.get(key)
        if (before is None and stage != 0) or (
            before is not None and stage not in (before, before + 1)
        ):
            raise ValueError("Nonsequential arrived history")
        started = time.perf_counter()
        encoded = self.inputs.one(visible)
        tokenized = time.perf_counter()
        self._stage(encoded)
        self.graph.replay()
        feature = self.graph_output.cpu().numpy()[0]
        encoded_at = time.perf_counter()
        projected, output_mass = self.density.prepare(feature)
        ended = time.perf_counter()
        self.arrivals[key] = stage
        self.calls += 1
        return dict(
            projected=projected.tolist(),
            output_mass=output_mass.tolist(),
            history=stage + 1,
            input_sha256=signature(visible),
            input_tokens=len(encoded["input_ids"]),
            seconds=ended - started,
            phase_seconds=dict(
                tokenize=tokenized - started,
                encoding=encoded_at - tokenized,
                head=ended - encoded_at,
            ),
        )
