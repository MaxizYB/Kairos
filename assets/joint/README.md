# External joint predictor weights

Git contains the serving code, tokenizer/config, selection locks and weight
hashes. The trained binaries are provisioned separately and ignored by Git:

```text
joint/
  encoder/                    tokenizer and config (included)
  checkpoint/
    SELECTION_LOCK.json        included
    expanded.pt               external trained encoder
    distribution.pt           external provenance checkpoint
  study/
    SELECTION_LOCK.json        included
    coupled.pt                external reference/provenance checkpoint
  density.npz                 external continuous conditional density
```

Use the existing CG1 training artifacts. Exact source names and SHA-256 values
are recorded in ../../provenance/joint_release.json. No model download occurs
automatically. Copy the included small files and the four existing binary files
into a deployment directory, then verify and launch:

```bash
python -m kairos.verify --joint-assets /path/to/joint
python -m kairos.run --method kairos --joint-assets /path/to/joint \
  --model /path/to/pinned/Qwen3-8B --output runs/dense
```

Without --joint-assets, the default directory is assets/joint. Placing local
weights there works, but .pt/.npz files remain excluded from Git and the code
release manifest. Missing or different assets fail explicitly before serving.
All CPU unit tests use a synthetic probability law and need no weights.
Recorded quotation checks and server encoder checks additionally need external
assets. Historical validation results remain archived.
