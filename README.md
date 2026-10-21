# DASNA

**Diffusion Augmentation with Selective Neighborhood Aggregation for Graph Anomaly Detection**

[中文说明](README.zh-CN.md)

DASNA studies supervised node anomaly detection with imbalanced labels. Its
pipeline fits a latent diffusion generator on the training subgraph, generates
synthetic anomalies offline, filters neighborhood candidates, and aggregates
relation-specific neighbor sequences with Mamba.

![DASNA framework](docs/figures/framework.png)

## Environments

| Stage | Example environment name | Observed runtime |
| --- | --- | --- |
| Augmentation | `dasna-augmentation` | Python 3.10, PyTorch 2.6.0+cu124, CPU graph partitioning |
| Detection | `dasna-detection` | Python 3.10, PyTorch 2.8.0+cu128, Mamba CUDA forward pass |

```bash
conda activate dasna-augmentation
python scripts/check_environment.py --stage augmentation --device cpu

conda activate dasna-detection
python scripts/check_environment.py --stage detection --device cuda
```

## Data preparation

```bash
conda activate dasna-detection
python -m dasna.preprocessing --datasets amazon
```

## Training and generation

```bash
conda activate dasna-augmentation
python -m dasna.augmentation.train --dataset amazon --ratio 10 --device -1 --output-dir data/generated/train-ratio

conda activate dasna-detection
python -m dasna.detection.train --config configs/amazon.yaml --generated-dir data/generated/train-ratio
```

Use `python -m dasna.augmentation.generate --dataset amazon --ratio 10 --device -1
--output-dir data/generated/train-ratio` to generate from new-format checkpoints
without retraining. Each entry point supports `--help`; generated MAT files are
not overwritten unless `--overwrite` is explicitly selected.
