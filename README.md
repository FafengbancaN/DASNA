# DASNA

**Diffusion Augmentation with Selective Neighborhood Aggregation for Graph Anomaly Detection**

[中文说明](README.zh-CN.md)

DASNA studies supervised node anomaly detection with imbalanced labels. Its
pipeline fits a latent diffusion generator on the training subgraph, generates
synthetic anomalies offline, filters neighborhood candidates, and aggregates
relation-specific neighbor sequences with Mamba.

![DASNA framework](docs/figures/framework.png)

## Environments

```bash
torch-geometric==2.7.0
pygod==1.1.0
tqdm==4.67.3
torch==2.8.0
numpy==2.2.6
scipy==1.15.3
scikit-learn==1.7.2
PyYAML==6.0.3
einops==0.8.2
transformers==4.36.2
mamba ssm==2.2.3 
causal-conv1d==1.4.0
```

## Data and Training

```bash
conda activate dasna-detection
python -m dasna.preprocessing --datasets amazon

conda activate dasna-augmentation
python -m dasna.augmentation.train --dataset amazon --ratio 10 --device -1 --output-dir data/generated/train-ratio

conda activate dasna-detection
python -m dasna.detection.train --config configs/amazon.yaml --generated-dir data/generated/train-ratio
```
