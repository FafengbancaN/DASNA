# DASNA

**Diffusion Augmentation with Selective Neighborhood Aggregation for Graph Anomaly Detection**

The text below is transcribed from the author-supplied abstract. Its experimental
claims have not been revalidated as part of this repository preparation.

## Abstract

Supervised anomaly detection on Web graphs learns from scarce positive labels
while aggregating neighborhoods that mix classes. Generative augmentation can
supply minority-class examples, but a generated node reaches a supervised update
only through the neighborhood that represents it. Designing augmentation and
aggregation separately leaves that interface unspecified. We present DASNA
(Diffusion Augmentation with Selective Neighborhood Aggregation), a framework for
imbalanced supervised node-level anomaly detection that specifies this interface.
A conditional latent diffusion generator, fitted to the training subgraph alone,
synthesizes anomalies offline; each synthetic positive then takes two distinct
roles, as a supervised training center and as a supplementary neighbor of training
positives. Every center’s neighborhood is organized in two stages, candidate
construction followed by feature-residual ranking that retains a fixed fraction,
before multi-branch selective state space aggregation encodes it. Supplementation
is disabled during evaluation and no synthetic edge reaches a validation or test
node. Across six datasets and ten baselines on one fixed split per dataset, DASNA
attains the highest five-run mean in 15 of 18 dataset–metric combinations (average
rank 1.28 among eleven methods); AUROC margins reach 2.50 points on Questions,
whereas YelpChi and Elliptic show metric-dependent trade-offs. A 2×2 study of the
two selection stages gives the highest G-Mean with both stages enabled on all
three datasets evaluated. The evidence is configuration-level: baseline rankings
are descriptive, the dual role of synthetic positives is not isolated, and
threshold-dependent metrics use a fixed threshold of 0.5.

The original Word document is retained locally. This Markdown transcription is
the shared abstract, so personal editing metadata from the source document is
not included in the repository.
