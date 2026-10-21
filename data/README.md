# Local data

Datasets, generated samples, and adjacency caches remain on this machine. Git
tracks the directory placeholders, this guide, and the small
[inventory](inventory.json), which records the prepared MAT schemas, split
counts, file sizes, and SHA-256 hashes. It does not contain node features or edges.

## Layout

```text
data/
  raw/         Original prepared MAT files and historical feature archives
  processed/   Adjacency-list caches rebuilt by dasna.preprocessing
  generated/   Standalone synthetic training graphs
  inventory.json
```

The local prepared datasets are:

| Dataset key | MAT filename | Nodes | Features | Positive nodes |
| --- | --- | ---: | ---: | ---: |
| amazon | Amazon.mat | 11,944 | 25 | 821 |
| yelpchi | YelpChi.mat | 45,954 | 32 | 6,677 |
| weibo | weibo.mat | 8,405 | 400 | 868 |
| questions | questions.mat | 48,921 | 301 | 1,460 |
| facebook | facebook.mat | 22,470 | 4,714 | 6,495 |
| elliptic | elliptic.mat | 203,769 | 165 | 4,545 |

These counts describe the files inspected during repository preparation. They
do not establish the provenance of the labels or the paper's exact preprocessing.
Dataset redistribution permissions are separate from source-code permissions.
The prepared files and their download locations must be supplied by the authors
for a fresh checkout to reproduce the same inputs; this repository does not
invent replacement sources or silently download a different split.

## MAT schema

Use either `features` / `label` / `homo` or
`Attributes` / `Label` / `Network`. Features have shape `(N, F)`, labels flatten
to `(N,)`, and adjacency matrices have shape `(N, N)`. Sparse adjacency is
recommended. Normal nodes have label `0`, anomalies have label `1`.

Prepared original files must include disjoint Boolean `train_mask`, `val_mask`,
and `test_mask`, each containing `N` values. The local six datasets already
include these masks. Keep the same masks across augmentation and detection.
For exact experimental comparisons, preserve the supplied split rather than
creating a new one. Loaders have fallback splitting for inputs without masks;
that is not a substitute for the same supplied split across both stages. The
documented end-to-end workflow uses the supplied masks, and the generator does
not write new splits into raw files.

Amazon's relation matrices are `net_upu`, `net_usu`, `net_uvu`; YelpChi's are
`net_rur`, `net_rtr`, `net_rsr`. Preprocessing preserves those distinct matrices.
For datasets with only one graph, that graph supplies each of the three
aggregation branches.

## Preprocessing and generated samples

Run `python -m dasna.preprocessing --help` for dataset and directory options.
Rebuild adjacency caches after moving or replacing a raw graph. Historical
`yelp_*` cache files are retained locally for reference; the new loader and
preprocessor use the canonical `yelpchi_*` prefix.

The generator writes `<dataset>_<ratio>pct_standalone.mat` under `generated/`.
New samples have `features`, `label`, `homo`, and three masks; every generated
node has label `1` and belongs only to training. The detector remaps these nodes
after original node IDs and adds no synthetic edge to an original validation
or test node. Training positives can use positive training nodes as supplementary
neighbors; evaluation disables that supplementation.

New generation defines `ratio` as a percentage of **training anomalies**. The
eight historical files retained locally used a percentage of **all anomalies**.
For example, the historical Amazon `10pct` file contains 82 nodes. Do not mix
these definitions in an experiment. Regenerate with the new CLI and keep a
separate output directory when comparing with historical runs.

To refresh the local inventory after a deliberate data change:

```bash
python scripts/inspect_data.py
```

Review changes to `inventory.json` before sharing it. Large data files are
ignored even if this inventory is committed.
