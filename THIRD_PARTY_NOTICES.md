# Third-party source notices

This project includes modified third-party source. Renaming packages and replacing
the original project descriptions does not remove the applicable source notices
or license obligations. This file does not grant a license to the project as a
whole. No project-wide Apache-2.0 or MIT license is asserted.

The supplied implementation includes inherited augmentation and detection
components with local modifications. Exact revisions of all inherited portions
remain to be established. The table below records component licensing scope
and preserves the source notices present in the supplied code.

| Upstream | Local scope | License evidence and treatment |
| --- | --- | --- |
| Inherited augmentation components | `dasna/augmentation/`; adapted generation/training code | The supplied augmentation implementation included Apache-2.0 license text, retained in [LICENSES/Apache-2.0.txt](LICENSES/Apache-2.0.txt) for the corresponding inherited portions. Preserve applicable source attribution and modification notices. This does not assign a license to every repository file. |
| Inherited detection components | `dasna/detection/` and its training/configuration workflow | Redistribution permission for all inherited portions has not been confirmed. Establish that scope before external publication; another component's license does not cover these portions automatically. |
| [CARE-GNN](https://github.com/YingtongDou/CARE-GNN) | Origin notices in detection `model.py` and `layers.py` | Upstream publishes [Apache-2.0](https://github.com/YingtongDou/CARE-GNN/blob/master/LICENSE). Retain applicable notices and the included Apache license text. This does not establish a license for all inherited detection code. |
| [graphsage-simple](https://github.com/williamleif/graphsage-simple) by William L. Hamilton | Origin notices in detection `graphsage.py`, `layers.py` and `utils.py` | No root license file or explicit reuse license was found; GitHub API reported `license: null`. Retain source attribution and establish redistribution permission before external publication. |
| [TabSyn](https://github.com/amazon-science/tabsyn) | `dasna/augmentation/diffusion.py`, identified by the supplied source header | Upstream publishes [Apache-2.0](https://github.com/amazon-science/tabsyn/blob/main/LICENSE) and a [NOTICE](https://github.com/amazon-science/tabsyn/blob/main/NOTICE). Retain [LICENSES/TabSyn-NOTICE.txt](LICENSES/TabSyn-NOTICE.txt) along with the license text. |
| [PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric) | `dasna/augmentation/sage.py` explicitly identifies `torch_geometric.nn.conv.SAGEConv` as its source | Upstream publishes [MIT](https://github.com/pyg-team/pytorch_geometric/blob/2.3.1/LICENSE). Preserve its copyright/license notice when redistributing this file. See [LICENSES/PyG-MIT.txt](LICENSES/PyG-MIT.txt). |

The supplied source contains project-specific generation and classification
changes. The local preparation additionally reorganizes Python packages, updates
imports/paths, and replaces inherited project marketing and packaging metadata.
These modification descriptions are not claims of authorship over the upstream
implementations. Runtime dependencies, including PyTorch and Mamba, remain subject
to their own licenses; they are not bundled as third-party source in this tree.

Before an external release, establish the permissions and exact source provenance
for the portions whose license is unresolved, and confirm any additional notices
against the source revisions actually used. Attribution alone does not resolve an
absent license.
