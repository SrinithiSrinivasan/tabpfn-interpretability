
# How TabPFN Learns: Retrieval and Feature Selection

This repository studies the internal mechanisms of **TabPFN v2** and **nanoTabPFN** during tabular in-context learning. We analyze attention heads at inference time to understand how the two attention heads work. In particular, we identify specialized datapoint-attention heads that perform implicit **class-conditional retrieval**, selectively attending to training examples that share the query example’s label, and feature-attention heads that exhibit strong feature selectivity.

We introduce two interpretability metrics:

- **Class Alignment Score (CAS)** — measures how strongly a datapoint-attention head attends to same-class examples
- **Feature Selectivity Score (FSS)** — measures how concentrated a feature-attention head is on specific columns

Using targeted head ablations across multiple sklearn datasets, we show that high-CAS and high-FSS heads are causally important for model performance. Across both architectures, our results suggest that tabular foundation models develop specialized retrieval and feature-selection circuits that support in-context learning.


## Layout

```
Repo/
├── README.md
├── requirements.txt
├── .env.example                    
├── .gitignore
├── src/
│   ├── instrumented_tabpfn.py   # TabPFN v2 capture + head-zeroing patches
│   ├── model.py                 # nanoTabPFN architecture
│   └── instrumented_model.py    # nanoTabPFN capture + head-zeroing wrapper
├── checkpoints/
│   └── nanotabpfn_weights.pt    # ~1.4 MB nanoTabPFN checkpoint (bundled)
├── notebooks/
│   ├── feature_head_scoring_multi_dataset.ipynb
│   └── nanotabpfn_feature_head_scoring.ipynb
└── figures/                     # PNGs written by the notebooks 
```

## Notebooks

### 1. `feature_head_scoring_multi_dataset.ipynb` — TabPFN v2

24 layers × 3 heads = **72 feature heads**. Datasets at full multi-class
cardinality (breast=2, wine=3, digits=10).

1. FSS heatmap (one panel per dataset).
2. Top-3 + bottom-3 attention maps per dataset.
3. Progressive FSS-ranked ablation, `N ∈ {5, 10, …, 70, 71, 72}`.
4. Ablation curves overlaid across the three datasets.
5. Permutation test: per-dataset top-35 FSS heads vs. random 35 feature
   heads sampled (without replacement) from the **non-aligned pool** —
   the 37 feature heads NOT in that dataset's top-35.

### 2. `nanotabpfn_feature_head_scoring.ipynb` — nanoTabPFN

3 layers × 4 heads = **12 feature heads**, binary backbone (`num_outputs=2`).
Wine is restricted to `y < 2` and digits is loaded with `n_class=2`.

1. FSS heatmap (one panel per dataset).
2. Top-3 + bottom-3 attention maps per dataset.
3. Single-head ablation: zero one head at a time, record ROC-AUC drop.
4. Single-head drop heatmap per dataset.

## Setup

Create virtual env

pip install -r requirements.txt
```

## Model access

The **TabPFN v2** notebook calls `TabPFNClassifier(...).fit(...)`, which
downloads the pretrained weights from Prior Labs the first time it runs.
Access requires a free token https://priorlabs.ai/tabpfn-nature>. The **nanoTabPFN** notebook needs no token —
its checkpoint is bundled in `checkpoints/`.


   ```powershell
   Copy-Item .env.example .env
   # then edit .env and set TABPFN_TOKEN=<your token>
   ```


## Running

Open either notebook in Jupyter or VS Code and **Restart kernel → Run All**.
Output PNGs in `figures/`:

| Notebook   | Output                                          | File                                       |
| ---------- | ----------------------------------------------- | ------------------------------------------ |
| TabPFN v2  | FSS heatmap                                     | `fss_heatmap_multi_dataset.png`            |
| TabPFN v2  | Top-3 / bottom-3 attention maps per dataset     | `topbot_<dataset>.png` (×3)                |
| TabPFN v2  | Progressive ablation curves                     | `fss_ablation_curves_multi_dataset.png`    |
| TabPFN v2  | Permutation test histograms                     | `fss_permutation_test_multi_dataset.png`   |
| nanoTabPFN | FSS heatmap                                     | `nano_fss_heatmap.png`                     |
| nanoTabPFN | Top-3 / bottom-3 attention maps per dataset     | `nano_topbot_<dataset>.png` (×3)           |
| nanoTabPFN | Single-head ablation drop heatmap               | `nano_single_head_ablation.png`            |

## Reproducibility

Seeds are fixed in both notebooks: `SEED = 0` for splits, model RNG, and
`torch.manual_seed`; `SEED_PERM = 0` for the TabPFN v2 permutation test.
With pinned dependency versions, the figures and printed numbers should
match exactly across runs on the same hardware. Minor numerical drift
(< 1e-4 in AUC) is possible across CPU vendors due to non-deterministic
floating-point reductions in PyTorch.


