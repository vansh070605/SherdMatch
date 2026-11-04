# AI-Based Archaeological Pottery Fragment Matching

**21CSE451T Pattern Recognition Techniques, case study / mini project (Review 2)**
Vansh Agrawal (RA2311026010120), CINTEL, Sec AC-1. Faculty: Dr Nithyashri J

Archaeologists recover vessels as hundreds of broken sherds. Given one sherd, this project ranks the
other sherds in a collection by how likely they are to **physically join** it. Then it previews the best fit as a
first step toward digital reconstruction. It *recommends* candidates for an expert to check. It does not
claim a guaranteed reconstruction.

The research question from Review 1: **does adding fracture-edge geometry to image similarity improve top-k
matching over image-only retrieval?**

## Results (gold standard, 124 sherds, 111 expert refit pairs, 5-fold vessel-grouped CV)

| Method | Hit@1 | Hit@5 | Hit@10 | MRR |
|---|---|---|---|---|
| Random ranking | 0.008 | 0.040 | 0.105 | 0.056 |
| Colour histogram (chi²) | 0.137 | 0.403 | 0.621 | 0.271 |
| ResNet-50 embedding (cosine) | 0.153 | 0.419 | 0.565 | 0.276 |
| DINOv2 embedding (cosine), *Review 1 image-only baseline* | 0.194 | 0.403 | 0.540 | 0.314 |
| Geometry features only (LR) | 0.024 | 0.105 | 0.250 | 0.095 |
| Appearance features (LR) | **0.258** | 0.524 | 0.677 | **0.392** |
| **Proposed: appearance + fracture-edge geometry (LR)** | 0.194 | **0.556** | **0.718** | 0.364 |

Main findings (95% paired bootstrap CIs are in `outputs/tables/bootstrap_ci.csv`):
* The proposed model beats the image-only baseline at Hit@5 (+15.3 pts, CI [5.6, 25.0]) and Hit@10
  (+17.7 pts, CI [9.7, 26.6]).
* Most of that gain already comes from learning to combine several appearance cues. Adding geometry gives a
  further +3–4 pts at Hit@5/10, which is **not statistically significant**, and it lowers Hit@1.
* The same edge matcher recovers the true pose for **47%** of clean synthetic breaks, but only **17%** of real
  pairs. Eroded and chipped fracture edges are the bottleneck, not the matching algorithm.

## Repository layout

```
src/pfm/            the library
  preprocess.py     validate -> mask (alpha / Otsu+GrabCut) -> CLAHE -> contour
  features.py       colour histogram, LBP texture, ResNet-50 / DINOv2 embeddings
  geometry.py       curvature-signature edge matching, Kabsch + ICP alignment, contact/overlap check
  metrics.py        Hit@k, MRR, mAP, CMC, precision/recall/F1
  matcher.py        query-time ranking + reconstruction preview (used by the demo)
scripts/
  fetch_real_subset.py   pulls the metadata + refit photos out of the 4.7 GB zip with HTTP range requests
  explore_dataset.py     dataset + metadata exploration figures
  run_experiments.py     all experiments (E1-E5) -> outputs/results.json
  make_figures.py        result figures + bootstrap CIs
app.py              Gradio demo
outputs/            figures, tables, results.json (data/ and caches are git-ignored)
docs/               Review 2 report (.docx) and deck (.pptx)
```

## Reproduce

Dataset: **Daxinzhuang-18K**, Zenodo record [21553799](https://zenodo.org/records/21553799). Download
`dataset_gold_standard.zip`, `sim_data_8.zip` and `expert_confirmed_physical_refit_pairs.xlsx` into `data/raw/`
and unzip them there.

```bash
pip install -r requirements.txt
python scripts/explore_dataset.py
python scripts/run_experiments.py
python scripts/make_figures.py
python app.py            # http://127.0.0.1:7860
```

Model weights (timm: `resnet50.a1_in1k`, `vit_small_patch14_dinov2.lvd142m`) download to `models/` on the first run.
