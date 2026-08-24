# RadiologyNET-CLIP

Retrieval of radiological images from diagnosis text, using CLIP-style contrastive learning.

Undergraduate thesis project. Croatian title: *Dohvat radioloških slika temeljem dijagnoza korištenjem CLIP neuronskih mreža*.

Given a radiological diagnosis, the model retrieves the image it corresponds to. It is built by fine-tuning [BiomedCLIP](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224) on the RadiologyNET dataset, collected during standard clinical practice at KBC Rijeka, covering five modalities (CR, CT, MR, RF, XA).

---

## Results

The task is diagnosis to image retrieval. Test split, 3046 images from 999 unique diagnoses, never used for training or model selection.

| model | Recall@1 | Recall@5 | Recall@10 |
|---|---|---|---|
| random chance | 0.03% | 0.2% | 0.3% |
| zero-shot BiomedCLIP | 8.0% | 22.2% | 33.3% |
| fine-tuned (first pass) | 10.0% | 30.0% | 44.1% |
| fine-tuned (final) | **10.8%** | **31.1%** | **45.2%** |

The final model retrieves a correct image in its top 10 out of 3046 candidates about 45% of the time, roughly 138x better than chance. Fine-tuning accounts for +11.9 percentage points over the zero-shot baseline at Recall@10, which at 5.5 standard errors is the only intervention in this project that produced a statistically solid improvement.

### Image to text, as a check on the above

CLIP's loss is symmetric, so the same training produces the reverse direction for free. It is not the task and there is no clinical use for it here, but it is a useful independent test of the central claim, which is that fine-tuning built a genuinely aligned shared embedding space rather than something that only works one way.

| model | image to text R@10 |
|---|---|
| random chance | 1.0% |
| zero-shot BiomedCLIP | 37.2% |
| fine-tuned (final) | 57.3% |

Both directions improve, and the improvement is larger and far more significant in this one (+20.0 points at 16.0 standard errors, against +11.9 at 5.5) despite it never being the target. That is the corroboration: the alignment holds symmetrically, so the gain on the actual task is not an artefact of one direction.

Full numbers, including clustering metrics and every intermediate model, live in `results/phase9_comparison.json`. **That file is the canonical source.** Earlier files (`results/phase7_metrics.json`, `results/phase8_metrics.json`) were computed before a slice-selection bug was fixed in phase 9 and their clustering figures are not reproducible as written.

---

## Repository layout

```
src/
  data/         splits, translation prep, truncation, Dataset, batch sampler
  models/       BiomedCLIP loading, freeze modes, benchmarking, weight averaging
  training/     loss, scheduler, training loop, experiment runner
  evaluation/   embeddings, retrieval metrics, clustering, inspection, comparison
notebooks/      one numbered notebook per phase, what was done and why
results/        metrics as structured JSON
checkpoints/    trained weights (not committed)
data/           dataset (not committed, patient data)
```

The split is deliberate: `src/` holds all working logic, notebooks are documentation only and contain no project logic.

---

## Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

The dataset is not in this repository. `src/` expects:

```
data/metainformation.csv     id, Modality, ExamID, DIAGNOSIS
data/diagnoses_final.csv     ExamID, DIAGNOSIS_TRUNCATED (translated, truncated)
data/splits.csv              ExamID, split
data/images/conversion_{MODALITY}/{id[:-3]}/{MODALITY}_{id}.dcm/*.png
```

---

## Running it

Train a model:

```bash
python -m src.training.train
```

Run the full tuning experiment matrix (resumable, writes `results/tuning/summary.json`):

```bash
python -m src.training.experiments
```

Evaluate a checkpoint on the test split (retrieval plus clustering):

```bash
python -m src.evaluation.full_eval finetuned checkpoints/best.pt results/metrics.json
```

Evaluate zero-shot BiomedCLIP by passing an empty checkpoint path:

```bash
python -m src.evaluation.full_eval zeroshot "" results/zeroshot.json
```

Compare tuning variants with error bars:

```bash
python -m src.evaluation.compare_variants
```

---

## Method in brief

**Fine-tuning.** Partial fine-tuning, unfreezing the last 2 transformer blocks of each encoder plus the projection heads, about 29.6M of 195.9M parameters. Full fine-tuning needs 9.87GB and does not fit on a 6GB card. bf16 mixed precision, AdamW, cosine schedule with warmup, gradient clipping.

**Evaluation.** Recall@K in both directions over an image-level pool. Ground truth is exact diagnosis text match, which handles both the several-images-per-exam case and the rare identical-text-across-exams case with one rule. Embeddings are also clustered with k-means and scored against the known modality label using ARI and NMI.

**Model selection.** Every variant is ranked on validation Recall@10 only. The test split is touched once, by the pre-declared winner. Differences smaller than two standard errors (0.018 on the combined score) are treated as ties, which turned out to matter: a weight-averaged model led on validation by 0.0026 and then lost on test by 0.0105.

---

## Notes and caveats

The dataset and any notebook whose output displays real diagnosis text or images are excluded from this repository. That covers notebooks 001 to 004 and 007, so the documentation trail here has gaps at data exploration and evaluation.

Validation loss and retrieval quality diverge on this dataset. Loss bottoms out around epoch 3 while Recall@10 keeps improving to epoch 6, so checkpoints are selected on retrieval, not loss. Selecting on loss costs roughly 1.7 points of Recall@10.

Tuning was explored fairly thoroughly, six training variants plus weight averaging and a hubness correction at inference. None beat the baseline by more than the noise bar. With translation quality ruled out separately, the evidence points at the dataset size, roughly 10,000 exams, as the binding constraint rather than the method.

Training runs do not currently set random seeds, so exact reproduction of a given run is not guaranteed. Evaluation is deterministic.
