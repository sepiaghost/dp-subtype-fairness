# Does differential privacy hurt rare breast cancer subtypes?

A small empirical study of how DP-SGD affects PAM50 subtype classification on TCGA-BRCA gene expression, and whether a cheap post-hoc fix helps.

**Short answer:** in this setup, strong privacy mostly hurts the rare subtypes while overall accuracy hides it. At epsilon = 8, overall accuracy fell from 0.85 to 0.79, but HER2 recall fell from 0.84 to 0.28. A post-hoc class-prior correction recovered much of the lost HER2 recall at epsilon = 2 to 8, but it failed at epsilon <= 1 and did little for LumB.

## Question

When a model is trained with differential privacy, do some patient subgroups lose more accuracy than others, and can a simple fix that costs almost no extra privacy close the gap?

This follows the disparate-impact result for DP-SGD (Bagdasaryan et al., NeurIPS 2019) and tests it on genomic data.

## Data

- **Source:** TCGA-BRCA from the UCSC Xena TCGA hub: RNA-seq expression (`HiSeqV2`, log2(RSEM + 1)) and the clinical matrix.
- **Labels:** `PAM50Call_RNAseq`. Only primary tumors with a call of LumA, LumB, Her2 or Basal were used (Normal-like is too small to evaluate).
- **Size:** 821 tumors: LumA 421, LumB 192, Basal 141, Her2 67.

## Method

- **Model:** MLP (500 -> 128 -> 64 -> 4), Adam, 30 epochs, batch size 64, learning rate 1e-3, no BatchNorm.
- **Features:** the 500 highest-variance genes, chosen on the training split only, then standardized.
- **Privacy:** DP-SGD with Opacus, target epsilon in {8, 4, 2, 1, 0.5}, delta = 1 / (training set size), clip norm 1.0 (sweep over 0.5, 1 and 2 below). A non-private baseline uses the same architecture.
- **Evaluation:** 10 random stratified 75/25 splits. Metrics are per-subtype recall, one-vs-rest AUROC, overall accuracy, and the recall gap (best subtype minus worst).
- **Mitigation ("prior_adjusted"):** divide predicted class probabilities by a class prior before taking the argmax. The prior comes from the training label counts released through a Laplace mechanism (sensitivity 1, epsilon = 0.1), so the total privacy cost of an adjusted result is the training epsilon plus 0.1.

## Results

### 1. DP-SGD hurts the rare subtypes first

Mean over 10 splits (clip norm 1.0).

| epsilon | HER2 recall | LumB recall | LumA recall | Basal recall | Overall accuracy | Recall gap |
|---|---|---|---|---|---|---|
| non-private | 0.84 | 0.70 | 0.88 | 0.97 | 0.85 | 0.27 |
| 8 | 0.28 | 0.51 | 0.94 | 0.99 | 0.79 | 0.71 |
| 4 | 0.06 | 0.26 | 0.99 | 0.99 | 0.74 | 0.94 |
| 2 | 0.00 | 0.03 | 1.00 | 0.99 | 0.69 | 1.00 |
| 1 | 0.00 | 0.01 | 1.00 | 0.99 | 0.68 | 1.00 |
| 0.5 | 0.00 | 0.04 | 0.99 | 0.82 | 0.66 | 0.99 |

The larger classes (LumA, Basal) keep high recall, and the smaller ones (HER2, LumB) collapse. Overall accuracy understates the damage. HER2 AUROC stayed high (0.99 non-private, about 0.96 at epsilon = 1) while its recall was 0, so the model still ranks HER2 samples sensibly but defaults to the large classes at the decision threshold.

### 2. The result is not an artifact of one clip norm

HER2 recall at epsilon = 8: 0.27 (clip 0.5), 0.28 (clip 1), 0.32 (clip 2). At epsilon = 4 it is about 0.05 to 0.06 for all three, and at epsilon <= 2 it is 0.00 for all three. Only three clip norms were tried, so this is not an exhaustive tuning study.

### 3. A prior correction helps at moderate privacy, and breaks at strict privacy

| epsilon | HER2 recall (plain -> adjusted) | LumB recall | Overall accuracy | Recall gap |
|---|---|---|---|---|
| non-private | 0.84 -> 0.88 | 0.70 -> 0.72 | 0.85 -> 0.85 | 0.27 -> 0.25 |
| 8 | 0.28 -> 0.76 | 0.51 -> 0.54 | 0.79 -> 0.81 | 0.71 -> 0.45 |
| 4 | 0.06 -> 0.66 | 0.26 -> 0.30 | 0.74 -> 0.77 | 0.94 -> 0.69 |
| 2 | 0.00 -> 0.62 | 0.03 -> 0.13 | 0.69 -> 0.73 | 1.00 -> 0.87 |
| 1 | 0.00 -> 0.95 | 0.01 -> 0.06 | 0.68 -> 0.60 | 1.00 -> 0.93 |
| 0.5 | 0.00 -> 1.00 | 0.04 -> 0.01 | 0.66 -> 0.13 | 0.99 -> 1.00 |

- At epsilon = 2 to 8 the correction recovers most of the lost HER2 recall and does not cost overall accuracy.
- At epsilon <= 1 it overcorrects. At epsilon = 0.5 overall accuracy drops to 0.13, because the model predicts HER2 for almost everything. The high HER2 recall at those levels is not a real recovery. A likely cause is that the model's outputs carry very little signal at that noise level, but I have not tested that explanation.
- LumB barely recovers, and it becomes the worst-performing subtype. LumB and LumA overlap in expression, so this looks like more than a class-frequency problem.

Prior correction is a standard idea from the long-tail classification literature. This project applies it under DP and measures where it holds. It is not a new method.

## Limitations

- **Privacy accounting is partial.** Gene selection and standardization use the training data but are not charged to the privacy budget. The reported epsilon covers only the training loop (plus 0.1 for the noisy class prior in adjusted results). Hyperparameters were fixed in advance and not tuned on test data, but their choice is also not charged.
- **Overlapping splits.** The 10 splits are random resamples of the same 821 samples, so confidence intervals reflect variability across splits and understate the true uncertainty.
- **Small classes.** A test split holds about 17 HER2 samples, so per-split HER2 recall moves in steps of about 6 points.
- **Label circularity.** The PAM50 calls were derived from RNA-seq expression, so predicting them from expression is partly circular. This is a study of how privacy affects a classifier, not a clinical result.
- **One dataset, one architecture, one task.** Nothing here shows the pattern generalizes.

## Reproduce

```
pip install opacus pandas scikit-learn matplotlib
python dp_subtype_fairness.py --dry-run                    # pipeline check on fake data
python -W ignore dp_subtype_fairness.py                    # main experiment
python -W ignore dp_subtype_fairness.py --clips 0.5 1 2 --out clip_sweep
python -W ignore dp_subtype_fairness.py --out mitigation   # includes the prior correction
```

Each run writes `results.csv` (every run), `summary.csv` (means and 95% intervals across splits) and recall-by-subtype figures. The data download is automatic and cached in `data/`.

## Files

- `dp_subtype_fairness.py` is the full experiment.
- `results/`, `clip_sweep/` and `mitigation/` hold the outputs behind the tables above.

## Reference

Bagdasaryan, E., Poursaeed, O., Shmatikov, V. (2019). Differential Privacy Has Disparate Impact on Model Accuracy. NeurIPS.
