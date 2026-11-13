"""
Does DP-SGD hurt some breast cancer subtypes more than others?

Task:    classify TCGA-BRCA tumors into PAM50 subtypes (LumA, LumB, Her2, Basal)
         from RNA-seq expression, with a small MLP trained with DP-SGD (Opacus).
Sweep:   epsilon in {non-private, 8, 4, 2, 1, 0.5}, repeated over many random splits.
Output:  results.csv (every run), summary.csv (mean and 95% CI), recall_by_subtype_*.png
Mitigation: "prior_adjusted" divides predicted probabilities by a noisy class prior (see main()).

Run in Colab or locally:
    pip install opacus pandas scikit-learn matplotlib
    python dp_subtype_fairness.py --dry-run      # plumbing check on fake data, no torch needed
    python dp_subtype_fairness.py                # real run (downloads TCGA-BRCA from UCSC Xena)

Known limitations (state these in any write-up):
  * Gene selection (top-variance genes) and standardization use the training data
    but are NOT charged to the privacy budget. The reported epsilon covers only
    the SGD training loop.
  * Hyperparameters (epochs, lr, clip norm) are fixed in advance, not tuned on the
    test data, but their selection is also not charged to the budget.
  * Her2 is a small class, so per-split recall is noisy. Look at the confidence
    intervals across seeds, not at single runs.
"""
import argparse
import gzip
import os
import urllib.request

import numpy as np
import pandas as pd

XENA = "https://tcga.xenahubs.net/download/TCGA.BRCA.sampleMap/"
EXPR_URL = XENA + "HiSeqV2.gz"  # log2(RSEM + 1), genes x samples
CLIN_URLS = [XENA + "BRCA_clinicalMatrix", XENA + "BRCA_clinicalMatrix.gz"]
LABEL_COL = "PAM50Call_RNAseq"
SUBTYPES = ["LumA", "LumB", "Her2", "Basal"]  # Normal-like is too small to evaluate


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def _download(url, path):
    if not os.path.exists(path):
        print("Downloading", url)
        urllib.request.urlretrieve(url, path)
    return path


def _read_table(path):
    with open(path, "rb") as f:
        gz = f.read(2) == b"\x1f\x8b"
    return pd.read_csv(path, sep="\t", index_col=0, compression="gzip" if gz else None, low_memory=False)


def load_tcga(cache="data"):
    os.makedirs(cache, exist_ok=True)
    expr = _read_table(_download(EXPR_URL, f"{cache}/HiSeqV2.gz")).T  # samples x genes
    clin = None
    for url in CLIN_URLS:
        try:
            clin = _read_table(_download(url, f"{cache}/clin_" + os.path.basename(url)))
            break
        except Exception as e:  # try the next URL variant
            print("Could not load", url, "->", e)
    if clin is None:
        raise RuntimeError("Could not download clinical data. Download BRCA_clinicalMatrix from "
                           "UCSC Xena by hand and place it in the data/ folder.")
    # Primary tumors only (sample IDs ending in -01) that have a PAM50 call.
    ids = [s for s in expr.index if s.endswith("-01") and s in clin.index]
    expr, clin = expr.loc[ids], clin.loc[ids]
    keep = clin[LABEL_COL].isin(SUBTYPES)
    expr, clin = expr[keep.values], clin[keep]
    y = clin[LABEL_COL].map({s: i for i, s in enumerate(SUBTYPES)}).values.astype(int)
    expr = expr.dropna(axis=1)
    print("Samples per subtype:", dict(zip(SUBTYPES, np.bincount(y, minlength=len(SUBTYPES)))))
    return expr.values.astype(np.float32), y


def synthetic_data(n=700, d=600, seed=0):
    """Fake data with TCGA-like class imbalance, only for checking the plumbing."""
    rng = np.random.default_rng(seed)
    props = np.array([0.5, 0.2, 0.08, 0.22])
    y = rng.choice(len(SUBTYPES), size=n, p=props)
    centers = rng.normal(0, 1, size=(len(SUBTYPES), d))
    X = centers[y] * 0.35 + rng.normal(0, 1, size=(n, d))
    return X.astype(np.float32), y


def prep_features(X_train, X_test, k):
    """Top-k variance genes chosen on the TRAINING split only, then standardized."""
    top = np.argsort(X_train.var(axis=0))[::-1][:k]
    A, B = X_train[:, top], X_test[:, top]
    mu, sd = A.mean(axis=0), A.std(axis=0) + 1e-6
    return (A - mu) / sd, (B - mu) / sd


# ----------------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------------
def train_torch(Xtr, ytr, Xte, eps, seed, clip, epochs, batch_size, lr):
    """MLP trained non-privately (eps=None) or with DP-SGD via Opacus. Returns class probabilities."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(seed)
    n_classes = len(SUBTYPES)
    model = nn.Sequential(  # no BatchNorm: it is not compatible with per-sample gradients
        nn.Linear(Xtr.shape[1], 128), nn.ReLU(),
        nn.Linear(128, 64), nn.ReLU(),
        nn.Linear(64, n_classes),
    )
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ds = TensorDataset(torch.tensor(Xtr), torch.tensor(ytr, dtype=torch.long))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
    if eps is not None:
        from opacus import PrivacyEngine
        pe = PrivacyEngine()
        model, opt, dl = pe.make_private_with_epsilon(
            module=model, optimizer=opt, data_loader=dl, epochs=epochs,
            target_epsilon=eps, target_delta=1.0 / len(ytr), max_grad_norm=clip)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        for xb, yb in dl:
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        return torch.softmax(model(torch.tensor(Xte)), dim=1).numpy()


def train_dry_run(Xtr, ytr, Xte, eps, seed, clip, epochs, batch_size, lr):
    """Stand-in model for --dry-run. It ignores eps; it only tests the pipeline."""
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(max_iter=300, C=0.05).fit(Xtr, ytr).predict_proba(Xte)


# ----------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------
def evaluate(yte, proba):
    from sklearn.metrics import roc_auc_score
    pred = proba.argmax(axis=1)
    rows = []
    for k, name in enumerate(SUBTYPES):
        m = yte == k
        auc = roc_auc_score((yte == k).astype(int), proba[:, k])
        rows.append(dict(group=name, n_test=int(m.sum()), recall=float((pred[m] == k).mean()), auroc=float(auc)))
    rec = [r["recall"] for r in rows]
    rows.append(dict(group="WORST_GROUP_RECALL", n_test=len(yte), recall=min(rec), auroc=np.nan))
    rows.append(dict(group="RECALL_GAP", n_test=len(yte), recall=max(rec) - min(rec), auroc=np.nan))
    rows.append(dict(group="ALL_ACCURACY", n_test=len(yte), recall=float((pred == yte).mean()),
                     auroc=float(np.mean([r["auroc"] for r in rows[:len(SUBTYPES)]]))))
    return rows


def summarize(res):
    def ci(x):
        return 1.96 * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan
    g = res.groupby(["method", "eps", "clip", "group"])
    out = g.agg(recall_mean=("recall", "mean"), recall_ci95=("recall", ci),
                auroc_mean=("auroc", "mean"), n_runs=("recall", "size"),
                n_test_mean=("n_test", "mean")).reset_index()
    return out


def plot(summary, path, clip, method):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    s = summary[(summary["clip"] == clip) & (summary["method"] == method)].copy()
    eps_vals = sorted(s["eps"].unique(), key=lambda e: (np.isinf(e), e))  # non-private last
    xs = {e: i for i, e in enumerate(eps_vals)}
    labels = ["non-private" if np.isinf(e) else f"eps={e:g}" for e in eps_vals]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name in SUBTYPES + ["RECALL_GAP"]:
        d = s[s["group"] == name].sort_values("eps", key=lambda c: c.map(xs))
        ax.errorbar([xs[e] for e in d["eps"]], d["recall_mean"], yerr=d["recall_ci95"],
                    marker="o", capsize=3, linestyle="--" if name == "RECALL_GAP" else "-",
                    color="black" if name == "RECALL_GAP" else None, label=name)
    ax.set_xticks(range(len(eps_vals)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Test recall (mean, 95% CI across splits)")
    ax.set_xlabel("Privacy budget (smaller = stronger privacy)")
    ax.set_title(f"DP-SGD: recall by PAM50 subtype (clip {clip:g}, {method})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="synthetic data + sklearn model, no torch")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--eps", type=float, nargs="+", default=[8, 4, 2, 1, 0.5])
    ap.add_argument("--clips", type=float, nargs="+", default=[1.0])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--k-genes", type=int, default=500)
    ap.add_argument("--prior-eps", type=float, default=0.1,
                    help="epsilon spent on a noisy class histogram for the prior correction")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    from sklearn.model_selection import train_test_split
    X, y = synthetic_data() if a.dry_run else load_tcga()
    trainer = train_dry_run if a.dry_run else train_torch
    os.makedirs(a.out, exist_ok=True)

    rows = []
    for seed in range(a.seeds):
        tr, te = train_test_split(np.arange(len(y)), test_size=0.25, stratify=y, random_state=seed)
        Xtr, Xte = prep_features(X[tr], X[te], a.k_genes)
        # Class prior for the post-hoc correction. The class counts come from private
        # training labels, so they are released through a Laplace mechanism
        # (sensitivity 1, scale 1/prior_eps). Total privacy cost = eps + prior_eps.
        rng = np.random.default_rng(1000 + seed)
        counts = np.bincount(y[tr], minlength=len(SUBTYPES)).astype(float)
        noisy = np.maximum(counts + rng.laplace(0, 1.0 / a.prior_eps, size=len(counts)), 1.0)
        prior = noisy / noisy.sum()
        for eps in [None] + list(a.eps):  # None = non-private baseline
            for clip in a.clips:
                proba = trainer(Xtr, y[tr], Xte, eps, seed, clip, a.epochs, a.batch_size, a.lr)
                for method, p in [("plain", proba), ("prior_adjusted", proba / prior)]:
                    for r in evaluate(y[te], p):
                        rows.append(dict(seed=seed, eps=np.inf if eps is None else eps,
                                         clip=clip, method=method, **r))
        print(f"finished split {seed + 1}/{a.seeds}")

    res = pd.DataFrame(rows)
    res.to_csv(f"{a.out}/results.csv", index=False)
    summ = summarize(res)
    summ.to_csv(f"{a.out}/summary.csv", index=False)
    for clip in a.clips:
        for method in ["plain", "prior_adjusted"]:
            plot(summ, f"{a.out}/recall_by_subtype_clip{clip:g}_{method}.png", clip, method)
    show = summ[summ["group"].isin(SUBTYPES + ["WORST_GROUP_RECALL", "RECALL_GAP", "ALL_ACCURACY"])]
    print(show.round(3).to_string(index=False))
    print(f"\nSaved results to {a.out}/")


if __name__ == "__main__":
    main()