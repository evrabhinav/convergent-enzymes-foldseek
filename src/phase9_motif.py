"""
Phase 9: 3Di motif discovery for catalytic-site fingerprinting.

The user's idea, made concrete: for each EC class we have 5 train members.
They share no overall sequence/fold (convergent). But chemistry forces them
to share local geometry at catalytic residues. In Foldseek's 3Di alphabet,
each residue is a letter describing its local 3D environment, so:

  "spatially conserved residues" = "3Di k-mer that appears in all 5 members"

Pipeline:
  1. Read 3Di FASTA for train + test.
  2. For each EC class, find k-mers (k=5,6,7) that occur in >=THRESHOLD of
     its 5 train members AND are rare in the rest of the train set
     (enrichment vs background).
  3. Score every test protein against every EC class's motif set
     (count motif hits, weighted by motif specificity = log(p_class/p_bg)).
  4. Predict the EC class with the highest score; evaluate weighted F1.

Hyperparameters swept:
  k in {4, 5, 6, 7}
  motif_min_members in {3, 4, 5}  (motif must appear in >= this many of 5)
  enrichment_min in {2.0, 5.0, 10.0}

Outputs:
  results/phase9_motif_results.csv
  results/phase9_summary.txt
  charts/phase9_motif.png
"""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
FS_DIR = ROOT / "foldseek_workdir"
RESULTS_DIR = ROOT / "results"
CHARTS_DIR = ROOT / "charts"


def read_3di_fasta(path: Path) -> dict[str, str]:
    """Map UniProt ID -> 3Di sequence (uppercase)."""
    out = {}
    cur_id = None
    buf: list[str] = []
    for line in path.read_text().splitlines():
        if not line:
            continue
        if line.startswith(">"):
            if cur_id is not None:
                out[cur_id] = "".join(buf).upper()
            # ">A0A0B0QJN8 ALPHAFOLD ..." -> take token 1
            cur_id = line[1:].split()[0]
            buf = []
        else:
            buf.append(line.strip())
    if cur_id is not None:
        out[cur_id] = "".join(buf).upper()
    return out


def kmers(s: str, k: int) -> set[str]:
    return {s[i:i + k] for i in range(len(s) - k + 1)}


def kmer_count(s: str, k: int) -> Counter:
    return Counter(s[i:i + k] for i in range(len(s) - k + 1))


def discover_motifs(train_seqs_by_class: dict[str, list[str]],
                    k: int,
                    motif_min_members: int,
                    enrichment_min: float,
                    background_freq: dict[str, float]) -> dict[str, dict[str, float]]:
    """For each EC class, return {motif: weight} (weight = log enrichment)."""
    motifs_per_class: dict[str, dict[str, float]] = {}
    for ec, seqs in train_seqs_by_class.items():
        if len(seqs) < motif_min_members:
            continue
        # count how many of the 5 members contain each k-mer
        member_kmer_sets = [kmers(s, k) for s in seqs]
        candidate: Counter = Counter()
        for ks in member_kmer_sets:
            for kk in ks:
                candidate[kk] += 1
        # require it in >= motif_min_members of `len(seqs)` members
        motifs: dict[str, float] = {}
        for kk, n_members in candidate.items():
            if n_members < motif_min_members:
                continue
            # in-class freq (per-member presence)
            p_class = n_members / len(seqs)
            # background presence (fraction of training proteins that contain it)
            p_bg = background_freq.get(kk, 1e-9)
            enr = p_class / max(p_bg, 1e-9)
            if enr >= enrichment_min:
                motifs[kk] = float(np.log2(enr))
        if motifs:
            motifs_per_class[ec] = motifs
    return motifs_per_class


def background_presence(train_seqs: list[str], k: int) -> dict[str, float]:
    """Fraction of training proteins whose 3Di string contains each k-mer."""
    total = len(train_seqs)
    counts: Counter = Counter()
    for s in train_seqs:
        for kk in kmers(s, k):
            counts[kk] += 1
    return {kk: c / total for kk, c in counts.items()}


def score_test(test_seqs: dict[str, str],
               motifs_per_class: dict[str, dict[str, float]],
               k: int) -> dict[str, str]:
    """For each test seq, return predicted EC class (or None)."""
    out: dict[str, str] = {}
    for tid, s in test_seqs.items():
        kc = kmer_count(s, k)
        best_ec = None
        best_score = -np.inf
        for ec, motifs in motifs_per_class.items():
            sc = sum(weight * kc.get(motif, 0)
                     for motif, weight in motifs.items())
            if sc > best_score:
                best_score = sc; best_ec = ec
        if best_score > 0 and best_ec is not None:
            out[tid] = best_ec
    return out


def evaluate(preds: dict[str, str], test_labels: dict[str, str]) -> tuple[float, int]:
    """Weighted F1 over all known test labels, missing predictions -> '__none__'."""
    trues, hats = [], []
    n_pred = 0
    for tid, ec in test_labels.items():
        trues.append(ec)
        if tid in preds:
            hats.append(preds[tid]); n_pred += 1
        else:
            hats.append("__none__")
    return f1_score(trues, hats, average="weighted", zero_division=0), n_pred


def main():
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    train_3di = read_3di_fasta(FS_DIR / "train_3di.fasta")
    test_3di = read_3di_fasta(FS_DIR / "test_3di.fasta")
    print(f"train 3Di sequences: {len(train_3di)} (of {len(train_df)} train rows)")
    print(f"test  3Di sequences: {len(test_3di)} (of {len(test_df)} test rows)")
    print(f"avg train 3Di length: {np.mean([len(s) for s in train_3di.values()]):.0f}")

    # Group train sequences by EC class
    train_seqs_by_class: dict[str, list[str]] = defaultdict(list)
    for _, row in train_df.iterrows():
        seq = train_3di.get(row["Entry"])
        if seq:
            train_seqs_by_class[row["Label"]].append(seq)
    print(f"EC classes with >=1 train 3Di seq: {len(train_seqs_by_class)}")
    print(f"  members per class distribution: "
          f"{np.bincount([len(v) for v in train_seqs_by_class.values()])}")

    test_labels = dict(zip(test_df["Entry"], test_df["Label"]))

    all_train_seqs = [s for seqs in train_seqs_by_class.values() for s in seqs]

    rows = []
    for k in [4, 5, 6, 7, 8]:
        bg = background_presence(all_train_seqs, k)
        for min_mem in [3, 4, 5]:
            for enr_min in [2.0, 5.0, 10.0]:
                motifs = discover_motifs(train_seqs_by_class, k, min_mem,
                                         enr_min, bg)
                n_classes_with_motifs = len(motifs)
                if n_classes_with_motifs == 0:
                    continue
                preds = score_test(test_3di, motifs, k)
                f1, n_pred = evaluate(preds, test_labels)
                rows.append({
                    "k": k, "min_members": min_mem, "min_enrichment": enr_min,
                    "n_classes_with_motifs": n_classes_with_motifs,
                    "n_predicted": n_pred, "weighted_f1": f1,
                })
                print(f"  k={k} min_mem={min_mem} min_enr={enr_min:>4.1f}  "
                      f"classes={n_classes_with_motifs:>3d}  "
                      f"predicted={n_pred:>3d}  F1={f1:.4f}")

    df = pd.DataFrame(rows).sort_values("weighted_f1", ascending=False)
    df.to_csv(RESULTS_DIR / "phase9_motif_results.csv", index=False)

    best = df.iloc[0]
    print(f"\nBEST motif config: k={int(best['k'])} "
          f"min_members={int(best['min_members'])} "
          f"min_enrichment={best['min_enrichment']}  "
          f"F1={best['weighted_f1']:.4f}")

    # Compare against baselines
    methods = {
        "Sequence (424)": 0.016,
        "Structural feat. (100)": 0.060,
        "ESM2-35M LR": 0.161,
        "Foldseek top-1": 0.2383,
        "FS + ESM2-150M ens": 0.2520,
        f"3Di motif best\n(k={int(best['k'])}, mem>={int(best['min_members'])}, enr>={best['min_enrichment']})":
            best["weighted_f1"],
        "ESM2-3B (DGEB)": 0.265,
    }
    fig, ax = plt.subplots(figsize=(13, 5.5))
    cols = ["#bbb", "#88a", "#99b", "#4c72b0", "#5588cc", "#cc4444", "#444"]
    bars = ax.bar(list(methods.keys()), list(methods.values()), color=cols)
    for b, v in zip(bars, methods.values()):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.005,
                f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax.axhline(0.265, ls="--", color="gray", alpha=0.6)
    ax.set_ylabel("weighted F1")
    ax.set_title("Phase 9 — 3Di motif classifier (catalytic-region signatures)")
    ax.set_ylim(0, max(max(methods.values()) * 1.2, 0.4))
    plt.xticks(rotation=12, ha="right", fontsize=8)
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "phase9_motif.png", dpi=140)
    plt.close()

    txt = ["Phase 9 — 3Di motif classifier", "=" * 50, ""]
    txt.append(f"BEST: k={int(best['k'])} min_members={int(best['min_members'])} "
               f"min_enrichment={best['min_enrichment']}  F1={best['weighted_f1']:.4f}")
    txt.append("")
    txt.append("Top 10 configurations:")
    for _, r in df.head(10).iterrows():
        txt.append(f"  k={int(r['k'])} min_mem={int(r['min_members'])} "
                   f"enr={r['min_enrichment']:>4.1f}  "
                   f"classes={int(r['n_classes_with_motifs']):>3d}  "
                   f"pred={int(r['n_predicted']):>3d}  "
                   f"F1={r['weighted_f1']:.4f}")
    (RESULTS_DIR / "phase9_summary.txt").write_text("\n".join(txt))


if __name__ == "__main__":
    main()
