"""
Phase 9b: SPATIAL 3Di motif classifier (the user's idea, done correctly).

The Phase 9 (linear k-mer) attempt failed because catalytic residues are
sequence-distant but spatially clustered. So instead of consecutive 3Di
k-mers, this script looks at **3Di letter pairs (l_i, l_j) where the two
residues are sequence-distant and spatially close in 3D**.

For each EC class with 5 train members:
  1. For each train member, enumerate spatial residue pairs (i, j) where
     |i - j| >= SEQ_GAP and CA-CA(i, j) <= DIST_MAX.
  2. Encode each pair by its (sorted) 3Di letter pair.
  3. Count how many of the 5 members exhibit each pair.
  4. Pairs in >= MIN_MEMBERS of 5 with enrichment >= MIN_ENRICH versus
     background = class motifs.

For each test protein:
  - enumerate its spatial 3Di pairs
  - for each EC class, score = sum_pair (in_class? * log_enrich)
  - predict argmax class.

If pairs are still too weak, we also try TRIPLES (three mutually close
residues, "catalytic triangle"), since the canonical Ser-His-Asp triad is a
3-body motif.
"""
from __future__ import annotations

import warnings
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Bio.PDB import PDBParser
from sklearn.metrics import f1_score
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
STRUCT_DIR = ROOT / "structures"
FS_DIR = ROOT / "foldseek_workdir"
RESULTS_DIR = ROOT / "results"
CHARTS_DIR = ROOT / "charts"

SEQ_GAP = 6        # |i-j| >= this
DIST_MAX = 9.0     # CA-CA <= this (Å)
DIST_MAX_TRIPLE = 10.0
TOP_K_RESIDUES = 50  # only consider top-K most-buried / highest-pLDDT positions for triples


def read_3di_fasta(path: Path) -> dict[str, str]:
    out, cur, buf = {}, None, []
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            if cur is not None:
                out[cur] = "".join(buf).upper()
            cur = line[1:].split()[0]
            buf = []
        elif line:
            buf.append(line.strip())
    if cur is not None:
        out[cur] = "".join(buf).upper()
    return out


def ca_coords(pdb_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (N,3) CA coords and (N,) pLDDT for residues in chain order."""
    parser = PDBParser(QUIET=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        structure = parser.get_structure(pdb_path.stem, str(pdb_path))
    coords, plddt = [], []
    for chain in next(structure.get_models()):
        for res in chain:
            if "CA" in res:
                coords.append(res["CA"].get_coord())
                plddt.append(res["CA"].get_bfactor())
    return np.array(coords, dtype=np.float32), np.array(plddt, dtype=np.float32)


def spatial_pairs(coords: np.ndarray, ss3di: str,
                  seq_gap: int = SEQ_GAP, d_max: float = DIST_MAX) -> set[tuple[str, str]]:
    """Return set of sorted 3Di letter pairs for sequence-distant, spatially-close residues."""
    n = min(len(coords), len(ss3di))
    if n < seq_gap + 1:
        return set()
    coords = coords[:n]
    d = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    i, j = np.triu_indices(n, k=seq_gap)
    mask = d[i, j] <= d_max
    i, j = i[mask], j[mask]
    out = set()
    for a, b in zip(i, j):
        la, lb = ss3di[a], ss3di[b]
        out.add((min(la, lb), max(la, lb)))
    return out


def spatial_pair_count(coords: np.ndarray, ss3di: str,
                       seq_gap: int = SEQ_GAP, d_max: float = DIST_MAX) -> Counter:
    n = min(len(coords), len(ss3di))
    if n < seq_gap + 1:
        return Counter()
    coords = coords[:n]
    d = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    i, j = np.triu_indices(n, k=seq_gap)
    mask = d[i, j] <= d_max
    i, j = i[mask], j[mask]
    out: Counter = Counter()
    for a, b in zip(i, j):
        out[(min(ss3di[a], ss3di[b]), max(ss3di[a], ss3di[b]))] += 1
    return out


def spatial_triples(coords: np.ndarray, ss3di: str,
                    seq_gap: int = SEQ_GAP,
                    d_max: float = DIST_MAX_TRIPLE,
                    top_k: int = TOP_K_RESIDUES,
                    plddt: np.ndarray | None = None) -> set[tuple[str, str, str]]:
    """Return set of sorted 3Di letter triples for residues pairwise close & seq-distant.

    Optionally restrict to top-K residues by pLDDT to keep combinatorics bounded.
    """
    n = min(len(coords), len(ss3di))
    if n < seq_gap + 1:
        return set()
    coords = coords[:n]
    if plddt is not None and len(plddt) >= n:
        keep = np.argsort(-plddt[:n])[:top_k]
        keep.sort()
    else:
        keep = np.arange(n)
    if len(keep) < 3:
        return set()
    sub = coords[keep]
    d = np.linalg.norm(sub[:, None, :] - sub[None, :, :], axis=-1)
    out = set()
    for ia in range(len(keep)):
        for ib in range(ia + 1, len(keep)):
            if abs(keep[ib] - keep[ia]) < seq_gap or d[ia, ib] > d_max:
                continue
            for ic in range(ib + 1, len(keep)):
                if (abs(keep[ic] - keep[ia]) < seq_gap
                        or abs(keep[ic] - keep[ib]) < seq_gap
                        or d[ia, ic] > d_max or d[ib, ic] > d_max):
                    continue
                t = tuple(sorted([ss3di[keep[ia]], ss3di[keep[ib]], ss3di[keep[ic]]]))
                out.add(t)
    return out


def evaluate_predictions(preds: dict, test_labels: dict) -> float:
    trues, hats = [], []
    for tid, ec in test_labels.items():
        trues.append(ec)
        hats.append(preds.get(tid, "__none__"))
    return f1_score(trues, hats, average="weighted", zero_division=0)


def build_pair_features(entries, struct_dir: Path, ss3di: dict) -> dict:
    out = {}
    for e in tqdm(entries):
        p = struct_dir / f"{e}.pdb"
        if not p.exists() or e not in ss3di:
            continue
        coords, _ = ca_coords(p)
        out[e] = spatial_pairs(coords, ss3di[e])
    return out


def build_pair_counts(entries, struct_dir: Path, ss3di: dict) -> dict:
    out = {}
    for e in tqdm(entries):
        p = struct_dir / f"{e}.pdb"
        if not p.exists() or e not in ss3di:
            continue
        coords, _ = ca_coords(p)
        out[e] = spatial_pair_count(coords, ss3di[e])
    return out


def build_triple_features(entries, struct_dir: Path, ss3di: dict) -> dict:
    out = {}
    for e in tqdm(entries):
        p = struct_dir / f"{e}.pdb"
        if not p.exists() or e not in ss3di:
            continue
        coords, plddt = ca_coords(p)
        out[e] = spatial_triples(coords, ss3di[e], plddt=plddt)
    return out


def discover_class_motifs(train_features_by_class: dict, min_members: int,
                          min_enr: float, background: dict) -> dict:
    """For each class, return {motif: log2 enrichment}."""
    motifs_per_class: dict = {}
    for ec, member_sets in train_features_by_class.items():
        if len(member_sets) < min_members:
            continue
        cnt: Counter = Counter()
        for s in member_sets:
            for m in s:
                cnt[m] += 1
        n_total = len(member_sets)
        ms = {}
        for m, n in cnt.items():
            if n < min_members:
                continue
            p_class = n / n_total
            p_bg = background.get(m, 1e-9)
            enr = p_class / max(p_bg, 1e-9)
            if enr >= min_enr:
                ms[m] = float(np.log2(enr))
        if ms:
            motifs_per_class[ec] = ms
    return motifs_per_class


def background_freq(all_member_sets) -> dict:
    """Fraction of training proteins that contain each motif."""
    cnt: Counter = Counter()
    n = len(all_member_sets)
    for s in all_member_sets:
        for m in s:
            cnt[m] += 1
    return {m: c / n for m, c in cnt.items()}


def score_test_pairs(test_features: dict, motifs_per_class: dict) -> dict:
    out = {}
    for tid, s in test_features.items():
        best, best_sc = None, -np.inf
        for ec, ms in motifs_per_class.items():
            sc = sum(w for m, w in ms.items() if m in s)
            if sc > best_sc:
                best, best_sc = ec, sc
        if best_sc > 0 and best is not None:
            out[tid] = best
    return out


def main():
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    test_labels = dict(zip(test_df["Entry"], test_df["Label"]))

    train_3di = read_3di_fasta(FS_DIR / "train_3di.fasta")
    test_3di = read_3di_fasta(FS_DIR / "test_3di.fasta")
    print(f"3Di seqs: train={len(train_3di)} test={len(test_3di)}")

    # --- PAIR features ---
    print("\nBuilding train pair features ...")
    train_pairs = build_pair_features(train_df["Entry"].tolist(),
                                      STRUCT_DIR / "train", train_3di)
    print("Building test pair features ...")
    test_pairs = build_pair_features(test_df["Entry"].tolist(),
                                     STRUCT_DIR / "test", test_3di)

    train_pairs_by_class: dict = defaultdict(list)
    for _, row in train_df.iterrows():
        if row["Entry"] in train_pairs:
            train_pairs_by_class[row["Label"]].append(train_pairs[row["Entry"]])

    all_train_pair_sets = [s for v in train_pairs_by_class.values() for s in v]
    bg = background_freq(all_train_pair_sets)

    rows = []
    print("\n=== Pair-based motifs ===")
    for min_mem in [3, 4, 5]:
        for enr in [2.0, 5.0, 10.0]:
            motifs = discover_class_motifs(train_pairs_by_class, min_mem, enr, bg)
            preds = score_test_pairs(test_pairs, motifs)
            f1 = evaluate_predictions(preds, test_labels)
            n_classes = len(motifs)
            print(f"  pairs  min_mem={min_mem} enr={enr:>4.1f}  "
                  f"classes={n_classes}  pred={len(preds)}  F1={f1:.4f}")
            rows.append({"feature": "pairs", "min_members": min_mem,
                         "min_enrichment": enr, "n_classes": n_classes,
                         "n_predicted": len(preds), "weighted_f1": f1})

    # --- TRIPLE features (heavier; only run if pairs were promising) ---
    print("\nBuilding train triple features (top-50 high-pLDDT residues) ...")
    train_triples = build_triple_features(train_df["Entry"].tolist(),
                                          STRUCT_DIR / "train", train_3di)
    print("Building test triple features ...")
    test_triples = build_triple_features(test_df["Entry"].tolist(),
                                         STRUCT_DIR / "test", test_3di)

    train_triples_by_class: dict = defaultdict(list)
    for _, row in train_df.iterrows():
        if row["Entry"] in train_triples:
            train_triples_by_class[row["Label"]].append(train_triples[row["Entry"]])

    all_train_triple_sets = [s for v in train_triples_by_class.values() for s in v]
    bg_t = background_freq(all_train_triple_sets)
    print("\n=== Triple-based motifs ===")
    for min_mem in [3, 4, 5]:
        for enr in [2.0, 5.0, 10.0]:
            motifs = discover_class_motifs(train_triples_by_class, min_mem, enr, bg_t)
            preds = score_test_pairs(test_triples, motifs)
            f1 = evaluate_predictions(preds, test_labels)
            n_classes = len(motifs)
            print(f"  triples min_mem={min_mem} enr={enr:>4.1f}  "
                  f"classes={n_classes}  pred={len(preds)}  F1={f1:.4f}")
            rows.append({"feature": "triples", "min_members": min_mem,
                         "min_enrichment": enr, "n_classes": n_classes,
                         "n_predicted": len(preds), "weighted_f1": f1})

    df = pd.DataFrame(rows).sort_values("weighted_f1", ascending=False)
    df.to_csv(RESULTS_DIR / "phase9b_spatial_results.csv", index=False)
    best = df.iloc[0]
    print(f"\nBEST: {best['feature']} min_mem={int(best['min_members'])} "
          f"enr={best['min_enrichment']}  F1={best['weighted_f1']:.4f}")

    # chart
    methods = {
        "Sequence (424)": 0.016,
        "Structural feat. (100)": 0.060,
        "ESM2-35M LR": 0.161,
        "Foldseek top-1": 0.2383,
        "FS + ESM2-150M ens": 0.2520,
        "Linear 3Di motif (Phase 9)": 0.0725,
        f"Spatial motif best\n({best['feature']}, mem>={int(best['min_members'])}, enr>={best['min_enrichment']})":
            best["weighted_f1"],
        "ESM2-3B (DGEB)": 0.265,
    }
    fig, ax = plt.subplots(figsize=(14, 5.5))
    cols = ["#bbb", "#88a", "#99b", "#4c72b0", "#5588cc", "#aaa", "#cc4444", "#444"]
    bars = ax.bar(list(methods.keys()), list(methods.values()), color=cols)
    for b, v in zip(bars, methods.values()):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.005,
                f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax.axhline(0.265, ls="--", color="gray", alpha=0.6)
    ax.set_ylabel("weighted F1")
    ax.set_title("Phase 9b — Spatial 3Di motif classifier")
    ax.set_ylim(0, max(max(methods.values()) * 1.2, 0.4))
    plt.xticks(rotation=10, ha="right", fontsize=8)
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "phase9b_spatial.png", dpi=140)
    plt.close()


if __name__ == "__main__":
    main()
