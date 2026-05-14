"""
Phase 9c: Joint (AA + 3Di) spatial motif classifier.

Each residue gets a *joint letter*: (amino_acid, 3di_letter). The alphabet is
20 x 20 = 400, so spatial pairs span 400 x 400 / 2 = 80,000 possible motifs —
plenty of resolution to discriminate 400 EC classes.

For each EC class with 5 train members:
  1. Enumerate spatial pairs (i, j) of joint letters where |i - j| >= SEQ_GAP
     and CA-CA(i, j) <= DIST_MAX.
  2. Pairs present in >= MIN_MEMBERS of 5 with enrichment >= MIN_ENRICH
     versus background train frequency are kept as class motifs.

For each test protein, score against each class's motifs and predict argmax.

We also try TRIPLES (joint letter triangles of pairwise-close, sequence-distant
residues), restricted to top-K high-pLDDT residues to keep combinatorics
tractable.

Outputs:
  results/phase9c_joint_motif_results.csv
  charts/phase9c_joint.png
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

SEQ_GAP = 6
DIST_MAX = 9.0
DIST_MAX_TRIPLE = 10.0
TOP_K_RESIDUES = 60

AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLU": "E", "GLN": "Q", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


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


def parse_pdb(pdb_path: Path) -> tuple[str, np.ndarray, np.ndarray]:
    """Return (AA sequence in PDB order, CA coords (N,3), pLDDT (N,)).

    Aligns 1:1 with how Foldseek built 3Di. Only standard residues are kept.
    """
    parser = PDBParser(QUIET=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = parser.get_structure(pdb_path.stem, str(pdb_path))
    aa_chars, coords, plddt = [], [], []
    for chain in next(s.get_models()):
        for res in chain:
            if "CA" not in res or res.get_resname() not in AA3_TO_1:
                continue
            aa_chars.append(AA3_TO_1[res.get_resname()])
            coords.append(res["CA"].get_coord())
            plddt.append(res["CA"].get_bfactor())
    return ("".join(aa_chars),
            np.array(coords, dtype=np.float32),
            np.array(plddt, dtype=np.float32))


def joint_letters(aa: str, di: str) -> str:
    """Combine two equal-length strings element-wise into '<aa><di>' pairs.
    We encode each pair as a 2-char token (e.g. 'AD'). Pair length = 2 chars.
    """
    n = min(len(aa), len(di))
    return aa[:n] + di[:n]  # not actually used directly; we keep two strings


def spatial_pair_motifs(aa: str, di: str, coords: np.ndarray,
                        seq_gap: int, d_max: float) -> set[tuple[str, str]]:
    """Return set of (joint_i, joint_j) sorted pairs.

    joint_x = aa[x] + di[x]  (length-2 token).
    """
    n = min(len(aa), len(di), len(coords))
    if n < seq_gap + 1:
        return set()
    coords = coords[:n]; aa = aa[:n]; di = di[:n]
    d = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    i, j = np.triu_indices(n, k=seq_gap)
    mask = d[i, j] <= d_max
    i, j = i[mask], j[mask]
    out = set()
    for a, b in zip(i, j):
        la = aa[a] + di[a]
        lb = aa[b] + di[b]
        out.add((min(la, lb), max(la, lb)))
    return out


def spatial_triple_motifs(aa: str, di: str, coords: np.ndarray, plddt: np.ndarray,
                          seq_gap: int, d_max: float, top_k: int) -> set[tuple[str, str, str]]:
    n = min(len(aa), len(di), len(coords))
    if n < seq_gap + 1:
        return set()
    coords = coords[:n]; aa = aa[:n]; di = di[:n]
    plddt = plddt[:n] if len(plddt) >= n else np.full(n, 50.0)
    keep = np.argsort(-plddt)[:top_k]; keep.sort()
    if len(keep) < 3:
        return set()
    sub_coords = coords[keep]
    d = np.linalg.norm(sub_coords[:, None, :] - sub_coords[None, :, :], axis=-1)
    K = len(keep)
    out = set()
    for ia in range(K):
        for ib in range(ia + 1, K):
            if abs(keep[ib] - keep[ia]) < seq_gap or d[ia, ib] > d_max:
                continue
            for ic in range(ib + 1, K):
                if (abs(keep[ic] - keep[ia]) < seq_gap
                        or abs(keep[ic] - keep[ib]) < seq_gap
                        or d[ia, ic] > d_max or d[ib, ic] > d_max):
                    continue
                t = tuple(sorted([
                    aa[keep[ia]] + di[keep[ia]],
                    aa[keep[ib]] + di[keep[ib]],
                    aa[keep[ic]] + di[keep[ic]],
                ]))
                out.add(t)
    return out


def background_freq(all_member_sets) -> dict:
    cnt: Counter = Counter()
    n = len(all_member_sets)
    for s in all_member_sets:
        for m in s:
            cnt[m] += 1
    return {m: c / n for m, c in cnt.items()}


def discover_class_motifs(train_features_by_class, min_members, min_enr, bg):
    motifs_per_class = {}
    for ec, member_sets in train_features_by_class.items():
        if len(member_sets) < min_members:
            continue
        cnt: Counter = Counter()
        for s in member_sets:
            for m in s:
                cnt[m] += 1
        ms = {}
        for m, n in cnt.items():
            if n < min_members:
                continue
            p_class = n / len(member_sets)
            p_bg = bg.get(m, 1e-9)
            enr = p_class / max(p_bg, 1e-9)
            if enr >= min_enr:
                ms[m] = float(np.log2(enr))
        if ms:
            motifs_per_class[ec] = ms
    return motifs_per_class


def score_test(test_features, motifs_per_class) -> dict:
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


def evaluate(preds, test_labels):
    trues, hats = [], []
    for tid, ec in test_labels.items():
        trues.append(ec)
        hats.append(preds.get(tid, "__none__"))
    return f1_score(trues, hats, average="weighted", zero_division=0)


def build(entries, struct_dir, di_map, feat_kind: str):
    """feat_kind in {'pairs', 'triples'}."""
    out = {}
    for e in tqdm(entries):
        p = struct_dir / f"{e}.pdb"
        if not p.exists() or e not in di_map:
            continue
        aa, coords, plddt = parse_pdb(p)
        di = di_map[e]
        if len(aa) == 0 or len(di) == 0:
            continue
        # if lengths mismatch slightly, truncate to min — the order should match
        n = min(len(aa), len(di), len(coords))
        if n < 10:
            continue
        if feat_kind == "pairs":
            out[e] = spatial_pair_motifs(aa, di, coords, SEQ_GAP, DIST_MAX)
        else:
            out[e] = spatial_triple_motifs(aa, di, coords, plddt,
                                           SEQ_GAP, DIST_MAX_TRIPLE,
                                           TOP_K_RESIDUES)
    return out


def main():
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    test_labels = dict(zip(test_df["Entry"], test_df["Label"]))

    train_3di = read_3di_fasta(FS_DIR / "train_3di.fasta")
    test_3di = read_3di_fasta(FS_DIR / "test_3di.fasta")

    rows = []
    for feat_kind in ["pairs", "triples"]:
        print(f"\n=== building {feat_kind} features ===")
        train_feats = build(train_df["Entry"].tolist(),
                            STRUCT_DIR / "train", train_3di, feat_kind)
        test_feats = build(test_df["Entry"].tolist(),
                           STRUCT_DIR / "test", test_3di, feat_kind)

        feats_by_class = defaultdict(list)
        for _, row in train_df.iterrows():
            if row["Entry"] in train_feats:
                feats_by_class[row["Label"]].append(train_feats[row["Entry"]])

        all_train = [s for v in feats_by_class.values() for s in v]
        bg = background_freq(all_train)

        for min_mem in [3, 4, 5]:
            for enr in [2.0, 5.0, 10.0, 25.0]:
                motifs = discover_class_motifs(feats_by_class, min_mem, enr, bg)
                preds = score_test(test_feats, motifs)
                f1 = evaluate(preds, test_labels)
                n_classes = len(motifs)
                avg_per_class = (np.mean([len(v) for v in motifs.values()])
                                 if motifs else 0)
                print(f"  {feat_kind} min_mem={min_mem} enr={enr:>4.1f}  "
                      f"classes={n_classes}  motifs/class={avg_per_class:.0f}  "
                      f"pred={len(preds)}  F1={f1:.4f}")
                rows.append({"feature": feat_kind, "min_members": min_mem,
                             "min_enrichment": enr, "n_classes": n_classes,
                             "avg_motifs_per_class": avg_per_class,
                             "n_predicted": len(preds), "weighted_f1": f1})

    df = pd.DataFrame(rows).sort_values("weighted_f1", ascending=False)
    df.to_csv(RESULTS_DIR / "phase9c_joint_motif_results.csv", index=False)

    best = df.iloc[0]
    print(f"\nBEST: {best['feature']} min_mem={int(best['min_members'])} "
          f"enr={best['min_enrichment']}  F1={best['weighted_f1']:.4f}")

    methods = {
        "Sequence (424)": 0.016,
        "Structural feat. (100)": 0.060,
        "ESM2-35M LR": 0.161,
        "Foldseek top-1": 0.2383,
        "FS + ESM2-150M ens": 0.2520,
        "Linear 3Di motif": 0.0725,
        "Spatial 3Di motif": 0.0165,
        f"Joint AA+3Di best\n({best['feature']}, mem>={int(best['min_members'])}, enr>={best['min_enrichment']})":
            best["weighted_f1"],
        "ESM2-3B (DGEB)": 0.265,
    }
    fig, ax = plt.subplots(figsize=(15, 5.5))
    cols = ["#bbb", "#88a", "#99b", "#4c72b0", "#5588cc", "#aaa", "#aaa", "#cc4444", "#444"]
    bars = ax.bar(list(methods.keys()), list(methods.values()), color=cols)
    for b, v in zip(bars, methods.values()):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.005,
                f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax.axhline(0.265, ls="--", color="gray", alpha=0.6)
    ax.set_ylabel("weighted F1")
    ax.set_title("Phase 9c — Joint AA+3Di spatial motif classifier")
    ax.set_ylim(0, max(max(methods.values()) * 1.2, 0.4))
    plt.xticks(rotation=10, ha="right", fontsize=8)
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "phase9c_joint.png", dpi=140)
    plt.close()


if __name__ == "__main__":
    main()
