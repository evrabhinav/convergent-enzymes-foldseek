"""
End-to-end orchestrator. Run after AlphaFold structures are downloaded.

  python src/run_all.py

Stages:
  1. phase2_features         (skip if features/feature_matrix.npz exists)
  2. sequence_features       (skip if features/sequence_feature_matrix.npz exists)
  3. phase3_train_eval       always re-run; cheap
  4. phase4_combined         always re-run; cheap

Pass --force to redo every stage.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import phase2_features
import phase3_train_eval
import phase4_combined
import sequence_features

ROOT = Path(__file__).resolve().parent.parent
FEATURES_DIR = ROOT / "features"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if args.force or not (FEATURES_DIR / "feature_matrix.npz").exists():
        print(">>> Phase 2: structural feature extraction")
        phase2_features.main()
    else:
        print(">>> Phase 2 skipped (features/feature_matrix.npz exists; --force to redo)")

    if args.force or not (FEATURES_DIR / "sequence_feature_matrix.npz").exists():
        print(">>> Sequence features")
        sequence_features.main()
    else:
        print(">>> Sequence features skipped")

    print(">>> Phase 3: train + evaluate (structural)")
    phase3_train_eval.main()

    print(">>> Phase 4: combined features")
    phase4_combined.main()


if __name__ == "__main__":
    main()
