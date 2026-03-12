"""
fit_classifier.py
Re-fit a sklearn RandomForestClassifier on training data extracted from an
ilastik .ilp project file, then save it as a pickle for use at inference time.

The training data (feature vectors + merge/split labels) is read directly from
the cached EdgeFeatures and EdgeLabelsDict stored inside the .ilp file, so no
feature re-computation is needed and the feature space is identical to what the
vigra RF inside ilastik was trained on.

Usage (CLI)
-----------
    python fit_classifier.py \
        --ilp my_project.ilp \
        --output rf.pkl \
        --n-estimators 200 \
        --n-jobs 8

Usage (Python)
--------------
    from fit_classifier import fit_rf_from_ilp
    rf = fit_rf_from_ilp("my_project.ilp")
    import pickle
    with open("rf.pkl", "wb") as f:
        pickle.dump(rf, f)
"""

import argparse
import pickle
import sys

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from ilp_reader import read_feature_names, read_training_data


def fit_rf_from_ilp(
    ilp_path: str,
    lane=None,
    n_estimators: int = 200,
    n_jobs: int = -1,
    random_state: int = 42,
    verbose: bool = True,
) -> RandomForestClassifier:
    """
    Read training data from *ilp_path* and return a fitted sklearn RF.

    Parameters
    ----------
    ilp_path : str
        Path to the ilastik .ilp project file.
    lane : int or None
        Lane index to read, or None (default) to read and concatenate all
        lanes.  Use None for multi-lane projects (e.g. trained on several
        sub-volume crops).
    n_estimators : int
        Number of trees in the random forest.
    n_jobs : int
        Number of parallel jobs for fitting (-1 = all CPUs).
    random_state : int
        Random seed for reproducibility.
    verbose : bool
        Print progress information.

    Returns
    -------
    sklearn.ensemble.RandomForestClassifier
        Fitted classifier. Call ``rf.predict_proba(X)[:, 1]`` to get the
        boundary probability (class 2 = split) for each edge.

    Notes
    -----
    Labels in ilastik are: 1 = merge, 2 = split/boundary.
    sklearn uses 0-indexed classes, but predict_proba columns correspond to
    sorted unique label values [1, 2], so column index 1 → class 2 (split).
    When class labels are exactly {1, 2}, ``rf.predict_proba(X)[:, 1]``
    gives P(split), which is the boundary probability expected by elf.
    """
    lane_desc = "all lanes" if lane is None else f"lane {lane}"
    if verbose:
        print(f"Reading training data from {ilp_path} ({lane_desc}) …")

    X, y, feature_cols = read_training_data(ilp_path, lane=lane)

    classes, counts = np.unique(y, return_counts=True)
    if verbose:
        for cls, cnt in zip(classes, counts):
            label_name = {1: "merge", 2: "split"}.get(int(cls), str(cls))
            print(f"  class {cls} ({label_name}): {cnt} examples")
        print(f"  {len(feature_cols)} features per edge")

    if len(classes) < 2:
        raise ValueError(
            "Training data contains only one class. "
            "Annotate both merge (1) and split (2) edges in ilastik before "
            "extracting the classifier."
        )

    if verbose:
        fn = read_feature_names(ilp_path)
        print("Feature names per channel:")
        for ch, feats in fn.items():
            print(f"  {ch}: {feats}")

    if verbose:
        print(
            f"\nFitting RandomForestClassifier "
            f"(n_estimators={n_estimators}, n_jobs={n_jobs}) …"
        )

    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        n_jobs=n_jobs,
        random_state=random_state,
    )
    rf.fit(X, y)

    if verbose:
        from sklearn.model_selection import cross_val_score
        scores = cross_val_score(rf, X, y, cv=min(5, counts.min()), scoring="f1_macro")
        print(f"  5-fold CV F1 (macro): {scores.mean():.3f} ± {scores.std():.3f}")

    return rf


def main():
    parser = argparse.ArgumentParser(
        description="Fit a sklearn RF from ilastik .ilp training data."
    )
    parser.add_argument("--ilp", required=True, help="Path to ilastik .ilp project file")
    parser.add_argument("--output", required=True, help="Output path for pickled sklearn RF")
    parser.add_argument(
        "--lane", type=int, default=None,
        help="Lane index (default: None = all lanes concatenated)",
    )
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    rf = fit_rf_from_ilp(
        ilp_path=args.ilp,
        lane=args.lane,
        n_estimators=args.n_estimators,
        n_jobs=args.n_jobs,
        random_state=args.random_state,
        verbose=True,
    )

    with open(args.output, "wb") as f:
        pickle.dump(rf, f)
    print(f"\nSaved classifier to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
