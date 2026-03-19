"""
fit_classifier.py
Obtain a random-forest edge classifier from an ilastik .ilp project file.

Two strategies are available:

* **ilp** (default): extract the *already-trained* vigra random forests
  stored inside the .ilp and wrap them in a thin sklearn-compatible adapter
  (``VigraRfSklearnWrapper``).  This gives predictions identical to ilastik.
  (No pickling needed; the classifier is already in the .ilp.)

* **sklearn**: re-fit a new ``sklearn.ensemble.RandomForestClassifier`` on the
  training data cached in the .ilp and save it as a pickle.

Usage (CLI)
-----------
    # Extract the RF from the .ilp (default; returns a wrapper, not pickled)
    python fit_classifier.py --ilp my_project.ilp --output rf.pkl

    # Re-fit a sklearn RF and pickle it
    python fit_classifier.py --ilp my_project.ilp --output rf.pkl \
        --classifier-source sklearn --n-estimators 100 --n-jobs 8

Usage (Python)
--------------
    from fit_classifier import extract_vigra_rf_from_ilp, fit_rf_from_ilp

    rf = extract_vigra_rf_from_ilp("my_project.ilp")   # from .ilp (no pickling)
    rf = fit_rf_from_ilp("my_project.ilp")              # sklearn re-fit

    import pickle
    with open("rf.pkl", "wb") as f:
        pickle.dump(rf, f)  # only needed for sklearn
"""

import argparse
import os
import pickle
import sys
import tempfile

import h5py
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from ilp_reader import read_feature_names, read_training_data


# ---------------------------------------------------------------------------
# Vigra RF wrapper with sklearn-compatible interface
# ---------------------------------------------------------------------------

class VigraRfSklearnWrapper:
    """Wrap one or more ``vigra.learning.RandomForest`` objects so they
    expose the two attributes consumed by the multicut pipeline:

    * ``classes_``  – 1-D array of class labels (e.g. ``[1, 2]``)
    * ``predict_proba(X)`` – return ``(n_samples, n_classes)`` probabilities

    The aggregation across sub-forests mirrors ilastik's
    ``ParallelVigraRfLazyflowClassifier.predict_probabilities``: each
    forest's output is weighted by its tree count, then normalised by the
    total number of trees.
    """

    def __init__(self, forests, known_labels):
        self._forests = forests
        self._tree_counts = [f.treeCount() for f in forests]
        self._total_trees = sum(self._tree_counts)
        self.classes_ = np.array(known_labels)

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float32)
        assert X.ndim == 2
        result = None
        for forest, tc in zip(self._forests, self._tree_counts):
            probs = forest.predictProbabilities(X)
            probs *= tc
            if result is None:
                result = probs
            else:
                result += probs
        result /= self._total_trees
        return result

    def __repr__(self):
        return (
            f"VigraRfSklearnWrapper("
            f"{len(self._forests)} forests, "
            f"{self._total_trees} trees, "
            f"classes={self.classes_.tolist()})"
        )


def extract_vigra_rf_from_ilp(ilp_path: str) -> VigraRfSklearnWrapper:
    """Extract the trained vigra random forests from an ilastik .ilp file.

    Parameters
    ----------
    ilp_path : str
        Path to the ilastik ``.ilp`` project file.

    Returns
    -------
    VigraRfSklearnWrapper
        A classifier with ``classes_`` and ``predict_proba`` matching the
        sklearn interface expected by the multicut pipeline.
    """
    import vigra  # intentionally late so ImportError is catchable

    print(f"Extracting vigra RF from {ilp_path} …")

    # vigra cannot read from an already-open HDF5 file (non-shared DLL issue),
    # so we copy the classifier group to a temporary file first.
    tmp_dir = tempfile.mkdtemp()
    cache_path = os.path.join(tmp_dir, "tmp_classifier_cache.h5").replace("\\", "/")

    try:
        with h5py.File(ilp_path, "r") as h5:
            src = h5["Training and Multicut/Output"]

            # Copy forest data to temp file
            with h5py.File(cache_path, "w") as cache:
                cache.copy(src, "Output")

            # Read known_labels while the file is open
            try:
                known_labels = list(src["known_labels"][:])
            except KeyError:
                # Older projects didn't store labels; infer from first forest
                known_labels = None

        # Load each sub-forest from the temp file
        forests = []
        with h5py.File(cache_path, "r") as cache:
            grp = cache["Output"]
            forest_keys = sorted(k for k in grp.keys() if k.startswith("Forest"))

        for fk in forest_keys:
            target = f"Output/{fk}"
            forests.append(vigra.learning.RandomForest(cache_path, target))

        if known_labels is None:
            known_labels = list(range(1, forests[0].labelCount() + 1))

        total_trees = sum(f.treeCount() for f in forests)
        print(
            f"  Loaded {len(forests)} sub-forests, "
            f"{total_trees} trees total, "
            f"labels {known_labels}"
        )

    finally:
        if os.path.exists(cache_path):
            os.remove(cache_path)
        if os.path.exists(tmp_dir):
            os.rmdir(tmp_dir)

    return VigraRfSklearnWrapper(forests, known_labels)


def fit_rf_from_ilp(
    ilp_path: str,
    lane=None,
    n_estimators: int = 100,
    n_jobs: int = -1,
    random_state: int = 42,
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
    print(f"Reading training data from {ilp_path} ({lane_desc}) …")

    X, y, feature_cols = read_training_data(ilp_path, lane=lane)

    classes, counts = np.unique(y, return_counts=True)
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

    fn = read_feature_names(ilp_path)
    print("Feature names per channel:")
    for ch, feats in fn.items():
        print(f"  {ch}: {feats}")

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

    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(rf, X, y, cv=min(5, counts.min()), scoring="f1_macro")
    print(f"  5-fold CV F1 (macro): {scores.mean():.3f} ± {scores.std():.3f}")

    return rf


def main():
    parser = argparse.ArgumentParser(
        description="Obtain a random-forest edge classifier from an ilastik .ilp file."
    )
    parser.add_argument("--ilp", required=True, help="Path to ilastik .ilp project file")
    parser.add_argument("--output", required=True, help="Output path for pickled classifier")
    parser.add_argument(
        "--classifier-source",
        choices=["ilp", "sklearn"],
        default="ilp",
        help=(
            "Where to get the classifier.  'ilp' (default) extracts the "
            "already-trained vigra RF from the .ilp (identical to ilastik).  "
            "'sklearn' re-fits a new sklearn RF from the cached training data."
        ),
    )
    parser.add_argument(
        "--lane", type=int, default=None,
        help="Lane index (default: None = all lanes concatenated).  Only used with --classifier-source sklearn.",
    )
    parser.add_argument(
        "--n-estimators", type=int, default=100,
        help="Number of trees (only used with --classifier-source sklearn).",
    )
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    if args.classifier_source == "ilp":
        rf = extract_vigra_rf_from_ilp(args.ilp)
    else:
        rf = fit_rf_from_ilp(
            ilp_path=args.ilp,
            lane=args.lane,
            n_estimators=args.n_estimators,
            n_jobs=args.n_jobs,
            random_state=args.random_state,
        )

    with open(args.output, "wb") as f:
        pickle.dump(rf, f)
    print(f"\nSaved classifier to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
