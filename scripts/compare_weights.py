"""
Used to verify bitwise identity between edge weights (probabilities)
and costs (after applying threshold and beta) in blimp vs ilastik.

Export the intermediate arrays like
`np.save(r'C:\Users\root\EM\blimp\blimp-output\weights.npy', probs)`
"""
import numpy as np
import sys

if __name__ == "__main__":
    a = np.load(sys.argv[1])
    b = np.load(sys.argv[2])

    print(f"Shapes: {a.shape} vs {b.shape}")
    print(f"Dtypes: {a.dtype} vs {b.dtype}")

    if a.shape != b.shape or a.dtype != b.dtype:
        print("Cannot compare further.")
        sys.exit(1)

    if np.array_equal(a, b, equal_nan=True):
        print("Bitwise identical (NaN-aware)")
    else:
        diff = np.abs(a - b)
        n_bitwise = (a.view(np.uint8).reshape(len(a), -1) != b.view(np.uint8).reshape(len(b), -1)).any(axis=1).sum()
        n_tol = (diff > 1e-4).sum()
        if n_tol > 0:
            idx = np.where(diff > 1e-4)[0]
            print(f"\nFirst 10 examples beyond 1e-4:")
            print(f"{'i':>6}  {'yours':>12}  {'theirs':>12}  {'diff':>12}")
            for i in idx[:10]:
                print(f"{i:>6}  {a[i]:>12.6f}  {b[i]:>12.6f}  {diff[i]:>12.6f}")
        print(f"Bitwise:    {n_bitwise}/{len(a)} elements differ")
        print(f"Beyond 1e-4: {n_tol}/{len(a)} elements differ")
        print(f"max abs diff:  {diff.max()}")
        print(f"mean abs diff: {diff.mean()}")
        print(f"yours  range: {a.min():.6f} – {a.max():.6f}")
        print(f"theirs range: {b.min():.6f} – {b.max():.6f}")
