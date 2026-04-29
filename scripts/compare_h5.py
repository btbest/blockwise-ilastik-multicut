import h5py
import numpy as np
import sys

def compare_h5(path1, path2):
    with h5py.File(path1, 'r') as f1, h5py.File(path2, 'r') as f2:
        keys1, keys2 = set(f1.keys()), set(f2.keys())

        if keys1 != keys2:
            print(f"Keys differ: {keys1} vs {keys2}")
        else:
            print(f"Keys eq: {sorted(keys1)}")

        for key in sorted(keys1 & keys2):
            a, b = f1[key][()], f2[key][()]
            print(f"\n[{key}]")
            if a.shape != b.shape:
                print(f"  Shapes differ: {a.shape} {b.shape}")
            else:
                print(f"  Shapes eq: {a.shape}")

            if a.dtype != b.dtype:
                print(f"  Dtypes differ: {a.dtype} {b.dtype}")
            else:
                print(f"  Dtypes eq: {a.dtype}")

            if a.shape == b.shape and a.dtype == b.dtype:
                if np.array_equal(a, b, equal_nan=True):
                    print(f"  Values eq (NaN-aware)")
                else:
                    diff = np.where(a.view(np.uint8) != b.view(np.uint8))
                    print(f"  Values DIFFER — {len(diff[0])} byte(s) differ")

        a = f1['Raw Data'][()]
        b = f2['Raw Data'][()]

    diff = a - b
    print(f"max abs diff:  {np.abs(diff).max()}")
    print(f"mean abs diff: {np.abs(diff).mean()}")
    print(f"first range:  {a.min():.4f} – {a.max():.4f}")
    print(f"other range:  {b.min():.4f} – {b.max():.4f}")
    print(f"first mean/std: {a.mean():.4f} / {a.std():.4f}")
    print(f"other mean/std: {b.mean():.4f} / {b.std():.4f}")

if __name__ == "__main__":
    compare_h5(sys.argv[1], sys.argv[2])