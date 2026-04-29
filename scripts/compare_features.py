"""
Used to verify bitwise identity between feature dataframes from blimp vs ilastik.

Export the intermediate dataframes like:
```
    features_path = r'C:\Users\root\EM\blimp\blimp-output\features.pkl'
    feature_df = pd.concat(feature_dfs, axis=1)
    feature_df.to_pickle(features_path)
```
"""
import pandas as pd
import argparse

def compare_dfs(df1, df2):
    if df1.shape != df2.shape:
        print("Shapes differ:", df1.shape, df2.shape)
    else:
        print("Shapes eq!", df1.shape)

    if not df1.dtypes.equals(df2.dtypes):
        print("Dtypes differ!")
        print("df1 dtypes:\n", df1.dtypes)
        print("df2 dtypes:\n", df2.dtypes)
    else:
        print("Dtypes eq!")
        print(df1.dtypes)

    if not df1.index.equals(df2.index):
        print("Index differs!")
        print("df1 index:\n", df1.index)
        print("df2 index:\n", df2.index)
    else:
        print("Index eq!", df1.index)

    if not df1.equals(df2):
        print("Values differ!")
        diff = df1.compare(df2)
        if not diff.empty:
            print("Differences:\n", diff)
        else:
            print("No element-wise differences found, but equals() returned False (check NaNs or object dtypes).")

def main():
    parser = argparse.ArgumentParser(description='Compare two pickled DataFrames for bitwise identity.')
    parser.add_argument('blimp_features', help='Path to blimp-features.pkl')
    parser.add_argument('ilastik_features', help='Path to ilastik-features.pkl')
    args = parser.parse_args()

    df1 = pd.read_pickle(args.blimp_features)
    df2 = pd.read_pickle(args.ilastik_features)

    compare_dfs(df1, df2)

if __name__ == '__main__':
    main()