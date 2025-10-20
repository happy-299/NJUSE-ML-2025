import argparse
import os
import glob
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Merge multiple CSV files into one and add a 'repo' column from filename")
    parser.add_argument("--input_dir", required=True, help="Directory containing CSV files")
    parser.add_argument("--output", required=True, help="Output merged CSV path")
    parser.add_argument("--pattern", default="*.csv", help="Filename pattern to match (default: *.csv)")
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.input_dir, args.pattern)))
    if not files:
        raise SystemExit(f"No CSV files found under {args.input_dir} with pattern {args.pattern}")

    dfs = []
    for fp in files:
        try:
            df = pd.read_csv(fp)
        except UnicodeDecodeError:
            df = pd.read_csv(fp, encoding="utf-8", errors="ignore")
        repo_name = os.path.splitext(os.path.basename(fp))[0]
        df["repo"] = repo_name
        dfs.append(df)

    merged = pd.concat(dfs, ignore_index=True)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    merged.to_csv(args.output, index=False)
    print(f"Merged {len(files)} files into {args.output}, total rows: {len(merged)}")


if __name__ == "__main__":
    main()
