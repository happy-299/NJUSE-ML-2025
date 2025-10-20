import os
import glob
import pandas as pd


def list_columns(path: str):
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext in [".xlsx", ".xls"]:
            df = pd.read_excel(path)
        elif ext == ".csv":
            df = pd.read_csv(path)
        else:
            print(f"[skip] {path} (unsupported ext)")
            return
        cols = df.columns.tolist()
        print(f"\n=== {os.path.basename(path)} ({len(cols)} columns) ===")
        for c in cols:
            print(c)
    except Exception as e:
        print(f"[error] {path}: {e}")


def main():
    root = os.path.dirname(os.path.dirname(__file__))
    eng_dir = os.path.join(root, "engineered")
    files = (
        sorted(glob.glob(os.path.join(eng_dir, "*.xlsx")))
        + sorted(glob.glob(os.path.join(eng_dir, "*.csv")))
    )
    if not files:
        print("No dataset files found under engineered/.")
        return
    print(f"Found {len(files)} files under engineered/...")
    for f in files:
        list_columns(f)


if __name__ == "__main__":
    main()
