import argparse
import json
import os
from typing import List, Dict, Any

import matplotlib.pyplot as plt


def load_history(out_dir: str):
    hpath = os.path.join(out_dir, "history.json")
    if not os.path.exists(hpath):
        return []
    with open(hpath, "r", encoding="utf-8") as f:
        return json.load(f)


def load_results(out_dir: str):
    rpath = os.path.join(out_dir, "results.json")
    if not os.path.exists(rpath):
        return None
    with open(rpath, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_loss(histories: Dict[str, List[Dict[str, Any]]], save_path: str):
    plt.figure(figsize=(6,4))
    for name, hist in histories.items():
        if not hist:
            continue
        xs = [r["epoch"] for r in hist]
        tr = [r["train_loss"] for r in hist]
        vl = [r["val_loss"] for r in hist]
        plt.plot(xs, tr, label=f"{name} train")
        plt.plot(xs, vl, label=f"{name} val")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Train/Val Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_bar(results: Dict[str, Dict[str, Any]], metric_keys: List[str], save_path: str, title: str):
    names = list(results.keys())
    values = [[results[n][k] for n in names] for k in metric_keys]

    plt.figure(figsize=(8, 4))
    x = range(len(names))
    width = 0.8 / len(metric_keys)
    for i, (k, vals) in enumerate(zip(metric_keys, values)):
        xs = [xx + (i - (len(metric_keys)-1)/2) * width for xx in x]
        plt.bar(xs, vals, width=width, label=k)
    plt.xticks(list(range(len(names))), names)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot loss curves and metric comparisons from outputs dirs")
    parser.add_argument("out_dirs", nargs="+", help="One or more output dirs that contain history.json/results.json")
    parser.add_argument("--out", default="./outputs/figures", help="Directory to save figures")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    histories = {os.path.basename(d.rstrip("/\\")): load_history(d) for d in args.out_dirs}
    # Loss curves
    plot_loss(histories, os.path.join(args.out, "loss.png"))

    # Metric bars
    results = {}
    for d in args.out_dirs:
        name = os.path.basename(d.rstrip("/\\"))
        res = load_results(d)
        if res is None:
            continue
        # flatten target metrics we care about
        results[name] = {
            "R2": res.get("test_reg", {}).get("R2"),
            "RMSE": res.get("test_reg", {}).get("RMSE"),
            "MAE": res.get("test_reg", {}).get("MAE"),
            "Accuracy": res.get("test_cls", {}).get("Accuracy"),
            "F1": res.get("test_cls", {}).get("F1"),
            "F1_macro": res.get("test_cls", {}).get("F1_macro"),
        }

    # Regression
    if results:
        plot_bar(results, ["R2"], os.path.join(args.out, "reg_R2.png"), "Regression R2 (higher=better)")
        plot_bar(results, ["RMSE", "MAE"], os.path.join(args.out, "reg_errors.png"), "Regression Errors (lower=better)")
        # Classification
        plot_bar(results, ["Accuracy", "F1", "F1_macro"], os.path.join(args.out, "cls_scores.png"), "Classification Scores (higher=better)")


if __name__ == "__main__":
    main()
