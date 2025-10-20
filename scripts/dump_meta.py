import argparse
import json
import os
import sys

import torch


def main():
    parser = argparse.ArgumentParser(description="Dump training meta (features) from best checkpoint")
    parser.add_argument("out_dir", help="Output directory that contains best.pt")
    args = parser.parse_args()

    best_path = os.path.join(args.out_dir, "best.pt")
    if not os.path.exists(best_path):
        print(f"best.pt not found under {args.out_dir}")
        sys.exit(1)

    # Safe load when supported
    try:
        ckpt = torch.load(best_path, map_location="cpu", weights_only=True)  # type: ignore[arg-type]
    except TypeError:
        ckpt = torch.load(best_path, map_location="cpu")

    meta = ckpt.get("meta", ckpt if isinstance(ckpt, dict) else {})
    print(json.dumps({
        "numeric_features": meta.get("numeric_features"),
        "categorical_features": meta.get("categorical_features"),
        "one_hot_categorical": meta.get("one_hot_categorical"),
        "numeric_dim": meta.get("numeric_dim"),
        "cat_cardinalities": meta.get("cat_cardinalities"),
        "cat_onehot_dim": meta.get("cat_onehot_dim"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
