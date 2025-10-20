import json
import os


def load_results(path: str):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def fmt_reg(m):
    return f"MAE={m['MAE']:.2f} | MSE={m['MSE']:.2f} | RMSE={m['RMSE']:.2f} | R2={m['R2']:.4f}"


def fmt_cls(m):
    return (
        f"Acc={m['Accuracy']:.4f} | P={m['Precision']:.4f} | R={m['Recall']:.4f} "
        f"| F1={m['F1']:.4f} | F1_macro={m['F1_macro']:.4f}"
    )


def main():
    runs = {
        "django_mmoe": "./outputs/django_mmoe/results.json",
        "django_shared": "./outputs/django_shared/results.json",
        "django_default": "./outputs/results.json",  # 若你用 configs/django.yaml 直接跑过
    }

    print("=== Experiment Results Comparison ===")
    for name, path in runs.items():
        res = load_results(path)
        if not res:
            print(f"- {name}: not found -> {path}")
            continue
        print(f"\n[{name}] best_epoch={res.get('best_epoch')}  val_best={res.get('val_best'):.4f}")
        print("  Regression:", fmt_reg(res["test_reg"]))
        print("  Classification:", fmt_cls(res["test_cls"]))


if __name__ == "__main__":
    main()
