import os
import numpy as np
import pandas as pd


def make_synth(n: int = 1000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    # numeric features
    lines_changed = rng.normal(100, 30, size=n).clip(0)
    files_changed = rng.integers(1, 20, size=n)
    comments = rng.poisson(3, size=n)
    author_reputation = rng.normal(0.5, 0.15, size=n).clip(0, 1)

    # categorical features
    base_branch = rng.choice(["main", "dev", "release"], size=n, p=[0.6, 0.3, 0.1])
    has_tests = rng.choice(["yes", "no"], size=n, p=[0.7, 0.3])

    # target generation (learnable signal)
    # close time (hours): increases with lines/files/comments; decreases with reputation and main branch
    close_time = (
        0.02 * lines_changed + 0.5 * files_changed + 1.0 * comments
        - 10.0 * author_reputation + rng.normal(0, 2.0, size=n)
    )
    close_time += np.where(base_branch == "release", 5.0, 0.0)
    close_time = close_time.clip(0)

    # merged probability: sigmoid of linear comb
    logits = (
        0.8 * author_reputation - 0.01 * lines_changed - 0.05 * files_changed - 0.1 * comments
    )
    logits += np.where(has_tests == "yes", 0.5, -0.3)
    logits += np.where(base_branch == "main", 0.3, 0.0)
    prob = 1 / (1 + np.exp(-logits))
    merged = (rng.random(n) < prob).astype(int)

    df = pd.DataFrame({
        "lines_changed": lines_changed,
        "files_changed": files_changed,
        "comments": comments,
        "author_reputation": author_reputation,
        "base_branch": base_branch,
        "has_tests": has_tests,
        "pr_close_time_hours": close_time,
        "pr_merged": merged,
    })
    return df


def main():
    out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "synth.csv")
    df = make_synth(n=1200)
    df.to_csv(out_path, index=False)
    print(f"Saved synth data to {out_path}")


if __name__ == "__main__":
    main()
import os
import numpy as np
import pandas as pd


def main(out_path: str = "./engineered/synth.csv", n: int = 2000, seed: int = 42):
    rng = np.random.RandomState(seed)
    # features
    lines_changed = rng.gamma(5.0, 10.0, size=n)
    files_touched = rng.poisson(3.0, size=n)
    author_exp = rng.randint(0, 10, size=n)
    repo = rng.choice(["A", "B", "C"], size=n, p=[0.5, 0.3, 0.2])
    weekday = rng.choice(list("Mon Tue Wed Thu Fri Sat Sun".split()), size=n)

    # targets
    pr_merged = (rng.rand(n) + 0.2*(author_exp/10) + 0.1*(files_touched<3) - 0.1*(lines_changed>80) > 0.5).astype(int)
    noise = rng.normal(0, 5, size=n)
    pr_close_time_hours = 48 + 0.2*lines_changed + 2.0*files_touched - 3.0*author_exp + noise - 8.0*pr_merged

    df = pd.DataFrame({
        "lines_changed": lines_changed,
        "files_touched": files_touched,
        "author_exp": author_exp,
        "repo": repo,
        "weekday": weekday,
        "pr_close_time_hours": pr_close_time_hours,
        "pr_merged": pr_merged,
    })
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Wrote {out_path} with shape {df.shape}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, default="./engineered/synth.csv")
    p.add_argument("--n", type=int, default=2000)
    args = p.parse_args()
    main(args.out, args.n)
