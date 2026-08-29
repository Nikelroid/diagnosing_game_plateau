"""Run the oracle-observation learner: the baseline recipe, one extra observation plane.

The comparison only means something if nothing else moves, so the config is not written by hand.
It is read from the baseline run's own config file, the name and seed are changed, and everything
else is passed through untouched. The training code itself is the main repository's, unmodified;
this script only switches the observation on before importing it.

    python scripts/train_oracle.py --seed 0                  # 3M steps, the baseline budget
    python scripts/train_oracle.py --seed 0 --smoke          # 20k steps, checks the plumbing

Results land in the main repository's sweep/curriculum/ as oracleobs_s<seed>.json, next to the
ppoarch_mlp_s<seed>.json baselines they are compared against.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# The curated seed models were pickled under the conda environment's NumPy 2.4.6, but
# ~/.local/lib holds NumPy 1.26.4, which shadows it whenever user site-packages are enabled.
# Loading a NumPy 2 pickle under NumPy 1 fails with "No module named numpy._core", and forcing
# the module alias segfaults, because the array reconstruction ABI differs. The project's own
# SLURM launcher sets PYTHONNOUSERSITE=1 for exactly this reason. The variable is read at
# interpreter start, so setting it here is too late and the process re-executes itself once.
if not os.environ.get("PYTHONNOUSERSITE"):
    os.execve(
        sys.executable,
        [sys.executable] + sys.argv,
        {**os.environ, "PYTHONNOUSERSITE": "1"},
    )

DEFAULT_MAIN_REPO = os.environ.get(
    "ADVCOEV_ROOT", os.path.expanduser("~/Adversarial-CoEvolution")
)


def build_config(main_repo: str, seed: int, steps: int, out_dir: str, arm: str = "oracle") -> str:
    """Copy the baseline config, change only the name, seed and step budget."""
    baseline = os.path.join(main_repo, "sweep", "ppoarch_cfgs", f"ppoarch_mlp_s{seed}.json")
    with open(baseline) as f:
        cfg = json.load(f)

    cfg["name"] = {"oracle": f"oracleobs_s{seed}", "placebo": f"placebo_s{seed}",
                   "baseline": f"basewide_s{seed}"}[arm]
    cfg["seed"] = seed
    cfg["steps"] = steps
    cfg["purpose"] = (
        "Oracle-observation probe: the ppoarch_mlp recipe with the opponent's hand appended "
        "as a fifth observation plane. Everything else is the baseline config verbatim."
    )
    if steps < 100_000:  # smoke runs must not sit for an hour in the final evaluation
        cfg["eval_every"] = max(5_000, steps // 2)
        cfg["ckpt_eval_games"] = 20
        cfg["final_eval_games"] = 20
        cfg["save_freq"] = steps

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{cfg['name']}.json")
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)
    return path


def main() -> int:
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arm", choices=("oracle", "placebo", "baseline"), default="oracle",
                    help="oracle = the opponent's real hand; placebo = same width, no "
                         "information; baseline = the original four planes")
    ap.add_argument("--steps", type=int, default=3_000_000)
    ap.add_argument("--smoke", action="store_true", help="20k steps, just to prove it runs")
    ap.add_argument("--main-repo", default=DEFAULT_MAIN_REPO)
    ap.add_argument("--config-dir", default=os.path.join(here, "data", "configs"))
    args = ap.parse_args()

    steps = 20_000 if args.smoke else args.steps
    main_repo = os.path.abspath(args.main_repo)
    if not os.path.isdir(main_repo):
        print(f"main repository not found at {main_repo}; pass --main-repo", file=sys.stderr)
        return 2

    cfg_path = build_config(main_repo, args.seed, steps, args.config_dir, args.arm)
    print(f"[oracle] config  {cfg_path}")
    print(f"[oracle] steps   {steps:,}")

    # the training code resolves its own imports relative to the repository root
    sys.path.insert(0, main_repo)
    sys.path.insert(0, here)
    os.environ["CFG"] = cfg_path
    os.environ.setdefault("OMP_NUM_THREADS", "1")  # forked vec envs deadlock otherwise

    from oracle_probes.oracle_env import enable, enable_placebo
    from oracle_probes.opponent_shim import enable_opponent_trim

    if args.arm == "oracle":
        enable(); enable_opponent_trim()
    elif args.arm == "placebo":
        enable_placebo(); enable_opponent_trim()
    else:
        pass                      # baseline keeps the original four planes, nothing patched
    print(f"[oracle] arm={args.arm}: "
          + ("four planes, unmodified" if args.arm == "baseline"
             else "observation widened to 5 planes; opponents trimmed to their own width"))

    from sweep import curriculum_train

    return curriculum_train.main() or 0


if __name__ == "__main__":
    sys.exit(main())
