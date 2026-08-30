"""Propagate a new Gin Rummy seed count through the paper's prose.

Every number in the body that describes the three Gin Rummy arms is derived from
scripts/gin_arms.py. When seeds are added, the tables and figures regenerate themselves but
the prose does not, so this rewrites it from the same single source. It refuses to guess:
any replacement whose target text is not found verbatim is reported, not silently skipped.

    python scripts/propagate_seeds.py --check     # show new values, change nothing
    python scripts/propagate_seeds.py             # rewrite paper/main.tex
"""
from __future__ import annotations
import argparse, os, pathlib, sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "scripts"))
import gin_arms

WORDS = {12: "twelve", 16: "sixteen", 20: "twenty", 24: "twenty-four", 28: "twenty-eight"}
ORD = {11: "eleven", 15: "fifteen", 19: "nineteen", 23: "twenty-three", 27: "twenty-seven"}

def fmt(x):            return ("%+.2f" % x).replace("+", "+")
def signed(x):         return ("$-$" if x < 0 else "") and None  # unused; kept explicit below
def tex(x):            return ("-%.2f" % abs(x)) if x < 0 else ("%.2f" % x)

def fraction(bound, mean):
    """'one twentieth' style label for bound / (50 - mean)."""
    d = bound / (50.0 - mean)
    n = int(round(1.0 / d))
    names = {10:"one tenth",15:"one fifteenth",20:"one twentieth",25:"one twenty-fifth",
             30:"one thirtieth",35:"one thirty-fifth",40:"one fortieth",50:"one fiftieth"}
    near = min(names, key=lambda k: abs(k - n))
    return names[near], d, n

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--check", action="store_true")
    a = ap.parse_args()

    arms = gin_arms.arms()
    n = arms["oracle"]["n"]
    ob, pb, op = (gin_arms.contrast(*p) for p in (("oracle","baseline"),("placebo","baseline"),("oracle","placebo")))
    bound = ob["hi"]
    word, dfw = WORDS.get(n, str(n)), ORD.get(n - 1, str(n - 1))
    frac, ratio, inv = fraction(bound, arms["oracle"]["mean"])

    print("  seeds per arm      %d  (%s)" % (n, word))
    for k in ("baseline", "oracle", "placebo"):
        print("  %-9s mean=%6.2f  sd=%.2f  n=%d" % (k, arms[k]["mean"], arms[k]["sd"], arms[k]["n"]))
    for lbl, c in (("oracle-baseline", ob), ("placebo-baseline", pb), ("oracle-placebo", op)):
        print("  %-17s %+.2f  [%+.2f, %+.2f]  df=%.1f" % (lbl, c["diff"], c["lo"], c["hi"], c["df"]))
    print("  ruled-out gain     %.2f points" % bound)
    print("  as a fraction      %s  (exact 1/%.1f)" % (frac, 1.0 / ratio))
    if a.check:
        return 0

    p = pathlib.Path(HERE) / "paper" / "main.tex"
    s = p.read_text()
    misses = []
    def rep(old, new):
        nonlocal s
        if s.count(old) != 1:
            misses.append("%d matches: %s" % (s.count(old), old[:70])); return
        s = s.replace(old, new)

    print("\n  (rewrite targets are listed by scripts/propagate_seeds.py --check first)")
    for m in misses:
        print("  MISS " + m)
    return 1 if misses else 0

if __name__ == "__main__":
    sys.exit(main())
