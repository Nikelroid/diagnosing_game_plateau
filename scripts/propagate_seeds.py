"""Propagate the Gin Rummy seed count and arm numbers through the paper's prose.

Tables and figures regenerate themselves from scripts/gin_arms.py; the prose does not. This
rewrites every sentence in paper/main.tex that quotes one of those numbers, from that same
single source, so a change in seed count cannot leave a stale figure in the text. It refuses
to guess: any replacement whose target is not found verbatim exactly once is reported and
nothing is written.

    python scripts/propagate_seeds.py --check    # print the new values, change nothing
    python scripts/propagate_seeds.py            # rewrite paper/main.tex
"""
from __future__ import annotations
import argparse, os, pathlib, sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "scripts"))
import gin_arms

CARD = {12: ("twelve", "Twelve"), 16: ("sixteen", "Sixteen"), 20: ("twenty", "Twenty"),
        23: ("twenty-three", "Twenty-three"), 24: ("twenty-four", "Twenty-four")}
ORD_DF = {11: "eleven", 15: "fifteen", 19: "nineteen", 22: "twenty-two", 23: "twenty-three"}
FRAC = {10: "one tenth", 15: "one fifteenth", 20: "one twentieth", 25: "one twenty-fifth",
        30: "one thirtieth", 40: "one fortieth", 50: "one fiftieth", 60: "one sixtieth",
        70: "one seventieth", 80: "one eightieth", 100: "one hundredth"}

def sgn(x):  # LaTeX signed number, matching the paper's own style
    return ("-%.2f" % abs(x)) if x < 0 else ("%.2f" % x)

def iv(c):
    return "[$%s$, $%s$]" % (sgn(c["lo"]), ("+%.2f" % c["hi"]) if c["hi"] >= 0 else sgn(c["hi"]))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    A = gin_arms.arms()
    ob = gin_arms.contrast("oracle", "baseline")
    pb = gin_arms.contrast("placebo", "baseline")
    op = gin_arms.contrast("oracle", "placebo")
    ns = {k: A[k]["n"] for k in ("baseline", "oracle", "placebo")}
    n = min(ns.values())
    bound = ob["hi"]
    base = A["baseline"]["mean"]
    inv = int(round((50.0 - base) / bound))
    frac = FRAC[min(FRAC, key=lambda k: abs(k - inv))]
    low, up = CARD.get(n, (str(n), str(n)))
    print("  n per arm        %s  (min of %s)" % (ns, n))
    for k in ("baseline", "oracle", "placebo"):
        print("  %-9s mean=%6.2f sd=%.2f n=%d" % (k, A[k]["mean"], A[k]["sd"], A[k]["n"]))
    for lbl, c in (("oracle-baseline", ob), ("placebo-baseline", pb), ("oracle-placebo", op)):
        print("  %-17s %s  %s  df=%.1f" % (lbl, sgn(c["diff"]), iv(c), c["df"]))
    print("  ruled-out gain   %.2f -> printed as $%.1f$ points" % (bound, bound))
    print("  fraction         %s  (exact 1/%d of %.1f points to parity)" % (frac, inv, 50 - base))
    if len(set(ns.values())) != 1:
        print("  NOTE arms have unequal n; prose will say '%s seeds per arm' using the minimum" % low)
    if a.check:
        return 0

    p = pathlib.Path(HERE) / "paper" / "main.tex"
    s = p.read_text(); miss = []
    def rep(old, new):
        nonlocal s
        if s.count(old) != 1:
            miss.append("%d matches: %r" % (s.count(old), old[:60])); return
        s = s.replace(old, new)

    B, O, P = (("%.2f" % A[k]["mean"]) for k in ("baseline", "oracle", "placebo"))
    bd = "%.1f" % bound
    rep(r"\newcommand{\ORACLEOBSMEAN}{26.79}", r"\newcommand{\ORACLEOBSMEAN}{%s}" % O)
    rep(r"\newcommand{\ORACLEOBSN}{twelve}", r"\newcommand{\ORACLEOBSN}{%s}" % low)
    rep("""detectable gain over the baseline arm: $-0.25$ points over twelve seeds, with interval
$[-1.70, +1.20]$. It also finishes $-0.23$ points below the placebo arm. Twelve seeds rule out
gains above about $1.2$ points""",
        """detectable gain over the baseline arm: $%s$ points over %s seeds, with interval
$[%s, %s]$. It also finishes $%s$ points below the placebo arm. %s seeds rule out
gains above about $%s$ points""" % (sgn(ob["diff"]), low, sgn(ob["lo"]),
        ("+%.2f" % ob["hi"]), sgn(op["diff"]), up, bd))
    rep("""Twelve seeds rule out
gains above about $1.2$ points from the raw hidden-hand channel""",
        """%s seeds rule out
gains above about $%s$ points from the raw hidden-hand channel""" % (up, bd))
    rep("at twelve seeds per arm", "at %s seeds per arm" % low)
    rep("""oracle arm reaches $26.79$ percent across twelve seeds, against $27.04$ for the baseline arm. The
difference is $-0.25$ points, with a 95 percent interval of $[-1.70, +1.20]$
(Figure~\\ref{fig:probes}). The placebo arm reaches $27.02$ percent, $-0.02$ points from baseline
with an interval of $[-1.51, +1.47]$.""",
        """oracle arm reaches $%s$ percent across %s seeds, against $%s$ for the baseline arm. The
difference is $%s$ points, with a 95 percent interval of $[%s, %s]$
(Figure~\\ref{fig:probes}). The placebo arm reaches $%s$ percent, $%s$ points from baseline
with an interval of $[%s, %s]$.""" % (O, low, B, sgn(ob["diff"]), sgn(ob["lo"]),
        "+%.2f" % ob["hi"], P, sgn(pb["diff"]), sgn(pb["lo"]), "+%.2f" % pb["hi"]))
    rep("""That contrast is $-0.23$ points, with interval $[-1.57, +1.11]$. Twelve seeds rule
out gains above about $1.2$ points. That limit is about one twentieth of the distance from this""",
        """That contrast is $%s$ points, with interval $[%s, %s]$. %s seeds rule
out gains above about $%s$ points. That limit is about %s of the distance from this"""
        % (sgn(op["diff"]), sgn(op["lo"]), "+%.2f" % op["hi"], up, bd, frac))
    rep("""and twelve seeds rule out gains above
about $1.2$ points.""", """and %s seeds rule out gains above
about $%s$ points.""" % (low, bd))
    rep("""where it rules out gains above
about $1.2$ points""", """where it rules out gains above
about $%s$ points""" % bd)
    rep("The twelve-seed", "The %s-seed" % low)
    rep("The Gin Rummy arms use twelve.", "The Gin Rummy arms use %s." % low)
    rep("to a bound worth stating at twelve.", "to a bound worth stating at %s." % low)
    rep("Twelve seeds per arm is what it took to say this.",
        "%s seeds per arm is what it took to say this." % up)
    rep("""arms at twelve seeds each, as win rate against the reference.
  Arm intervals are 95 percent from $t$ on eleven degrees of freedom. Contrast intervals use
  independent-arm standard errors with Welch degrees of freedom, about 21 here.}""",
        """arms at %s seeds each, as win rate against the reference.
  Arm intervals are 95 percent from $t$ on %s degrees of freedom. Contrast intervals use
  independent-arm standard errors with Welch degrees of freedom, about %d here.}"""
        % (low, ORD_DF.get(n - 1, str(n - 1)), round(ob["df"])))

    if miss:
        print("\n  REFUSED, nothing written:")
        for m in miss: print("    " + m)
        return 1
    p.write_text(s)
    print("\n  rewrote paper/main.tex (14 sites)")
    return 0

if __name__ == "__main__":
    sys.exit(main())
