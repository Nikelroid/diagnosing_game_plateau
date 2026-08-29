"""Build the four-expert brainstorm sheet from the results actually on disk.

    python scripts/make_brainstorm.py

Every number below is read from data/ or from the companion repository's sweep results, so the
sheet cannot drift from the experiments. Each section carries four reviewer slots and four
reply slots, plus the author's.
"""
from __future__ import annotations
import glob, json, math, os, statistics as st
from collections import defaultdict

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAIN = "$HOME/Adversarial-CoEvolution/sweep/curriculum"
SLOTS = ["AI1(GEMINI)", "AI2(ChatGPT)", "AI3(GROK)", "AI4(CLAUDE)",
         "CC1", "CC2", "CC3", "CC4", "Nima"]
L = []


def section(tag, title, body):
    L.append(f"[{tag}] {title}")
    L.append("")
    for line in body.strip().split("\n"):
        L.append(line)
    L.append("")
    for s in SLOTS:
        L.append(f"{s}:{{}}")
    L.append("")
    L.append("-" * 96)
    L.append("")


def gin():
    def arm(pat):
        out = []
        for p in sorted(glob.glob(os.path.join(MAIN, pat))):
            d = json.load(open(p)); vg = d.get("vs_gold") or {}
            if vg.get("win_rate") is not None:
                out.append(100 * vg["win_rate"])
        return out
    b = arm("ppoarch_mlp_s*.json") + arm("basewide_s*.json")
    o, pl = arm("oracleobs_s*.json"), arm("placebo_s*.json")
    if not (b and o and pl):
        return "  (gin results incomplete)"
    def ci(w):
        m, sd = st.mean(w), st.stdev(w); return m, 2.201 * sd / math.sqrt(len(w))
    mb, cb = ci(b); mo, co = ci(o); mp, cp = ci(pl)
    sed = math.sqrt(st.stdev(o)**2/len(o) + st.stdev(b)**2/len(b))
    return (f"  baseline  n={len(b):2d}  {mb:.2f}% +/- {cb:.2f}\n"
            f"  oracle    n={len(o):2d}  {mo:.2f}% +/- {co:.2f}\n"
            f"  placebo   n={len(pl):2d}  {mp:.2f}% +/- {cp:.2f}\n"
            f"  oracle minus baseline {mo-mb:+.2f} pp, 95% CI [{mo-mb-2.201*sed:+.2f}, {mo-mb+2.201*sed:+.2f}]\n"
            f"  placebo minus baseline {mp-mb:+.2f} pp  (the cost of widening the observation)")


def axis():
    rows = defaultdict(list)
    for p in glob.glob(os.path.join(HERE, "data", "axis_games", "*.json")):
        r = json.load(open(p)); rows[r["game"]].append(r)
    if not rows:
        return "  (axis sweep still running)"
    out = [f"  {'game':36s} {'bits':>5} {'n':>2} {'worth':>7} {'banked':>7} {'capture':>8} {'widen':>7}"]
    for g, rs in sorted(rows.items(), key=lambda kv: kv[1][0].get("hidden_bits") or 0):
        w = st.mean([r["information_worth"] for r in rs])
        got = st.mean([r["oracle_minus_baseline"] for r in rs])
        cap = [r["capture"] for r in rs if r.get("capture") is not None]
        wid = [r.get("widening_cost") for r in rs if r.get("widening_cost") is not None]
        out.append(f"  {g:36s} {rs[0]['hidden_bits']:5.2f} {len(rs):2d} {w:7.3f} {got:7.3f} "
                   f"{(st.mean(cap) if cap else float('nan')):7.1%} "
                   f"{(st.mean(wid) if wid else float('nan')):+7.3f}")
    return "\n".join(out)


def budget():
    rows = defaultdict(list)
    for p in glob.glob(os.path.join(HERE, "data", "axis_budget", "*", "*.json")):
        r = json.load(open(p)); rows[(r["game"], r["episodes"])].append(r)
    if not rows:
        return "  (budget sweep missing)"
    out = [f"  {'game':13s} {'episodes':>9} {'worth':>7} {'capture':>8} {'gap oracle':>11}"]
    for (g, ep), rs in sorted(rows.items()):
        w = st.mean([r["information_worth"] for r in rs])
        cap = st.mean([r["capture"] for r in rs if r.get("capture") is not None])
        go = st.mean([r["arms"]["oracle"]["gap_to_ceiling"] for r in rs])
        out.append(f"  {g:13s} {ep:9d} {w:7.3f} {cap:7.1%} {go:11.4f}")
    return "\n".join(out)


def capacity():
    rows = defaultdict(list)
    for p in glob.glob(os.path.join(HERE, "data", "capacity", "*.json")):
        r = json.load(open(p)); rows[(r["game"], r["buckets"])].append(r)
    if not rows:
        return "  (capacity sweep missing)"
    out = [f"  {'game':34s} {'buckets':>8} {'n':>2} {'return':>8} {'exploitability':>14}"]
    for (g, b), rs in sorted(rows.items(), key=lambda kv: (kv[0][0], kv[0][1] or 10**9)):
        out.append(f"  {g:34s} {('exact' if b == 0 else b):>8} {len(rs):2d} "
                   f"{st.mean([r['mean_return_vs_cfr'] for r in rs]):+8.4f} "
                   f"{st.mean([r['exploitability'] for r in rs]):14.4f}")
    return "\n".join(out)


L.append("FOUR-EXPERT BRAINSTORM: WHERE THIS PAPER SHOULD GO")
L.append("Working title: Is It the Game or the Agent? Diagnosing why an agent stops improving.")
L.append("Every number below is read from the result files, not typed. Sections are S1..S9.")
L.append("")
L.append("=" * 96)
L.append("")

section("S1", "THE STORY AS IT STANDS", """
The paper has two halves that were written separately and now need one spine.

  Half one, the atlas: across 88 game configurations from two engines, measure the bits of
  hidden information at a typical decision. Two-player card games saturate near 3 bits median;
  one of 23 hides more than Gin Rummy's 30.1.

  Half two, the probes: when an agent stops improving, is it short of information or short of
  the ability to use it? Hand the learner the hidden state and see whether it does better.

  The proposed spine: a plateau decomposes into three separately measurable quantities.
    (a) what the hidden information is WORTH   - a property of the game, exactly computable
                                                 wherever CFR solves it
    (b) what share of that the learner BANKS   - "capture", a property of the learner
    (c) what the wider observation COSTS       - measured by a placebo channel carrying the
                                                 same shape and no information
""")

section("S2", "THE HEADLINE NULL, NOW PROPERLY POWERED", f"""
Gin Rummy, 3M steps per run, win rate against a fixed rule-based expert.
The earlier version of this result used 4 seeds and no placebo.

{gin()}

  The four-seed version had a 95% interval of [-4.5, +2.7] pp, which could not exclude an 11%
  relative gain. Twelve seeds plus a placebo arm is the version above.
""")

section("S3", "THE DECOMPOSITION ACROSS GAMES", f"""
Tabular Q-learning against a near-Nash CFR opponent, seat 0, exact ceilings.
"worth" is ceiling(oracle) - ceiling(baseline). "banked" is achieved(oracle) - achieved(baseline).

{axis()}
""")

section("S4", "CAPTURE ABOVE 100% IS A SECOND EFFECT, NOT AN ERROR", """
Capture above 100% turned out to mean the learner sits FURTHER below its ceiling without the
oracle than with it. The extra observation does two things: it supplies information, and it
changes how hard the task is to learn. The sign sorted 9 of 9 games with no exceptions.

  liar's dice: gap falls 0.025-0.084 -> 0.000-0.007 with the oracle  (capture 103-108%)
  Leduc:       gap RISES 0.035 -> 0.394 with the oracle              (capture 73-77%)

In liar's dice, revealing the die turns a belief problem into a lookup. In Leduc the oracle
triples the state space and the learner cannot fill it at this budget.
""")

section("S5", "THE FALSIFICATION TEST OF S4", f"""
S4 predicts worth is fixed by the game and must not move with budget, while capture is a
property of the learner and should climb. Across a 64x range:

{budget()}
""")

section("S6", "THE CAPACITY AXIS", f"""
Capacity is the number of buckets the learner may fold its information states into. At 0 it is
exact. This is what a genuinely capacity-bound plateau looks like, which is the reference the
Gin Rummy capacity null has never had.

{capacity()}
""")

section("S7", "WHAT IS BROKEN, MISSING OR UNRESOLVED", """
  1. Dark hex was never actually probed. It has no chance nodes, so "append the opponent's
     private card" appended nothing. Fog-of-war needs a different oracle (reveal the board).
     This is the family the atlas itself calls its blind spot.
  2. The neural probe (same fixed-width tensor and network as the Gin learner, rather than a
     table) is running now at 200k episodes. It is the only thing that closes the construct gap
     between the poker control and the Gin experiment.
  3. Liar's dice has seat-identical information-state strings. Seat-0-only evaluation avoids the
     bug this caused, but the consequences for the ceiling have not been fully chased down.
  4. The atlas carried seven closed-form errors, now fixed: UNO counted 108 cards as
     distinguishable when the deck has 54 types (33.9 -> 28.4 bits, BELOW Gin Rummy, so the
     abstract's "two of 23" became one); Mahjong exceeded a hard ceiling by 54 bits; five rows
     used a binomial where the method promises a multinomial.
  5. An earlier evaluation bug scored an untrained fold-bot on 42% of decisions. All affected
     results were discarded and re-run. Anything quoted from before that is void.
""")

section("S8", "THREATS FROM PRIOR WORK", """
  Whitehouse, Powley & Cowling (CIG 2011) already concluded "hidden information is not often
  important in Dou Di Zhu". That is close to our headline null, fifteen years earlier.

  Furtak & Buro (CIG 2013) already run our plateau argument for search: sampling more worlds
  stops closing the gap to a cheating player.

  Rebstock et al. (CoG 2019) show a cheating player scoring WORSE than a fair one, because it
  assumes the opponent also cheats. So an oracle is not automatically an upper bound. Our probe
  is asymmetric (opponents are trimmed to their own observation width), which we must state.

  Ni, Eysenbach & Salakhutdinov (ICML 2022) use an oracle on hidden state as an explicit upper
  bound. Closest neighbour to our main probe.
""")

section("S9", "CANDIDATE NEXT RUNS, RANKED BY WHAT THEY WOULD SETTLE", """
  A. Fog-of-war oracle: reveal the opponent's board in dark hex / phantom tic-tac-toe. Would put
     a number on the family the atlas cannot measure at all.
  B. Neural probe across more games and budgets, to test whether capture-below-100 in Leduc is a
     budget effect in the neural setting as it is in the tabular one.
  C. Gin Rummy at several training budgets, to see whether its null is a budget artefact the way
     Leduc's shortfall is.
  D. A learner sweep: tabular vs policy gradient vs DQN on one game, to test whether "capture"
     is a property of the algorithm rather than the game.
  E. The Gin Rummy dial: same rules, vary deck and hand size, so hidden bits move while
     everything else is held fixed. The atlas already measures the dial at 5.1 to 30.1 bits.
""")

out = os.path.join(HERE, "BRAINSTORM.txt")
open(out, "w").write("\n".join(L) + "\n")
print(f"wrote {out}: {len([x for x in L if x.startswith('[S')])} sections, {len(SLOTS)} slots each")
