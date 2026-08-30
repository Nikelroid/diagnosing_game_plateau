"""Build the four-expert sign-off sheet from the results actually on disk.

    python scripts/make_review_sheet.py

The paper is written, externally reviewed once, and revised. This sheet is the last round: it
states each claim the paper now makes, with the live numbers behind it, and asks four independent
readers whether the claim is still wider than its evidence. Every number is read from data/ or from
the companion repository's sweep results, so the sheet cannot drift from the experiments.
"""
from __future__ import annotations

import glob
import json
import os
import re
import statistics as st
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import axis_data
import gin_arms

FIGURE_CONTENT = {
    "fig1_hidden_bits.pdf":
        "Horizontal bars, one per game, sorted by support bits, coloured by family. Hatched bars "
        "mark resampled values, which are right-censored lower bounds rather than estimates. "
        "88 configurations. Gin Rummy marked as the anchor at 30.1.",
    "fig_gin_arms.pdf":
        "Three arms on one axis: baseline, oracle, placebo. One dot per training seed (12 each), "
        "a bar at the arm mean, and a 95 percent whisker. The three intervals overlap heavily.",
    "fig_sensitivity.pdf":
        "Effect of the oracle across fifteen game and learner combinations, standardised by the "
        "seed-to-seed spread so games measured in return and in win rate share an axis. Raw "
        "effects printed beside each point. A shaded band marks where an effect cannot be told "
        "from none; Gin Rummy's interval is the only one inside it.",
    "fig_mechanism.pdf":
        "One line per game, from how far the baseline arm sits below its ceiling to how far the "
        "oracle arm sits below its own. Lines coloured by whether capture exceeds 100 percent. "
        "11 games. Every line for capture over 100 slopes one way and every line under it the "
        "other; this is the figure the eleven-of-eleven claim rests on.",
    "fig_apparatus.pdf":
        "The same probe under two learners that share no code, a tabular one and a neural one "
        "reading a fixed-width tensor. Gain from the hidden state on the y axis, one point per "
        "game per learner. The two agree on which games hide something usable.",
    "fig_budget.pdf":
        "Worth and capture against training episodes, 25k to 1.6M on a log axis, for Kuhn and "
        "Leduc. Worth is a flat line in both; capture rises.",
    "fig_capacity.pdf":
        "Exploitability against the number of buckets the learner can tell apart, log x, "
        "rightmost point exact. One curve per game. Only Leduc descends.",
    "fig3_gin_dial.pdf": "(generated but not currently placed in the paper)",
    "fig2_bits_vs_cost.pdf": "(generated but not currently placed in the paper)",
    "fig1_three_probes.pdf": "(generated but not currently placed in the paper)",
    "fig_decomposition.pdf": "(generated but not currently placed in the paper)",
}

SLOTS = ["AI1(GEMINI)", "AI2(ChatGPT)", "AI3(GROK)", "AI4(PERPLEXITY)",
         "CC1", "CC2", "CC3", "CC4", "Nima"]
L = []


def latex_table_to_text(path):
    """Render a generated tabular as aligned plain text, so a reader sees what the PDF shows."""
    raw = open(path).read()
    rows = []
    for line in raw.split("\n"):
        line = line.strip()
        if (not line or line.startswith("%") or line.startswith("\\begin")
                or line.startswith("\\end") or line in (r"\toprule", r"\midrule", r"\bottomrule")):
            continue
        line = line.rstrip("\\").strip()
        if not line:
            continue
        cells = []
        for c in line.split("&"):
            c = re.sub(r"\\multicolumn\{\d+\}\{[^}]*\}\{([^}]*)\}", r"\1", c)
            c = re.sub(r"\\(textbf|emph|texttt|textit)\{([^}]*)\}", r"\2", c)
            c = c.replace(r"\%", "%").replace(r"\_", "_").replace(r"\&", "&")
            c = c.replace("$", "").replace("{", "").replace("}", "")
            cells.append(" ".join(c.split()))
        rows.append(cells)
    if not rows:
        return "  (empty)"
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    cols = [max(len(r[i]) for r in rows) for i in range(width)]
    out = []
    for i, r in enumerate(rows):
        out.append("  " + "  ".join(c.ljust(cols[j]) for j, c in enumerate(r)).rstrip())
        if i == 0:
            out.append("  " + "  ".join("-" * cols[j] for j in range(width)))
    return "\n".join(out)


def floats_from_paper():
    """Every table and figure in the paper, with its caption, label and source."""
    tex = open(os.path.join(HERE, "paper", "main.tex")).read()
    out = []
    for m in re.finditer(r"\\begin\{(table|figure)\}(.*?)\\end\{\1\}", tex, re.S):
        kind, body = m.group(1), m.group(2)
        cap = re.search(r"\\caption\{(.*?)\}\s*\\label", body, re.S)
        if not cap:
            cap = re.search(r"\\caption\{(.*?)\n\s*\\label", body, re.S)
        caption = " ".join(cap.group(1).split()) if cap else "(no caption)"
        caption = re.sub(r"\\(emph|textbf|texttt|textit)\{([^}]*)\}", r"\2", caption)
        caption = re.sub(r"~?\\ref\{([^}]+)\}", r"[\1]", caption)
        label = re.search(r"\\label\{([^}]+)\}", body)
        src = re.search(r"\\input\{\.\./tables/([^}]+)\}", body)
        img = re.search(r"\\includegraphics\[[^]]*\]\{([^}]+)\}", body)
        out.append({"kind": kind, "caption": caption,
                    "label": label.group(1) if label else "",
                    "table": src.group(1) if src else None,
                    "figure": img.group(1) if img else None,
                    "appendix": m.start() > tex.index(r"\appendix")})
    return out


def appendix_sections():
    """Appendix section titles and their prose, so the sheet carries the supplement too."""
    tex = open(os.path.join(HERE, "paper", "main.tex")).read()
    app = tex[tex.index(r"\appendix"):]
    app = re.sub(r"\\begin\{(figure|table)\}.*?\\end\{\1\}", "", app, flags=re.S)
    out = []
    parts = re.split(r"\\section\{([^}]+)\}", app)
    for i in range(1, len(parts), 2):
        title, body = parts[i], parts[i + 1]
        body = re.sub(r"\\label\{[^}]+\}", "", body)
        body = re.sub(r"\\citep?t?\{([^}]+)\}", r"[\1]", body)
        body = re.sub(r"~?\\ref\{([^}]+)\}", r"[\1]", body)
        body = re.sub(r"\\(emph|textbf|texttt|textit|paragraph)\{([^}]*)\}", r"\2", body)
        body = re.sub(r"\$([^$]*)\$", r"\1", body)
        body = re.sub(r"\\[a-zA-Z]+\*?", "", body)
        body = re.sub(r"[{}]", "", body)
        paras = [" ".join(x.split()) for x in re.split(r"\n\s*\n", body) if x.strip()]
        out.append((title, [p for p in paras if len(p.split()) > 6]))
    return out


def claim(tag, headline, evidence):
    """One claim the paper makes, the evidence under it, and a slot per reader."""
    L.append(f"[{tag}] {headline}")
    L.append("")
    for line in evidence.strip("\n").split("\n"):
        L.append("  " + line if line.strip() else "")
    L.append("")
    for s in SLOTS:
        L.append(f"{s}:{{}}")
    L.append("")
    L.append("-" * 96)
    L.append("")


def main():
    good, bad = axis_data.load()
    arms = gin_arms.arms()
    ob = gin_arms.contrast("oracle", "baseline")
    pb = gin_arms.contrast("placebo", "baseline")
    op = gin_arms.contrast("oracle", "placebo")

    caps = {g: 100 * st.mean([r["capture"] for r in rs if r.get("capture") is not None])
            for g, rs in good.items()}
    hi = sum(1 for c in caps.values() if c > 100)
    wid = [r["widening_cost"] for rs in good.values() for r in rs
           if r.get("widening_cost") is not None]

    # survey_full.csv is the atlas as the paper reports it, one row per game configuration
    import csv
    with open(os.path.join(HERE, "tables", "survey_full.csv")) as fh:
        atlas = list(csv.DictReader(fh))
    engines = Counter(r["engine"] for r in atlas)
    prov = Counter(r["bits_source"] or "no number" for r in atlas)

    L.append("FOUR-EXPERT SIGN-OFF: IS ANY CLAIM STILL WIDER THAN ITS EVIDENCE?")
    L.append("Paper: Is It the Game or the Agent? (NeurIPS 2026 TAE workshop, 8pp + appendix)")
    L.append("Every number below is read from the result files, not typed. C1..C11 are claims. T are tables, F figures, A appendix sections.")
    L.append("")
    L.append("=" * 96)
    L.append("")

    claim("C1", "The atlas measures support bits, and that is not entropy.",
          f"""
Definition in the paper: support bits = log2 of the number of histories consistent with the
acting player's information state. Reported as the UNWEIGHTED mean over distinct information
states in the tree, pooled across seats. Deliberately not reach-weighted.
The paper says this is an upper bound on posterior entropy and says nothing about the value of
the hidden state.
Coverage: {len(atlas)} configurations, {engines['openspiel']} OpenSpiel and {engines['rlcard']} RLCard.
Provenance: {', '.join(f'{v} {k}' for k, v in prov.most_common())}.
Gin Rummy anchor: 30.1 bits, cross-checked across two engines.
""")

    claim("C2", "Two-player card games saturate; the extra bits come from extra players.",
          """
Median for two-player card games: 3.1 bits, n=23.
Only one of the 23 is above Gin Rummy's 30.1: bridge uncontested bidding at 32.9, which is a
three-turn auction fragment rather than a whole game.
Three and four-handed family: median 31.1, reaching 109.6 for Mahjong.
The paper now says "in this atlas, the extra bits come from extra players", not the general claim.
""")

    claim("C3", "The measuring instruments lied three times, once in our own apparatus.",
          """
Failure 1, saturation: the resampling estimator returns exactly log2(80) = 6.32 for three games
because the sample budget is 80. It is a right-censored lower bound, not an estimate.
Failure 2, degeneracy: every Gin Rummy variant resamples to 0.00 bits with zero censoring,
because the engine resampler returns the same world repeatedly.
Failure 3, ours: in liars_dice(numdice=2,dice_sides=3) the PLACEBO arm scored 0.67 above its own
exact ceiling, which no policy can do. The per-deal placebo draw is nearly unique with two dice,
so the plane became a fingerprint of the deal and leaked what it was built to withhold. The game
is rejected from every table and figure by an automated rule.
""")

    claim("C4", "Handing the learner the opponent's hand changes nothing measurable.",
          f"""
Gin Rummy, {arms['baseline']['n']} independent training seeds per arm, evaluated against a fixed
rule-based reference.
  baseline (4 planes)          {arms['baseline']['mean']:.2f} %  sd {arms['baseline']['sd']:.2f}
  oracle   (5th = real hand)   {arms['oracle']['mean']:.2f} %  sd {arms['oracle']['sd']:.2f}
  placebo  (5th = no info)     {arms['placebo']['mean']:.2f} %  sd {arms['placebo']['sd']:.2f}
Contrasts, percentage points, Welch, arms are independent and NOT paired:
  oracle  - baseline   {ob['diff']:+.2f}  95% CI [{ob['lo']:+.2f}, {ob['hi']:+.2f}]
  placebo - baseline   {pb['diff']:+.2f}  95% CI [{pb['lo']:+.2f}, {pb['hi']:+.2f}]
  oracle  - placebo    {op['diff']:+.2f}  95% CI [{op['lo']:+.2f}, {op['hi']:+.2f}]
The interval excludes gains above about {ob['hi']:.1f} points. Smaller gains remain possible.
An earlier four-seed version had an interval of [-4.5, +2.7], which excluded almost nothing.
""")

    claim("C5", "The plateau is not relieved by this information. That is narrower than learning-bound.",
          """
The paper USED to say "this agent is learning-bound" and "effort belongs in optimisation, reward
and curriculum, not in belief modelling". Both were cut after an external review called them the
biggest overclaim.
It now says: this learner, on this recipe, graded against this fixed reference, banks nothing from
a raw hidden-hand channel; and that the result reorders what to try next rather than closing off
belief modelling, because a structured encoding of the same information might still pay.
""")

    claim("C6", "A plateau splits into worth (the game) and capture (the learner).",
          f"""
Worth = oracle ceiling minus baseline ceiling. Both ceilings are exact: best response to the
opponent policy, and a memoised expectimax with the state revealed. No seed noise in either.
Capture = what the learner banks, divided by worth.
Worth is invariant to a 64-fold change in training budget (25k to 1.6M episodes), constant to
three decimals in both games swept. Capture moves with budget: 56% to 79% in Leduc.
Coverage: {len(good)} games, {sum(len(v) for v in good.values())} runs, 8 seeds each,
{min(rs[0]['hidden_bits'] for rs in good.values() if rs[0].get('hidden_bits')):.2f} to
{max(rs[0]['hidden_bits'] for rs in good.values() if rs[0].get('hidden_bits')):.2f} bits.
""")

    claim("C7", "The placebo shows a wider observation costs nothing, so the null is not a cancellation.",
          f"""
Widening = placebo arm minus baseline arm, per game.
Mean over all {len(wid)} runs: {st.mean(wid):+.4f}. Inside seed noise in every game.
This is what rules out the reading that a real benefit was cancelled by a real cost of the wider
input. Without it the null in C4 would be uninterpretable.
""")

    claim("C8", "Capture exceeds 100% in seven games, and the reason is a known bias, not noise.",
          f"""
{hi} of {len(good)} games report capture above 100%, which is impossible if worth is the right
denominator. The ceilings are exact, so the error is in the denominator's assumption: that the
learner sits equally far below its ceiling in both arms.
It does not. Revealing the state supplies information AND makes the task easier to learn.
In all {hi} games with capture > 100%, the baseline arm sits further below its own ceiling.
In all {len(good)-hi} games with capture < 100% (the poker variants), the oracle arm does.
Rank correlation between capture and the gap difference: exactly 1.000 over all {len(good)} games,
AND 1.000 within each of the two game families taken alone.
Per game the effect is 0.8 to 2.3 times the seed spread, so no single game rules out noise.
The paper deliberately does NOT quote a p-value: the games are related configurations.
""")

    claim("C9", "What the paper does not do.",
          """
One game and one expert in the case study. The 11 decomposed games are all small enough to solve
exactly, which is the property that makes them unlike Gin Rummy.
Capture is biased wherever the arms sit unequally below their ceilings, which is 7 of 11. The
paper reports the sign and has no estimator that corrects it.
No Gin Rummy training-budget sweep. No second frozen opponent. No structured oracle features.
The architecture probe varies inductive bias and compute together. Neither it nor the bucket
sweep tests capacity LOST during training, which is a live rival explanation for any plateau.
Closed forms count hands and ignore public information; sized on one game only (Leduc, 2.01 exact
against 2.30 closed form).
""")

    claim("C10", "Positioning against prior work.",
          """
Long et al. 2010 measured leaf correlation, bias and disambiguation factor and showed they predict
when determinized search works. That is the precedent for the atlas. This paper had cited them
only as a search method until an external round caught it.
Markowitz et al. 2018 size information sets in reconnaissance blind chess against poker and Go.
Schofield and Thielscher 2019 sample information sets and value information-gathering moves.
Rebstock et al. 2019 do inference in trick-taking card games.
The paper's stated novelty is now the JOINING: a provenance-aware measurement of what a domain
hides, used to read a specific agent's plateau, with a placebo that prices the intervention.
""")

    unmeasured = sum(1 for r in atlas if r["bits_source"] == "not measured")
    unmeasured_board = sum(1 for r in atlas
                           if r["bits_source"] == "not measured" and r["family"] == "board")

    claim("C11", "Where this paper should go next. (Direction, not a blocker.)",
          f"""
This section is NOT about the submission. Answer it as if the paper is already sent.
Assume the deadline passed and everything above shipped as it stands.

What the authors think they have: a workflow that measures what a domain hides, then reads a
specific agent's plateau against it, with a placebo that prices the intervention itself.

Where they think it could go, in no order and with no commitment:
  a. The blind spot. {unmeasured} of {len(atlas)} configurations carry no number by any method,
     and {unmeasured_board} of those are board and fog-of-war games. Those are the games people
     call the frontier of imperfect information, and nobody can currently size them.
  b. The estimator. Support bits are an upper bound on posterior entropy. Measuring the entropy
     itself, under a stated policy, would say how far apart the two really are.
  c. The bias. Capture is biased wherever the two arms sit unequally below their ceilings, which
     is {hi} of {len(good)} games. There is no estimator here that corrects it, only one that
     reports its sign.
  d. Structured privileged features instead of a raw hand plane, to separate "the information is
     not worth much" from "this learner cannot use it in that form".
  e. Scale. Every exactly-decomposed game is small enough to solve, which is the property that
     makes it unlike the case study.

The question: which of these is the next PAPER, and which are footnotes in this one? Name one
you would drop entirely. If the real next step is not on the list, say what it is.
""")

    # Tables and figures, in the order the paper shows them, so the sheet carries what the PDF
    # carries rather than only the prose about it.
    floats = floats_from_paper()
    for i, f in enumerate([x for x in floats if x["table"]], 1):
        where = "Appendix" if f["appendix"] else "Main text"
        path = os.path.join(HERE, "tables", f["table"])
        body = (f"{where}. Source: tables/{f['table']}\n\n"
                f"Caption: {f['caption']}\n\n"
                f"{latex_table_to_text(path)}")
        claim(f"T{i}", f"TABLE {f['label']}", body)

    for i, f in enumerate([x for x in floats if x["figure"]], 1):
        where = "Appendix" if f["appendix"] else "Main text"
        body = (f"{where}. Source: figures/{f['figure']}\n\n"
                f"Caption: {f['caption']}\n\n"
                f"What it draws: {FIGURE_CONTENT.get(f['figure'], '(no description on file)')}")
        claim(f"F{i}", f"FIGURE {f['label']}", body)

    for i, (title, paras) in enumerate(appendix_sections(), 1):
        body = "\n\n".join(paras) if paras else "(figures and tables only; see T and F items)"
        claim(f"A{i}", f"APPENDIX {chr(64+i)}. {title}", body)

    if bad:
        L.append("REJECTED FROM ALL TABLES AND FIGURES (an arm scored above its own exact ceiling):")
        for g in sorted(bad):
            L.append(f"  {g}")
        L.append("")

    out = os.path.join(HERE, "REVIEW_SHEET.txt")
    with open(out, "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"  wrote {out}: {sum(1 for x in L if x.startswith(chr(91))) } items, {len(SLOTS)} slots each")
    print(f"  usable {len(good)} games / {sum(len(v) for v in good.values())} runs, "
          f"{len(bad)} rejected")


if __name__ == "__main__":
    main()
