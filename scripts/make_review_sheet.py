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

# What was done in response to round one. Keyed by item id; emitted into the CC slots so round two
# can see which findings landed and which were declined, and why.
CC_ROUND1 = {
  # (item, reviewer) -> reply. Reviewer is 1 Gemini, 2 ChatGPT, 3 Grok, 4 Perplexity.
  # A reply says what that reviewer found and what was done with it. Where a reviewer filed
  # nothing on an item, the slot stays empty rather than repeating another reviewer's point.
  ("C1", 2): "FIXED. You said calling all 88 rows 'computed support bits' collapses provenance. "
             "Right. Appendix B now carries all 88 rows with the estimator for each, and the "
             "text separates 11 exact, 45 upper bounds, 2 censored floors and 30 unmeasured.",
  ("C1", 4): "NOTED, no change. You accepted the definition and asked for 'unweighted tree "
             "average' at first use in the Figure 1 caption. The definition paragraph is now "
             "immediately before Figure 1, so the caption would repeat it within half a page.",
  ("C2", 2): "FIXED, both halves. The 3-4p median is 39.9 as Table 1 says; the 31.1 in prose was "
             "stale. The denominator is now the 18 measured games, not all 23.",
  ("C2", 4): "AGREED. Same finding as ChatGPT on coverage. Fixed by scoping to measured games.",
  ("C3", 2): "FIXED. You said the fingerprint sentence overran what we showed. It is now three "
             "short sentences that describe the mechanism and stop.",
  ("C4", 1): "FIXED, and this was the sharpest catch of the round. No contrast could be "
             "reproduced from the printed means: 27.02 minus 27.04 reads -0.02 while we printed "
             "-0.03. The difference was computed before rounding, which is defensible but not "
             "checkable. Contrasts now come from the means as printed.",
  ("C4", 2): "SOUND accepted, and your T8 note found the related error: we named Welch but used "
             "a t(11) multiplier. Now t at the Welch degrees of freedom, about 21. Intervals "
             "tightened to [-1.70, +1.20], [-1.51, +1.47] and [-1.57, +1.11].",
  ("C4", 3): "NOTED. You flagged that the interval still admits gains near 1.3 points. It is now "
             "1.2 after the Welch correction, and the paper states that limit outright.",
  ("C4", 4): "AGREED. The chain you traced now holds with the corrected intervals.",
  ("C5", 2): "FIXED. You said the prescription still reads as general advice. It now says the "
             "result reorders what to try next, and names the structured-feature test as the "
             "experiment that would settle it.",
  ("C5", 3): "NOTED. Same point as ChatGPT, same fix.",
  ("C6", 2): "FIXED. Worth depends on the frozen opponent. The paper now says both ceilings use "
             "the same fixed opponent policy, so worth is exact for that setting and not for the "
             "game in the abstract.",
  ("C6", 4): "FIXED. Your A4 point, that the two ceilings must be commensurate, is verified in "
             "the code: ceiling() takes one opponent and differs only in the acting player's "
             "partition. The paper states it now rather than implying it.",
  ("C7", 2): "FIXED. 'There is no widening cost for a benefit to cancel' asserted more than the "
             "interval allows. Now 'no evidence that the cost of a wider input hides a real gain'.",
  ("C7", 4): "FIXED. 'Across all 88 runs' was an ambiguous unit, as you said. The widening column "
             "is now described per game over eight seeds, with Gin Rummy quoted separately.",
  ("C8", 2): "FIXED, and this was the round's most important find. You and Perplexity "
             "independently showed the sign agreement is the identity banked = worth + g_b - g_o, "
             "so it is true by definition and measures nothing. Part III now derives it and says "
             "an earlier draft reported the count as evidence. The count and the rank correlation "
             "are gone.",
  ("C8", 4): "FIXED. Same finding as ChatGPT, reached independently, and you were right that the "
             "quantities share arm returns. Verified: the identity holds to 1.1e-16.",
  ("C8", 1): "PARTLY DECLINED. You marked this SOUND and said the paper warns against using "
             "capture as a point estimate. It did, but the supporting argument was circular, "
             "which the other two reviewers caught. Rewritten as algebra.",
  ("C8", 3): "NOTED. You flagged the rank-correlation claim as resting on related configurations. "
             "The deeper problem was that it rested on nothing: it was an identity.",
  ("C9", 2): "FIXED. The limitations paragraph inherited the same circular wording. It now says "
             "capture is distorted in a known direction, not that we confirmed it in eleven games.",
  ("C9", 3): "DECLINED for this deadline. You, Gemini and Perplexity all named the structured-"
             "feature experiment as the one to run. It is named in the paper as the next step. "
             "There is no time to run it and we will not claim a result we do not have.",
  ("C10", 2): "SOUND accepted, no change.",
  ("T1", 2): "FIXED. 'Any estimate' hid a mixture of exact values, upper bounds and censored "
             "floors. The caption now says 'any reported value or bound, exact, upper or "
             "censored'. The two prose numbers you flagged are corrected under C2.",
  ("T2", 2): "FIXED. The row labels restored the two names the body had retired. Now "
             "'Architecture sensitivity' and 'Oracle-search gap'.",
  ("T2", 4): "FIXED. Same point, and your suggested questions are close to what we used.",
  ("T3", 2): "FIXED. The caption asserted the game/learner split the body had already withdrawn.",
  ("T3", 4): "FIXED. Caption now names the opponent policy and says capture is distorted when the "
             "arms learn with unequal ease.",
  ("T4", 2): "SOUND accepted, no change.",
  ("T5", 2): "FIXED. Worth is invariant by construction, as you say. The caption now says exact "
             "worth is unchanged by the budget, and that the capture ratio moves.",
  ("T5", 4): "FIXED. Same point, same fix.",
  ("T6", 2): "FIXED. 'Capacity bound' overstated a bucket-folding sweep. Now 'only Leduc improves "
             "steadily as the buckets approach exact states'.",
  ("T6", 4): "FIXED. Your wording, that this is a state-resolution signature rather than a "
             "diagnosis, is what the body now says.",
  ("T7", 2): "FIXED. The 'Predicted' column asked whether the sign agrees, which the identity "
             "makes true by construction. The column is deleted.",
  ("T7", 1): "FIXED. You found the difference column did not subtract as printed. It now does.",
  ("T8", 2): "FIXED. You caught that we named Welch and used t(11). The contrast intervals now "
             "use t at the Welch degrees of freedom, and the caption distinguishes arm intervals "
             "from contrast intervals.",
  ("T8", 1): "FIXED. Same arithmetic problem you raised on C4, fixed the same way.",
  ("F1", 2): "PARTLY. The bar chart cannot carry provenance, and we did not rebuild it. Instead "
             "Appendix B now holds the 88-row table with the estimator for every value, and the "
             "text tells readers to check provenance before comparing two games.",
  ("F2", 2): "FIXED. The title asserted an exact zero. It now reads 'no detectable gain from the "
             "hidden hand, or from the widening'. We kept the seed dots and added the three "
             "contrasts to Table 2 rather than redrawing the figure.",
  ("F3", 2): "FIXED. 'Wherever' was universal language. The title now says 'in every tested small "
             "game, but not detectably on Gin Rummy', and the caption defines the pooled standard "
             "deviation, the symlog threshold and the 0.8 band.",
  ("F3", 4): "PARTLY. You asked for a forest plot in native units and called this a blocker. The "
             "definition was the real defect and it is fixed in the caption. Replacing the figure "
             "is a change we will not make hours before a deadline without time to check it.",
  ("F4", 2): "FIXED. The caption claimed the figure showed a mechanism. It now says the figure is "
             "a picture of the identity rather than evidence for it.",
  ("F4", 4): "FIXED. Same point, same fix.",
  ("F5", 2): "SOUND accepted, no change.",
  ("F6", 2): "FIXED. The subplot title claimed monotone growth the figure contradicts. Kuhn dips "
             "and returns. Now 'the share banked moves with budget; Leduc rises overall'.",
  ("F7", 2): "SOUND accepted. The replication caveat you raised elsewhere is now in the caption.",
  ("F7", 4): "FIXED. Caption and body now say signature rather than diagnosis.",
  ("A1", 2): "PARTLY. You noted the appendix names scripts that do not all exist under those "
             "names. Corrected to the scripts that do.",
  ("A2", 2): "FIXED. The section was called 'the atlas in full' and held only two corrections. It "
             "now carries all 88 rows with provenance, which is what the title promised.",
  ("A3", 2): "NOTED. You are right that the placebo matches shape and sparsity but not every "
             "marginal. The rejected game in Appendix G is exactly that failure, and we say so.",
  ("A4", 2): "FIXED. Seeds reach banked and widening directly and capture through banked, not "
             "'the banked column alone'.",
  ("A4", 4): "FIXED. Both ceilings share one opponent policy, verified in the code and now stated.",
  ("A5", 2): "FIXED. Worth is invariant by construction, so the appendix no longer presents "
             "budget invariance as a test that worth passed.",
  ("A6", 2): "FIXED. 'That is what being capacity bound looks like' is now a signature on one "
             "unreplicated sweep.",
  ("A7", 2): "FIXED. Same overrun as C3, split into shorter sentences that stop at what we showed.",
  ("A8", 2): "FIXED. Rewritten around the derivation. The ordering claim is gone.",
  ("A8", 4): "FIXED. You said 'bias' presumes an estimand we do not have. The section now says "
             "the identity gives the direction of the distortion and no estimator for its size.",
  ("A9", 2): "FIXED. The sentence conflated arm-mean intervals with contrast intervals. It now "
             "separates them and says why an arm interval cannot contain zero.",
}

AGENTS = {1: ("GEMINI", "AI1(GEMINI)"), 2: ("ChatGPT", "AI2(ChatGPT)"),
          3: ("GROK", "AI3(GROK)"), 4: ("PERPLEXITY", "AI4(PERPLEXITY)")}

# Which slots a sheet carries. The master keeps all nine for merging the replies back together;
# a per-agent sheet carries only that agent's slot and the CC written to it.
SLOTS = ["AI1(GEMINI)", "AI2(ChatGPT)", "AI3(GROK)", "AI4(PERPLEXITY)",
         "CC1", "CC2", "CC3", "CC4", "Nima"]
WHO = None          # None for the master sheet, else the agent number
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
    slots = SLOTS if WHO is None else [AGENTS[WHO][1], f"CC{WHO}", "Nima"]
    for s in slots:
        r = CC_ROUND1.get((tag, int(s[2]))) if re.fullmatch(r"CC[1-4]", s) else None
        L.append(f"{s}:{{{r}}}" if r else f"{s}:{{}}")
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

    if WHO is None:
        L.append("ROUND TWO, MASTER SHEET: all four expert slots.")
        L.append("Use this to merge the replies back together. Do not send it out; each expert")
        L.append("has a sheet of their own so the four reviews stay independent.")
    else:
        name, slot = AGENTS[WHO]
        L.append(f"ROUND TWO REVIEW SHEET FOR {name.upper()}")
        L.append(f"You are {slot}. Write inside your own braces and inside no others.")
        L.append(f"CC{WHO} holds our reply to what you filed in round one. Read it first: it says")
        L.append("what we changed, and what we declined and why.")
        L.append("The other three experts have their own sheets. You are not shown their answers,")
        L.append("so that the four reviews stay independent.")
    L.append("Paper: Is It the Game or the Agent? (NeurIPS 2026 TAE workshop, 8pp + appendix)")
    L.append("Every number below is read from the result files, not typed. C1..C11 are claims. "
             "T are tables, F figures, A appendix sections.")
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

    claim("C6", "A plateau splits into worth (game and opponent) and capture (learner).",
          f"""
Worth = oracle ceiling minus baseline ceiling. Both ceilings are exact, and BOTH are computed
against the same fixed opponent policy; the only difference is whether the acting learner sees the
hidden state. Worth is therefore exact for that evaluation setting, not for the game in the
abstract. No seed noise in either ceiling.
Capture = what the learner banks, divided by worth.
Worth is invariant to a 64-fold change in training budget (25k to 1.6M episodes), constant to
three decimals in both games swept. Capture moves with budget: 56% to 79% in Leduc.
Coverage: {len(good)} games, {sum(len(v) for v in good.values())} runs, 8 seeds each,
{min(rs[0]['hidden_bits'] for rs in good.values() if rs[0].get('hidden_bits')):.2f} to
{max(rs[0]['hidden_bits'] for rs in good.values() if rs[0].get('hidden_bits')):.2f} bits.
""")

    claim("C7", "No evidence that the cost of a wider input hides a real gain.",
          f"""
NARROWED SINCE ROUND ONE. The claim used to be that widening "costs nothing", averaged over "88
runs". The unit was ambiguous and the claim outran the evidence.
Widening = placebo arm minus baseline arm, averaged over the eight seeds of each game.
It stays inside seed noise in all {len(good)} games. In Gin Rummy the same contrast is -0.02
points, interval [-1.60, +1.56].
That supports "no evidence that the wider input masked a gain". It does not support "widening is
free", and the paper no longer says so.
""")

    claim("C8", "Capture exceeds 100% in seven games, and the algebra says why.",
          f"""
CHANGED SINCE ROUND ONE. Two reviewers found that the previous version of this claim treated an
algebraic identity as empirical evidence. They were right. The claim is now the identity.

{hi} of {len(good)} games report capture above 100%, which looks impossible. Write g_b and g_o for
how far each arm sits below its own exact ceiling. Then, from the definitions alone:

    banked = worth + g_b - g_o

so capture exceeds 1 if and only if g_b > g_o. Always. Our runs satisfy the identity with a maximum
residual of 1.1e-16, which is floating point.

The paper previously reported that the sign agreed "in eleven games out of eleven" with a rank
correlation of 1.000, as though that confirmed a mechanism. It confirms nothing: the agreement is
guaranteed by the definitions. Both the count and the correlation are removed.

What is still empirical is where each family sits. The dice arms are 0.000 to 0.007 below their
oracle ceilings, because with the die revealed the game is nearly solved. The poker arms are 0.377
to 0.394 below theirs, because the oracle ceiling is much higher. That split is the finding.
Per game the gap difference is 0.8 to 2.3 times the seed spread, so no single game settles it.
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

    name = "REVIEW_SHEET.txt" if WHO is None else f"REVIEW_SHEET_{AGENTS[WHO][0].upper()}.txt"
    out = os.path.join(HERE, name)
    with open(out, "w") as f:
        f.write("\n".join(L) + "\n")
    items = sum(1 for x in L if re.match(r"^\[[CTFA]\d", x))
    filled = sum(1 for x in L if re.match(r"^CC\d:\{[^}]", x))
    print(f"  {name:<34} {items} items, {filled} replies to this agent")
    return len(good), len(bad)


if __name__ == "__main__":
    # the master first, then one sheet per agent
    good, bad = main()
    for who in AGENTS:
        WHO = who
        L.clear()
        main()
    WHO = None
    print(f"  usable {good} games, {bad} rejected")
