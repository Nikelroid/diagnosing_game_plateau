# Research summary — for CV, resume and cover-letter use

**Nima Kelidari** · written 30 August 2026

This file is the durable record of one research project: what it asked, what it built, what it
found, and which skills it demonstrates. Every number below is reproduced by a script in this
archive; none is estimated.

---

## 1. The work in one paragraph

When a reinforcement learning agent stops improving, practitioners routinely blame the game for
hiding too much. That is an explanation nobody measures. I built an evaluation protocol that
separates three quantities a win rate reports as one: how much a game hides, how much that hiding
is worth against a stated opponent, and how much of it a learner actually banks. The protocol is
calibrated against exact solutions on eleven games where the answer is computable, then applied
unchanged to a game far too large to solve. Applied to Gin Rummy, it rules out gains above about
0.3 percentage points from giving the learner the opponent's hand, so for that agent and reference
hidden information is not the missing ingredient. The protocol also detects when its own
instruments fail, which it did three times, once in my own experimental control.

---

## 2. Publication record

| | |
|---|---|
| **Paper** | *Is It the Game or the Agent? Measuring What Games Hide, Diagnosing a Plateau* |
| **Venue** | IAEval / TAE workshop, "Can We Trust AI Evaluation?", NeurIPS 2026 |
| **Status** | Submitted 30 August 2026, submission #253. Non-archival, double-blind. |
| **Authors** | Nima Kelidari, Mohammadsaeed Haghi, Mahdi Salmani |
| **Format** | 8 content pages plus a 10-page appendix |
| **Code** | https://github.com/Nikelroid/diagnosing_game_plateau |

**Prior related publication**

| | |
|---|---|
| **Paper** | *A Gold-Standard Study of What Makes a Lightweight Game-Playing Agent Strong* |
| **Venue** | AIIDE 2026 (AAAI Conference on AI and Interactive Digital Entertainment) |
| **Status** | **Accepted, poster.** Belo Horizonte, November 2026. arXiv:2607.06854 |

**Open-source contribution**

Merged upstream fix to **PettingZoo** (Farama Foundation): seed and payoff correctness in the
Gin Rummy environment. PR #1335, issue #1312, merged to `main`.

---

## 3. What the protocol is

**Support bits.** At any decision, count the complete game histories still consistent with what
the acting player has seen, and take the base-two logarithm: `b(o) = log2 |I(o)|`. This upper-bounds
the player's posterior uncertainty, `H(Z | O = o) <= b(o)`, with equality only under a uniform
posterior. It is deliberately *not* a measure of difficulty, and the paper is careful never to use
it as one.

**Worth.** For a fixed opponent policy, `worth = V*(O, Z) - V*(O)`: the best return achievable with
the hidden state minus the best achievable without it. A fixed-policy sequential analogue of the
expected value of perfect information. Computed exactly, not estimated, wherever an optimum exists.

**Banked.** The observed difference between two learners trained identically except that one
receives the hidden state. `capture = banked / worth`.

**The three-arm design.** Baseline, oracle (hidden state appended to the observation), and a
**placebo** arm carrying an addition of identical shape and sparsity but no information. The placebo
is a negative control on the observation channel: without it, "the oracle plane changed nothing" and
"the information's benefit exactly cancelled the cost of a wider input" are the same measurement and
support opposite conclusions.

---

## 4. Results

**The atlas.** 88 game configurations across two engines (79 OpenSpiel, 9 RLCard), every value
tagged with the method that produced it: exact enumeration, a hand-only closed form that is an upper
bound, or censored resampling.

**Three silent instrument failures, all caught by an independent bound rather than by a number
looking wrong:**

1. *Saturation.* Hearts, Euchre and contract bridge each returned exactly 6.32 bits, which is
   log2(80), the sample count, with every point censored. The closed form places Hearts 49.9 bits
   higher.
2. *Degeneracy.* Every Gin Rummy configuration returned 0.00 bits with zero censoring, because the
   engine's resampler returns the same world on every draw. A survey trusting that column would
   place a 30-bit game beside Connect Four.
3. *Our own control.* In Liar's dice with two dice per player, the placebo arm scored 0.738 against
   an exact ceiling of 0.062, which no policy can exceed. The fake plane was drawn once per deal from
   a space where nearly every draw is unique, so it became a fingerprint of the deal and leaked what
   it existed to withhold. The game was rejected rather than repaired after the fact.

**Calibration.** A fixed screen defines the candidate games before any result is examined:
two-player, zero sum, sequential, imperfect information, and small enough for counterfactual regret
minimisation. Five base games survive; parameter sweeps return sixteen solvable configurations;
twelve were run; the automatic ceiling check rejected one; eleven calibrate the instrument at eight
seeds each. The probe detects the hidden state in every one.

**The large case study — Gin Rummy, 24 seeds per arm, 3M steps each:**

| arm | mean win rate vs reference | sd |
|---|---|---|
| baseline | 27.03 | 1.98 |
| oracle | 26.37 | 1.36 |
| placebo | 26.65 | 1.69 |

| contrast | difference | 95% interval |
|---|---|---|
| oracle − baseline | −0.66 | [−1.65, +0.33] |
| placebo − baseline | −0.38 | [−1.45, +0.69] |
| oracle − placebo | −0.28 | [−1.17, +0.61] |

All three intervals contain zero: no detectable improvement or deterioration. **Twenty-four seeds
rule out gains above about 0.3 percentage points**, roughly one seventieth of the distance from this
agent to the parity its reference defines. An earlier four-seed version gave [−4.5, +2.7] and
excluded almost nothing, which is itself the argument for seed counts.

---

## 5. Scale and infrastructure

- **~520 CPU-hours** of training and solving: 461 hours across 188 curriculum runs and ~62 hours
  across the axis, neural, budget and capacity sweeps.
- **SLURM on a shared HPC cluster**, run as job arrays up to 36 tasks in parallel.
- **Stack:** Python, PyTorch, stable-baselines3 (PPO), OpenSpiel (CFR, best-response,
  expectimax), RLCard, PettingZoo, NumPy, SciPy, Matplotlib, Weights & Biases.
- **Reproducibility by construction:** no number in any table or figure is typed by hand. Every
  value flows from a JSON run record through a generator script into the paper, and
  `scripts/propagate_seeds.py` rewrites every prose figure from one source so the text cannot drift
  from the data.

---

## 6. Skills this project demonstrates

**Experimental design.** Built a three-arm design with a matched negative control, and correctly
diagnosed why the control failed on one game (cardinality, not shape) rather than quietly dropping
the result.

**Statistical discipline.** Welch intervals on independent training runs; reported a bound rather
than a point estimate; went from 4 to 12 to 24 seeds specifically because the earlier intervals
excluded almost nothing; refused to claim causation from intervals containing zero.

**Measurement validity.** Treated the evaluation itself as an instrument with failure modes, a
provenance record, and independent checks that can contradict it. Caught three failures, including
one of my own making, and reported it.

**Intellectual honesty under pressure.** Proved that no third calibration game family exists rather
than conceding the limitation vaguely: the only non-poker, non-dice candidate the screen returns is
solved outright, so its worth is exactly zero. Presented `banked = worth + g_b - g_o` as an algebraic
identity and explicitly refused to read its sign agreement across eleven games as evidence.

**HPC engineering.** SLURM job arrays, diagnosing a hung job as a blocked network call in the
experiment-tracking client rather than a training fault, and isolating per-run state on scratch.

**Open source.** A merged upstream correctness fix to a widely used RL environment library.

---

## 7. Résumé lines you can lift directly

> Designed and ran a three-arm evaluation protocol that separates how much a game hides from how
> much a learner gains, calibrated against exact game-theoretic solutions on eleven games and
> applied to a game too large to solve; bounded the effect of privileged information to under 0.4
> percentage points across 24 independent training runs per arm.

> Built a provenance-tracked atlas of hidden information across 88 game configurations from two RL
> engines, whose independent consistency checks exposed three silent measurement failures, including
> one in the study's own experimental control.

> ~520 CPU-hours of SLURM-orchestrated training across 188 runs, with an end-to-end pipeline in
> which no number in the paper is entered by hand.

> Merged an upstream correctness fix (seeding and payoffs) to the Farama Foundation's PettingZoo
> Gin Rummy environment.

---

## 8. Cover-letter framing

The honest and most distinctive angle is **not** "I trained an agent." It is:

> I care whether an evaluation measures what it claims. In this project the headline result is a
> bounded negative: giving the agent the information everyone assumed it needed changed nothing
> measurable. Getting to a null I could defend meant building the instrument first, calibrating it
> where the true answer was computable, and adding a control that would tell me when the instrument
> was lying. It told me three times, once about my own control, and I reported that rather than
> quietly fixing it.

That story travels well to any role where the cost of a wrong measurement is high: evaluation and
benchmarking, model assessment, applied research, or safety and reliability work. It is also honest
about scope: one learner, one reference, and a calibration set of two game families.

---

## 9. Known limitations, stated plainly

Say these before a reviewer or interviewer does; they are already in the paper.

- The large-case result covers **one learner, one training recipe and one fixed reference**.
- The eleven calibration games are **two families** (seven Liar's dice variants, four poker), because
  those are what an exact-solvability screen returns. I verified no third family exists under that
  constraint.
- Support bits describe the **game**, not the difficulty of learning it. The claim is that the two
  are routinely conflated, not that they are unrelated.
- The oracle-search probe is **confounded with strategy fusion** and is reported as a bounded proxy,
  not as a value of information.
- The interval leaves gains of about 0.3 points or less unresolved. The result is a bound, not a
  proof of zero.
