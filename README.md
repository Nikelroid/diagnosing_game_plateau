# Is it the game or the agent?

Measuring what a game hides, and diagnosing why an agent stops improving. One repository behind one
paper: an **atlas** of 88 imperfect-information games, and three **probes** that turn a plateau
against a fixed reference into a diagnosis.

```
python scripts/run_survey.py         # measure every game        -> data/*.json
python scripts/atlas_tables.py       # rankings + family summary -> tables/
python scripts/atlas_figures.py      # -> figures/fig1..fig3
python scripts/verify_oracle_plane.py  # prove the oracle observation is correct (run before training)
python scripts/train_oracle.py --seed 0    # the baseline recipe, one extra observation plane
python scripts/collect_probes.py     # gather every probe number -> data/probes.json
python scripts/probe_tables.py       # -> tables/probes.tex
python scripts/probe_figures.py      # -> figures/fig1_three_probes
cd paper && pdflatex main.tex        # the paper; needs `module load texlive/2026` on CARC
```

## Part 1, the atlas

Hidden information at a decision is `log2` of the number of worlds consistent with what the player
to move knows. 88 games, 58 with a number, from OpenSpiel (79) and RLCard (9).

| Family | Games | Measured | Median bits |
|---|---|---|---|
| Card, 2p | 23 | 18 | 3.1 |
| Gin Rummy dial | 8 | 8 | 16.6 |
| Card, 3-4p | 21 | 15 | 31.1 |
| Dice | 6 | 6 | 6.5 |
| Bargaining, signalling | 8 | 5 | 1.0 |
| Board and fog of war | 19 | 6 | 0.0 |
| Solo vs chance | 3 | 0 | — |

Three findings worth the trip:

**Two-player card games saturate.** Only two of 23 hide more than Gin Rummy's 30.1 bits: bridge
uncontested bidding (32.9) and UNO (33.9). A ladder built from them bunches instead of spanning.

**The instrument lies in two directions.** Resampling returned exactly 6.32 bits for Hearts, Euchre
and bridge, which is `log2(80)`, the sample count, with 100% of points censored, against a
closed-form 56.2 for Hearts. And it returns **0.00 bits with zero censoring** for every Gin Rummy
configuration, because the engine's resampler hands back the same world every draw.

**Fog of war is a blind spot.** Of 19 board games the only six we can measure are the
perfect-information controls. Every dark chess, Kriegspiel, phantom Go and RBC row is unmeasured.

## Part 2, the probes

| Probe | Question | Measured (% vs the fixed expert) | Verdict |
|---|---|---|---|
| Capacity | is the network the limit? | every encoder in 21.2–30.0 | no |
| Value of information | what is the hidden state worth? | fair 10.3–25.6 vs oracle 40.7–85.3 | upper bound only |
| Oracle observation | information-bound or learning-bound? | 24.7 vs 25.5 baseline | no benefit detected |

**The oracle observation is the new experiment.** The learner normally sees four card planes; the
oracle variant appends a fifth holding the opponent's true hand. PettingZoo's own
`opponents_hand_visible=True` is *not* this: it exposes an "unknown cards" plane holding 41 cards on
a fresh deal against the opponent's 10, and carries no hidden information. Our plane is verified
against engine ground truth at 1,200 decision points before any compute is spent, and each
curriculum opponent's input is trimmed to the width its own network was trained on so the extra
plane reaches the learner alone.

## Layout

```
hidden_info_survey/   the atlas: three estimators, the catalog, the two engine probes
oracle_probes/        the oracle observation, the opponent shim, probe readers
scripts/              run the survey, train, collect, build tables and figures
data/                 every measurement, committed; the source for every number in the paper
tables/ figures/      generated; regenerate rather than edit
paper/main.tex        the paper (4 pages + references)
slurm/oracle.slurm    array launcher for the oracle seeds
```

## Running on CARC

`PYTHONNOUSERSITE=1` is required for anything that loads the curated seed models: the conda env has
NumPy 2.4.6 while `~/.local` has 1.26.4 and shadows it. `OMP_NUM_THREADS=1` or the forked vectorised
envs deadlock after the first evaluation. Both are set by `slurm/oracle.slurm`.

## Plateau decomposition (new)

A plateau splits into three measurable parts, and the split is what this repository now
measures:

- **what the hidden information is worth** — the difference between the best return achievable
  with and without it, exact wherever counterfactual regret minimisation solves the game;
- **how much of that the learner banks** — `capture`, its competence at using what it is given;
- **what the wider observation costs** — a placebo arm carrying a fifth channel with the same
  shape, the same sparsity and no information.

Worth is a property of the game and does not move: measured across a 64x range of training
budgets it is constant to three decimals (Kuhn 0.388, Leduc 1.317). Capture is a property of
the learner and grows with budget (Leduc 55.6% -> 78.7%).

Capture above 100% is not an error. It means the learner sits further below its ceiling without
the oracle than with it, so the extra observation both supplies information and makes the task
easier to learn. All five liar's dice variants behave that way; Leduc is the opposite, where the
oracle triples the state space and capture falls to 73%.

On Gin Rummy, at twelve seeds per arm: oracle minus baseline -0.25pp (95% CI -1.78 to +1.28),
and the placebo shows the widening itself costs -0.03pp. The null is not an artefact of paying
for a wider observation, because there is no cost to pay.

Reproduce: `scripts/probe_axis.py` (tabular, exact ceilings), `scripts/probe_capacity.py`,
`scripts/probe_neural.py` (same apparatus as the Gin Rummy learner), `scripts/collect_axis.py`,
`scripts/probe_figures_v2.py`.
