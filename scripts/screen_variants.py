"""Screen parameterised OpenSpiel variants for the probe.

Most games are far too large at their default settings, but nearly all of them take size
parameters. Shrinking a game keeps its rules and changes only how much it hides, which is the
axis this paper is built on. This finds every variant small enough to solve exactly.
"""
from __future__ import annotations
import json, os, signal
import pyspiel

CAP, SECS = 40_000, 25
class TO(Exception): pass
signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TO()))

CANDIDATES = []
for n in (3, 4, 5, 6):
    CANDIDATES.append(f"kuhn_poker(players=2,betting_abstraction=fc)" if False else f"kuhn_poker")
    break
CANDIDATES = [
    "kuhn_poker",
    "leduc_poker",
    "leduc_poker(suit_size=2,card_set_size=3)",
    "leduc_poker(action_mapping=true)",
    "liars_dice(numdice=1,dice_sides=3)",
    "liars_dice(numdice=1,dice_sides=4)",
    "liars_dice(numdice=1,dice_sides=5)",
    "liars_dice(numdice=1,dice_sides=6)",
    "liars_dice_ir(numdice=1,dice_sides=4)",
    "liars_dice_ir(numdice=1,dice_sides=6)",
    "goofspiel(num_cards=3,imp_info=True,points_order=descending)",
    "goofspiel(num_cards=4,imp_info=True,points_order=descending)",
    "goofspiel(num_cards=5,imp_info=True,points_order=descending)",
    "goofspiel(num_cards=4,imp_info=True,points_order=random)",
    "oshi_zumo(coins=4,size=2)",
    "oshi_zumo(coins=6,size=2)",
    "oshi_zumo(coins=8,size=3)",
    "first_sealed_auction(max_value=3)",
    "first_sealed_auction(max_value=5)",
    "first_sealed_auction(max_value=7)",
    "tiny_hanabi",
    "dark_hex(board_size=2)",
    "dark_hex_ir(board_size=2)",
    "dark_hex(board_size=3)",
    "phantom_ttt_ir",
    "blotto(coins=4,fields=2,players=2)",
    "matrix_rps",
]

# --- second pass: widen the axis with more families and sizes ---
EXTRA = [
    "liars_dice(numdice=2,dice_sides=3)",
    "liars_dice(numdice=1,dice_sides=8)",
    "liars_dice_ir(numdice=1,dice_sides=5)",
    "liars_dice_ir(numdice=1,dice_sides=8)",
    "leduc_poker(players=2,suit_isomorphism=true)",
    "kuhn_poker(players=2)",
    "dark_hex_ir(board_size=3,gameversion=adh)",
    "phantom_ttt_ir(obstype=reveal-nothing)",
    "python_kuhn_poker",
    "python_leduc_poker",
]


def count(game):
    seen, stack = set(), [game.new_initial_state()]
    while stack:
        s = stack.pop()
        if s.is_terminal():
            continue
        if not s.is_chance_node():
            seen.add(s.information_state_string())
            if len(seen) > CAP:
                raise TO()
        acts = [o for o, _ in s.chance_outcomes()] if s.is_chance_node() else s.legal_actions()
        for a in acts:
            c = s.clone(); c.apply_action(a); stack.append(c)
    return len(seen)

rows = []
for spec in CANDIDATES + EXTRA:
    try:
        signal.alarm(SECS)
        g = pyspiel.load_game(spec)
        t = g.get_type()
        if not (g.num_players() == 2
                and t.information == pyspiel.GameType.Information.IMPERFECT_INFORMATION
                and t.utility == pyspiel.GameType.Utility.ZERO_SUM
                and t.dynamics == pyspiel.GameType.Dynamics.SEQUENTIAL):
            signal.alarm(0); print(f"  no   {spec:52s} wrong type"); continue
        n = count(g)
        signal.alarm(0)
        rows.append({"spec": spec, "info_states": n, "actions": g.num_distinct_actions()})
        print(f"  OK   {spec:52s} {n:7d} info states", flush=True)
    except TO:
        signal.alarm(0); print(f"  big  {spec:52s} > {CAP} states", flush=True)
    except Exception as e:
        signal.alarm(0); print(f"  skip {spec:52s} {type(e).__name__}: {str(e)[:50]}", flush=True)

rows.sort(key=lambda r: r["info_states"])
here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
json.dump(rows, open(os.path.join(here, "data", "variants.json"), "w"), indent=1)
print(f"\n{len(rows)} usable variants")

