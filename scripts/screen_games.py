"""Which OpenSpiel games can carry the probe?

The probe needs four things: two players, zero sum, imperfect information, and a game small
enough that counterfactual regret minimisation gives a true optimum in reasonable time. This
screens every registered game against those requirements and caps the tree walk, so a game
that is too big is rejected quickly instead of hanging.
"""
from __future__ import annotations
import json, os, signal, sys, time
import pyspiel

CAP = 60_000          # information states we are willing to enumerate
PER_GAME_SECONDS = 20

class Timeout(Exception): pass
def _alarm(*_): raise Timeout()
signal.signal(signal.SIGALRM, _alarm)

def count_states(game):
    seen, stack = set(), [game.new_initial_state()]
    while stack:
        s = stack.pop()
        if s.is_terminal():
            continue
        if not s.is_chance_node():
            seen.add(s.information_state_string())
            if len(seen) > CAP:
                raise Timeout()
        for a in (o for o, _ in s.chance_outcomes()) if s.is_chance_node() else s.legal_actions():
            c = s.clone(); c.apply_action(a); stack.append(c)
    return len(seen)

rows = []
for name in [g.short_name for g in pyspiel.registered_games() if g.default_loadable]:
    try:
        signal.alarm(PER_GAME_SECONDS)
        game = pyspiel.load_game(name)
        t = game.get_type()
        ok = (game.num_players() == 2
              and t.information == pyspiel.GameType.Information.IMPERFECT_INFORMATION
              and t.utility == pyspiel.GameType.Utility.ZERO_SUM
              and t.dynamics == pyspiel.GameType.Dynamics.SEQUENTIAL
              and t.chance_mode != pyspiel.GameType.ChanceMode.SAMPLED_STOCHASTIC)
        if not ok:
            signal.alarm(0); continue
        n = count_states(game)
        signal.alarm(0)
        rows.append({"game": name, "info_states": n,
                     "actions": game.num_distinct_actions(),
                     "tensor": game.information_state_tensor_size()})
        print(f"  OK   {name:28s} {n:7d} info states", flush=True)
    except Timeout:
        signal.alarm(0); print(f"  big  {name:28s} over {CAP} states or {PER_GAME_SECONDS}s", flush=True)
    except Exception as e:
        signal.alarm(0); print(f"  skip {name:28s} {type(e).__name__}", flush=True)

rows.sort(key=lambda r: r["info_states"])
json.dump(rows, open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                  "data", "screen.json"), "w"), indent=1)
print(f"\n{len(rows)} games usable for the probe")
