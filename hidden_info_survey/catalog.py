"""What gets measured, and why each game is on the list.

Two engines. OpenSpiel is the primary one, because it exposes information-state strings and world
resampling, which is what makes exact and resampled estimates possible at all. RLCard is here for
the games OpenSpiel does not ship, Mahjong and UNO above all, and for one deliberate duplicate:
Gin Rummy is measured by both, so the closed form can be checked across libraries.

`hidden` is the closed-form bit count where the hidden state is a known deal, written as the
arithmetic rather than a constant so it can be checked by eye. It is left None wherever the hidden
state is not simply a set of hands, which is most of the fog-of-war and bargaining games.

Families group games by what kind of problem they pose, not by publisher:
  card   two-player card games          dial   the Gin Rummy deck and hand-size dial
  board  board games and fog of war     dice   dice games
  comm   bargaining, signalling, auctions
  multi  three and four-handed card games
  solo   one player against a shuffled deck, no opponent to model
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from hidden_info_survey.hidden_bits import deal_bits, hand_bits

__all__ = ["GameSpec", "OPENSPIEL_GAMES", "RLCARD_GAMES", "FAMILIES"]

FAMILIES = {
    "card": "Card games, 2p",
    "dial": "Gin Rummy dial",
    "board": "Board and fog of war",
    "dice": "Dice games",
    "comm": "Bargaining and signalling",
    "multi": "Card games, 3-4p",
    "solo": "Solo against chance",
}


@dataclass(frozen=True)
class GameSpec:
    key: str  # the engine's own game string
    label: str  # what appears in tables and figures
    family: str
    hidden: float | None = None  # closed-form bits, when the deal is known
    note: str = ""  # what is hidden, in words


def _dice(n):  # each die is one of six faces, and they are independent
    return n * math.log2(6)


# --------------------------------------------------------------------------- OpenSpiel
OPENSPIEL_GAMES = [
    # -- the anchor and its dial: identical rules, only the deck and hand size move ------------
    GameSpec("gin_rummy(num_ranks=7,num_suits=2,hand_size=5,knock_card=5)",
             "Gin dial: deck 14, hand 5", "dial", hand_bits(8, 5)),
    GameSpec("gin_rummy(num_ranks=8,num_suits=2,hand_size=6,knock_card=6)",
             "Gin dial: deck 16, hand 6", "dial", hand_bits(9, 6)),
    GameSpec("gin_rummy(num_ranks=9,num_suits=3,hand_size=7,knock_card=7)",
             "Gin dial: deck 27, hand 7", "dial", hand_bits(19, 7)),
    GameSpec("gin_rummy(num_ranks=10,num_suits=3,hand_size=8,knock_card=8)",
             "Gin dial: deck 30, hand 8", "dial", hand_bits(21, 8)),
    GameSpec("gin_rummy(num_ranks=11,num_suits=4,hand_size=9,knock_card=9)",
             "Gin dial: deck 44, hand 9", "dial", hand_bits(34, 9)),
    GameSpec("gin_rummy(num_ranks=13,num_suits=4,hand_size=10,knock_card=10)",
             "Gin rummy (standard)", "dial", hand_bits(41, 10),
             "opponent's 10 cards out of the 41 I have not seen"),

    # -- two-player card games ------------------------------------------------------------------
    GameSpec("kuhn_poker", "Kuhn poker", "card", hand_bits(2, 1)),
    GameSpec("leduc_poker", "Leduc poker", "card", hand_bits(5, 1)),
    GameSpec("repeated_leduc_poker", "Repeated Leduc poker", "card", hand_bits(5, 1)),
    GameSpec("universal_poker", "Universal poker (default)", "card"),
    GameSpec("tiny_bridge_2p", "Tiny bridge 2p", "card", hand_bits(6, 2)),
    GameSpec("bridge_uncontested_bidding", "Bridge uncontested bidding", "card",
             hand_bits(39, 13), "partner's 13 cards out of 39 unseen"),
    GameSpec("crazy_eights(players=2)", "Crazy eights 2p", "card", hand_bits(46, 5)),
    GameSpec("crazy_eights(players=2,use_special_cards=True)",
             "Crazy eights 2p (special cards)", "card", hand_bits(46, 5)),
    GameSpec("cribbage(players=2)", "Cribbage 2p", "card", deal_bits(46, [6])),
    GameSpec("hanabi(players=2)", "Hanabi 2p (self-hidden)", "card",
             note="your own hand, which only your partner can see"),
    GameSpec("hanabi(players=2,hand_size=3)", "Hanabi 2p, hand 3", "card"),
    GameSpec("tiny_hanabi", "Tiny Hanabi", "card"),
    GameSpec("goofspiel(num_cards=5,imp_info=True,points_order=descending)",
             "Goofspiel-5 (hidden bids)", "card"),
    GameSpec("goofspiel(num_cards=13,imp_info=True,points_order=descending)",
             "Goofspiel-13 (hidden bids)", "card"),
    # perfect-information controls: the zero end of the axis has to be on the plot
    GameSpec("goofspiel(num_cards=5,imp_info=False,points_order=descending)",
             "Goofspiel-5 (open)", "card", 0.0),
    GameSpec("goofspiel(num_cards=13,imp_info=False,points_order=descending)",
             "Goofspiel-13 (open)", "card", 0.0),

    # -- dice -----------------------------------------------------------------------------------
    GameSpec("liars_dice(numdice=1)", "Liar's dice, 1 die", "dice", _dice(1)),
    GameSpec("liars_dice(numdice=2)", "Liar's dice, 2 dice", "dice", _dice(2)),
    GameSpec("liars_dice(numdice=5)", "Liar's dice, 5 dice", "dice", _dice(5)),
    GameSpec("liars_dice_ir(numdice=2)", "Liar's dice 2 dice (imperfect recall)", "dice",
             _dice(2), "same dice, but the player forgets the bidding history"),

    # -- board games and fog of war -------------------------------------------------------------
    GameSpec("connect_four", "Connect Four", "board", 0.0),
    GameSpec("oshi_zumo", "Oshi Zumo", "board", 0.0),
    GameSpec("phantom_ttt", "Phantom tic-tac-toe", "board"),
    GameSpec("phantom_ttt_ir", "Phantom tic-tac-toe (imperfect recall)", "board"),
    GameSpec("latent_ttt", "Latent tic-tac-toe", "board"),
    GameSpec("dark_hex(board_size=3)", "Dark Hex 3x3 (perfect recall)", "board"),
    GameSpec("dark_hex_ir(board_size=3)", "Dark Hex 3x3", "board"),
    GameSpec("dark_hex_ir(board_size=4)", "Dark Hex 4x4", "board"),
    GameSpec("dark_chess(board_size=4)", "Dark chess 4x4", "board"),
    GameSpec("dark_chess", "Dark chess 8x8", "board"),
    GameSpec("kriegspiel(board_size=4)", "Kriegspiel 4x4", "board"),
    GameSpec("phantom_go(board_size=5)", "Phantom Go 5x5", "board"),
    GameSpec("rbc(board_size=4)", "Reconnaissance blind chess 4x4", "board"),
    GameSpec("rbc", "Reconnaissance blind chess 8x8", "board"),
    GameSpec("battleship", "Battleship 10x10", "board",
             note="where the opponent placed their ships"),

    # -- bargaining, signalling, auctions: hidden preferences rather than hidden cards ----------
    GameSpec("bargaining", "Bargaining", "comm"),
    GameSpec("negotiation", "Negotiation", "comm"),
    GameSpec("trade_comm", "Trade and communicate", "comm"),
    GameSpec("sheriff", "Sheriff of Nottingham", "comm"),
    GameSpec("lewis_signaling", "Lewis signaling", "comm"),
    GameSpec("coordinated_mp", "Coordinated matching pennies", "comm"),
    GameSpec("first_sealed_auction", "First-price sealed auction", "comm"),
    GameSpec("coop_box_pushing", "Cooperative box pushing", "comm"),

    # -- three and four-handed card games -------------------------------------------------------
    GameSpec("dou_dizhu", "Dou Dizhu 3p", "multi", deal_bits(34, [17])),
    GameSpec("skat", "Skat 3p", "multi", deal_bits(22, [10, 10, 2])),
    GameSpec("hearts", "Hearts 4p", "multi", deal_bits(39, [13, 13, 13])),
    GameSpec("spades", "Spades 4p", "multi", deal_bits(39, [13, 13, 13])),
    GameSpec("euchre", "Euchre 4p", "multi", deal_bits(19, [5, 5, 5])),
    GameSpec("bridge", "Contract bridge 4p", "multi", deal_bits(39, [13, 13, 13])),
    GameSpec("tiny_bridge_4p", "Tiny bridge 4p", "multi"),
    GameSpec("oh_hell", "Oh Hell", "multi"),  # hand size varies by round, so no single closed form
    GameSpec("tarok(players=3)", "Tarok 3p", "multi"),
    GameSpec("colored_trails", "Colored trails 3p", "multi"),
    GameSpec("crazy_eights(players=4)", "Crazy eights 4p", "multi", hand_bits(47, 5)),

    # -- breadth added for the workshop version -------------------------------------------------
    # perfect-information controls: the zero end of the axis needs more than two points
    GameSpec("tic_tac_toe", "Tic-tac-toe", "board", 0.0),
    GameSpec("hex(board_size=3)", "Hex 3x3", "board", 0.0),
    GameSpec("breakthrough(rows=4,columns=4)", "Breakthrough 4x4", "board", 0.0),
    GameSpec("nim", "Nim", "board", 0.0),
    # more rungs on the one dial where rules are held fixed
    GameSpec("gin_rummy(num_ranks=6,num_suits=2,hand_size=4,knock_card=4)",
             "Gin dial: deck 12, hand 4", "dial", hand_bits(7, 4)),
    GameSpec("gin_rummy(num_ranks=12,num_suits=4,hand_size=10,knock_card=10)",
             "Gin dial: deck 48, hand 10", "dial", hand_bits(37, 10)),
    # dice, filling the gap between 2 and 5
    GameSpec("liars_dice(numdice=3)", "Liar's dice, 3 dice", "dice", _dice(3)),
    GameSpec("liars_dice(numdice=4)", "Liar's dice, 4 dice", "dice", _dice(4)),
    # cooperative hidden information as the table grows
    GameSpec("hanabi(players=3)", "Hanabi 3p", "multi"),
    GameSpec("hanabi(players=4)", "Hanabi 4p", "multi"),
    GameSpec("hanabi(players=5)", "Hanabi 5p", "multi"),
    # the smallest poker benchmarks at a bigger table
    GameSpec("kuhn_poker(players=3)", "Kuhn poker 3p", "multi", hand_bits(3, 2)),
    GameSpec("leduc_poker(players=3)", "Leduc poker 3p", "multi", hand_bits(5, 2)),
    # bidding games at more sizes
    GameSpec("goofspiel(num_cards=4,imp_info=True,points_order=descending)",
             "Goofspiel-4 (hidden bids)", "card"),
    GameSpec("goofspiel(num_cards=8,imp_info=True,points_order=descending)",
             "Goofspiel-8 (hidden bids)", "card"),
    GameSpec("crazy_eights(players=3)", "Crazy eights 3p", "multi", hand_bits(46, 5)),
    GameSpec("crazy_eights(players=5)", "Crazy eights 5p", "multi", hand_bits(48, 5)),

    # -- one player against a shuffled deck: the Balatro shape ----------------------------------
    GameSpec("blackjack", "Blackjack (solo vs deck)", "solo",
             note="the dealer's hole card and the order of the shoe"),
    GameSpec("solitaire", "Klondike solitaire", "solo",
             note="the order of the stock, and nobody to model"),
]

# --------------------------------------------------------------------------- RLCard
RLCARD_GAMES = [
    # 108 cards, 7 each. From my seat 108 - my 7 - the face-up card = 100 unseen.
    GameSpec("uno", "UNO 2p", "card", deal_bits(100, [7]),
             "opponent's hand, drawn from the unseen deck"),
    # 136 tiles, 13 each to four players. 136 - my 13 = 123 unseen, three hands of 13.
    GameSpec("mahjong", "Mahjong 4p", "multi", deal_bits(123, [13, 13, 13]),
             "three opponents' hands out of the unseen wall"),
    GameSpec("doudizhu", "Dou Dizhu 3p (RLCard)", "multi", deal_bits(34, [17]),
             "the opposing hands once the kitty is public"),
    # 52 - my 10 - the face-up discard = 41. Must match the OpenSpiel gin_rummy row exactly:
    # the same game measured by two engines is the cheapest cross-check available.
    GameSpec("gin-rummy", "Gin Rummy (RLCard)", "card", deal_bits(41, [10]),
             "opponent's 10-card hand out of 41 unseen"),
    GameSpec("limit-holdem", "Limit Texas hold'em 2p", "card", deal_bits(50, [2]),
             "opponent's two hole cards"),
    GameSpec("no-limit-holdem", "No-limit Texas hold'em 2p", "card", deal_bits(50, [2]),
             "opponent's two hole cards"),
    GameSpec("leduc-holdem", "Leduc hold'em (RLCard)", "card", deal_bits(5, [1]),
             "opponent's single card from a six-card deck"),
    GameSpec("bridge", "Contract bridge (RLCard)", "multi", deal_bits(39, [13, 13, 13]),
             "the other three hands"),
    GameSpec("blackjack", "Blackjack (RLCard)", "solo",
             note="the dealer's hole card and the shoe order, no opponent to model"),
]
