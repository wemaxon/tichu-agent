from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from typing import Any

from agents import Agent, create_agent
from scripts.round_runner import round_config_from_args
from tichu.actions import (
    Action,
    BombAction,
    DeclareSmallTichu,
    DragonGiftAction,
    ExchangeAction,
    GrandTichuDecision,
    PassAction,
    PlayAction,
    combination_from_dict,
)
from tichu.cards import CARD_BY_ID, Special
from tichu.combinations import Combination, CombinationKind
from tichu.round import Phase, TichuRound, partner_of, team_of


class ConsoleQuit(Exception):
    pass


def command_console(args: argparse.Namespace) -> int:
    human = int(args.seat)
    config = round_config_from_args(args)
    agent_names = _agent_names_from_args(args)
    agents = _build_agents(agent_names, human, int(args.seed))

    round_ = TichuRound.new(seed=int(args.seed), dealer=int(args.dealer), config=config)
    print(f"You are P{human}. Your partner is P{partner_of(human)}.")
    print(
        "Agents: "
        + " ".join(f"P{player}={name}" for player, name in _agent_seats(agent_names, human))
    )
    print("Type q to quit at any prompt.")

    try:
        round_ = run_console_round(round_, human, agents)
    except ConsoleQuit:
        print("\nGame aborted.")
        return 130

    assert round_.round_scores is not None
    print("\nRound over.")
    print(f"Finish order: {' '.join(f'P{player}' for player in round_.finish_order)}")
    print(f"Round score: {round_.round_scores[0]}:{round_.round_scores[1]}")

    if args.log:
        round_.save_replay(args.log)
        print(f"Replay written to {args.log}")
    return 0


def run_console_round(
    round_: TichuRound,
    human: int,
    agents: Mapping[int, Agent],
    *,
    safety_limit: int = 20_000,
) -> TichuRound:
    steps = 0
    while not round_.terminal:
        player = round_.current_player
        if player is None:
            raise RuntimeError("No current player in a non-terminal round")

        if player == human:
            _print_human_turn(round_, human)
            action = _prompt_human_action(round_, human)
        else:
            agent = agents[player]
            action = agent.choose_action(round_.observation(player), round_.legal_actions())

        before_events = len(round_.public_history)
        round_ = round_.apply(action)
        _print_action_result(player, action, round_.public_history[before_events:], human=human)

        steps += 1
        if steps > safety_limit:
            raise RuntimeError("Round exceeded the safety step limit")

    return round_


def _agent_names_from_args(args: argparse.Namespace) -> tuple[str, str, str]:
    if args.agents is not None:
        names = tuple(str(name) for name in args.agents)
        if len(names) != 3:
            raise ValueError("Exactly three agent names are required")
        return names[0], names[1], names[2]
    return args.agent, args.agent, args.agent


def _agent_seats(agent_names: Sequence[str], human: int) -> tuple[tuple[int, str], ...]:
    seats = tuple(player for player in range(4) if player != human)
    return tuple(zip(seats, agent_names))


def _build_agents(agent_names: Sequence[str], human: int, seed: int) -> dict[int, Agent]:
    seats = tuple(player for player in range(4) if player != human)
    if len(agent_names) != len(seats):
        raise ValueError("Exactly three agent names are required")
    return {
        player: create_agent(name, seed * 17 + player)
        for player, name in zip(seats, agent_names)
    }


def _print_human_turn(round_: TichuRound, human: int) -> None:
    print("\n" + "=" * 72)
    print(
        f"Phase: {round_.phase.value} | Current: P{round_.current_player} "
        f"| You: P{human} team {team_of(human)}"
    )
    print(
        "Hand sizes: "
        + " ".join(f"P{player}={len(round_.hands[player])}" for player in range(4))
    )
    if round_.finish_order:
        print("Finished: " + " ".join(f"P{player}" for player in round_.finish_order))
    if any(call is not None for call in round_.tichu_calls):
        print(
            "Tichu calls: "
            + " ".join(
                f"P{player}={call}"
                for player, call in enumerate(round_.tichu_calls)
                if call is not None
            )
        )
    if round_.wish_rank is not None:
        print(f"Wish: {_rank_label(round_.wish_rank)}")
    if round_.top_combination is None:
        print("Top play: none")
    else:
        top_player = "unknown" if round_.last_play_player is None else f"P{round_.last_play_player}"
        print(f"Top play: {top_player} {_format_combination(round_.top_combination)}")

    print("\nYour hand:")
    _print_indexed_cards(_sorted_hand_ids(round_.hands[human]))


def _prompt_human_action(round_: TichuRound, human: int) -> Action:
    actions = round_.legal_actions()
    if not actions:
        raise RuntimeError("No legal actions are available")

    if round_.phase is Phase.GRAND_TICHU:
        return _prompt_grand_tichu(actions)
    if round_.phase is Phase.EXCHANGE:
        return _prompt_exchange(round_, human, actions)
    if round_.phase is Phase.PLAY:
        return _prompt_play_action(round_, human, actions)
    if round_.phase is Phase.BOMB_WINDOW:
        return _prompt_bomb_action(round_, human, actions)
    if round_.phase is Phase.DRAGON_GIFT:
        return _prompt_dragon_gift(actions)
    return _prompt_disambiguation(actions)


def _prompt_grand_tichu(actions: Sequence[Action]) -> GrandTichuDecision:
    while True:
        raw = _input("Call grand Tichu? [y/N]: ").lower()
        if raw in {"", "n", "no"}:
            action = GrandTichuDecision(False)
        elif raw in {"y", "yes"}:
            action = GrandTichuDecision(True)
        else:
            print("Enter y or n.")
            continue
        if action in actions:
            return action
        print("That decision is not legal now.")


def _prompt_exchange(round_: TichuRound, human: int, actions: Sequence[Action]) -> ExchangeAction:
    recipients = tuple(player for player in range(4) if player != human)
    prompt = " ".join(f"P{recipient}" for recipient in recipients)
    hand = _sorted_hand_ids(round_.hands[human])

    while True:
        raw = _input(f"Enter card indexes to send to {prompt}: ")
        parts = raw.split()
        if len(parts) != len(recipients):
            print("Enter exactly three indexes.")
            _redisplay_hand(round_, human)
            continue
        try:
            indexes = tuple(int(part) for part in parts)
        except ValueError:
            print("Indexes must be numbers.")
            _redisplay_hand(round_, human)
            continue
        if len(set(indexes)) != len(indexes):
            print("Each exchanged card must be different.")
            _redisplay_hand(round_, human)
            continue
        if any(index < 0 or index >= len(hand) for index in indexes):
            print("One or more indexes are outside your hand.")
            _redisplay_hand(round_, human)
            continue

        selected = tuple(hand[index] for index in indexes)
        action = ExchangeAction(tuple(sorted(zip(recipients, selected), key=lambda item: item[0])))
        if action in actions:
            return action
        print("That exchange is not legal now.")
        _redisplay_hand(round_, human)


def _prompt_play_action(round_: TichuRound, human: int, actions: Sequence[Action]) -> Action:
    print("\n" + _play_requirement(round_, actions))
    if _small_tichu_action(actions) is not None:
        print("Type t to declare small Tichu before playing.")
    if _pass_action(actions) is not None:
        print("Type p to pass.")

    while True:
        raw = _input("Enter card indexes to play: ").lower()
        if raw in {"t", "tichu", "small", "small tichu"}:
            action = _small_tichu_action(actions)
            if action is not None:
                return action
            print("Small Tichu is not legal now.")
            _redisplay_turn(round_, human, actions)
            continue
        if raw in {"p", "pass"}:
            action = _pass_action(actions)
            if action is not None:
                return action
            print("Passing is not legal now.")
            _redisplay_turn(round_, human, actions)
            continue

        selected = _selected_card_ids(raw, round_, human)
        if selected is None:
            _redisplay_turn(round_, human, actions)
            continue

        matches = _matching_play_actions(actions, selected)
        if not matches:
            print("Those cards are not a legal play here.")
            _redisplay_turn(round_, human, actions)
            continue
        return _choose_matching_play(matches)


def _prompt_bomb_action(round_: TichuRound, human: int, actions: Sequence[Action]) -> BombAction:
    print("\nNeed: a legal bomb stronger than the current top play, or decline.")
    print("Type n to decline.")

    while True:
        raw = _input("Enter bomb card indexes: ").lower()
        if raw in {"n", "no", "d", "decline", "p", "pass"}:
            action = _decline_bomb_action(actions)
            if action is not None:
                return action
            print("Declining is not legal now.")
            _redisplay_bomb_prompt(round_, human)
            continue

        selected = _selected_card_ids(raw, round_, human)
        if selected is None:
            _redisplay_bomb_prompt(round_, human)
            continue

        matches = [
            action
            for action in actions
            if (
                isinstance(action, BombAction)
                and action.combination is not None
                and tuple(sorted(action.combination.card_ids)) == selected
            )
        ]
        if not matches:
            print("Those cards are not a legal bomb here.")
            _redisplay_bomb_prompt(round_, human)
            continue
        return matches[0]


def _prompt_dragon_gift(actions: Sequence[Action]) -> DragonGiftAction:
    recipients = [
        action.recipient
        for action in actions
        if isinstance(action, DragonGiftAction)
    ]
    print("\nChoose who receives the Dragon trick: " + " ".join(f"P{player}" for player in recipients))

    while True:
        raw = _input("Recipient player: ").lower().removeprefix("p")
        try:
            recipient = int(raw)
        except ValueError:
            print("Enter one of the listed player numbers.")
            continue
        action = DragonGiftAction(recipient)
        if action in actions:
            return action
        print("That recipient is not legal.")


def _prompt_disambiguation(actions: Sequence[Action]) -> Action:
    print("\nChoose interpretation:")
    for index, action in enumerate(actions):
        print(f"[{index}] {_format_action(action)}")

    while True:
        raw = _input("Choice: ")
        try:
            index = int(raw)
        except ValueError:
            print("Enter a number.")
            continue
        if 0 <= index < len(actions):
            return actions[index]
        print("That choice is outside the listed range.")


def _input(prompt: str) -> str:
    try:
        raw = input(prompt).strip()
    except EOFError as exc:
        raise ConsoleQuit from exc
    if raw.lower() in {"q", "quit", "exit"}:
        raise ConsoleQuit
    return raw


def _print_action_result(
    player: int,
    action: Action,
    events: Sequence[Mapping[str, object]],
    *,
    human: int,
) -> None:
    actor = "You" if player == human else f"P{player}"
    if events:
        for event in events:
            print(_format_event(event, human=human))
        return
    print(f"{actor}: {_format_action(action)}")


def _print_indexed_cards(card_ids: Sequence[int]) -> None:
    items = [f"[{index}] {_card_label(card_id)}" for index, card_id in enumerate(card_ids)]
    for offset in range(0, len(items), 4):
        print("  " + "  ".join(items[offset : offset + 4]))


def _sorted_hand_ids(card_ids: Sequence[int]) -> tuple[int, ...]:
    return tuple(sorted(card_ids, key=_hand_card_key))


def _redisplay_hand(round_: TichuRound, human: int) -> None:
    print("\nYour hand:")
    _print_indexed_cards(_sorted_hand_ids(round_.hands[human]))


def _redisplay_turn(round_: TichuRound, human: int, actions: Sequence[Action]) -> None:
    _print_human_turn(round_, human)
    print("\n" + _play_requirement(round_, actions))
    if _small_tichu_action(actions) is not None:
        print("Type t to declare small Tichu before playing.")
    if _pass_action(actions) is not None:
        print("Type p to pass.")


def _redisplay_bomb_prompt(round_: TichuRound, human: int) -> None:
    _print_human_turn(round_, human)
    print("\nNeed: a legal bomb stronger than the current top play, or decline.")
    print("Type n to decline.")


def _play_requirement(round_: TichuRound, actions: Sequence[Action]) -> str:
    top = round_.top_combination
    wish_rank = round_.wish_rank
    legal_plays = [action for action in actions if isinstance(action, PlayAction)]
    forced_wish = (
        wish_rank is not None
        and bool(legal_plays)
        and all(action.combination.contains_normal_rank(wish_rank) for action in legal_plays)
    )

    if top is None:
        if forced_wish:
            return f"Need: a legal combination containing wished rank {_rank_label(wish_rank)}."
        return "Need: lead any legal combination."

    if top.is_bomb:
        need = f"Need: a stronger bomb than {_format_combination(top)}."
    else:
        need = (
            f"Need: {_kind_label(top.kind)}"
            f"{_length_suffix(top)} stronger than {_format_combination(top)}, or a bomb."
        )
    if forced_wish:
        need += f" It must contain wished rank {_rank_label(wish_rank)}."
    if _pass_action(actions) is not None:
        need += " You may pass."
    return need


def _length_suffix(combo: Combination) -> str:
    if combo.kind in {
        CombinationKind.SINGLE,
        CombinationKind.PAIR,
        CombinationKind.TRIPLE,
        CombinationKind.FULL_HOUSE,
        CombinationKind.DOG,
    }:
        return ""
    return f" of length {combo.length}"


def _selected_card_ids(raw: str, round_: TichuRound, human: int) -> tuple[int, ...] | None:
    hand = _sorted_hand_ids(round_.hands[human])
    parts = raw.replace(",", " ").split()
    if not parts:
        print("Enter at least one card index.")
        return None
    try:
        indexes = tuple(int(part) for part in parts)
    except ValueError:
        print("Indexes must be numbers.")
        return None
    if len(set(indexes)) != len(indexes):
        print("Each selected card must be different.")
        return None
    if any(index < 0 or index >= len(hand) for index in indexes):
        print("One or more indexes are outside your hand.")
        return None
    return tuple(sorted(hand[index] for index in indexes))


def _matching_play_actions(actions: Sequence[Action], selected: tuple[int, ...]) -> tuple[PlayAction, ...]:
    return tuple(
        action
        for action in actions
        if (
            isinstance(action, PlayAction)
            and tuple(sorted(action.combination.card_ids)) == selected
        )
    )


def _choose_matching_play(matches: Sequence[PlayAction]) -> PlayAction:
    if len(matches) == 1:
        return matches[0]

    combinations = {action.combination for action in matches}
    if len(combinations) == 1 and next(iter(combinations)).contains_special(Special.MAHJONG):
        return _prompt_wish_rank(matches)

    return _prompt_disambiguation(matches)


def _prompt_wish_rank(actions: Sequence[PlayAction]) -> PlayAction:
    by_rank = {action.wish_rank: action for action in actions}
    if len(by_rank) == 1:
        return actions[0]

    while True:
        raw = _input("Wish rank 2-10,J,Q,K,A, or blank for no wish: ").lower()
        wish_rank = _parse_wish_rank(raw)
        if wish_rank in by_rank:
            return by_rank[wish_rank]
        print("That wish is not legal for this play.")


def _parse_wish_rank(raw: str) -> int | None:
    if raw in {"", "n", "no", "none"}:
        return None
    labels = {"j": 11, "jack": 11, "q": 12, "queen": 12, "k": 13, "king": 13, "a": 14, "ace": 14}
    if raw in labels:
        return labels[raw]
    try:
        value = int(raw)
    except ValueError:
        return -1
    if value in range(2, 15):
        return value
    return -1


def _small_tichu_action(actions: Sequence[Action]) -> DeclareSmallTichu | None:
    return next((action for action in actions if isinstance(action, DeclareSmallTichu)), None)


def _pass_action(actions: Sequence[Action]) -> PassAction | None:
    return next((action for action in actions if isinstance(action, PassAction)), None)


def _decline_bomb_action(actions: Sequence[Action]) -> BombAction | None:
    return next(
        (
            action
            for action in actions
            if isinstance(action, BombAction) and action.combination is None
        ),
        None,
    )


def _format_event(event: Mapping[str, object], *, human: int) -> str:
    event_type = event.get("type")
    player = event.get("player")
    actor = _actor_label(player, human)

    if event_type == "grand_tichu":
        return f"{actor} called grand Tichu."
    if event_type == "small_tichu":
        return f"{actor} called small Tichu."
    if event_type == "exchange_submitted":
        return f"{actor} submitted exchange cards."
    if event_type == "exchange_complete":
        return "Exchange complete."
    if event_type == "wish_made":
        return f"{actor} wished for {_rank_label(_int_event(event, 'rank'))}."
    if event_type == "wish_fulfilled":
        return f"{actor} fulfilled the wish for {_rank_label(_int_event(event, 'rank'))}."
    if event_type == "play":
        combo = combination_from_dict(_dict_event(event, "combination"))
        return f"{actor} played {_format_combination(combo)}."
    if event_type == "pass":
        return f"{actor} passed."
    if event_type == "decline_bomb":
        return f"{actor} declined to bomb."
    if event_type == "bomb":
        combo = combination_from_dict(_dict_event(event, "combination"))
        return f"{actor} bombed with {_format_combination(combo)}."
    if event_type == "dragon_gift":
        winner = _actor_label(event.get("winner"), human)
        recipient = _actor_label(event.get("recipient"), human)
        return f"{winner} gave the Dragon trick to {recipient}."
    if event_type == "trick_won":
        return f"{actor} won the trick."
    if event_type == "final_trick_won":
        return f"{actor} won the final trick."
    if event_type == "went_out":
        return f"{actor} went out in position {_int_event(event, 'position')}."
    if event_type == "round_over":
        scores = event.get("scores")
        double = event.get("double_victory")
        return f"Round over. Double victory={double}. Scores={scores}."
    return f"Event: {dict(event)}"


def _format_action(action: Action) -> str:
    if isinstance(action, GrandTichuDecision):
        return "call grand Tichu" if action.call else "decline grand Tichu"
    if isinstance(action, ExchangeAction):
        return "exchange " + ", ".join(
            f"{_card_label(card_id)} to P{recipient}"
            for recipient, card_id in action.transfers
        )
    if isinstance(action, DeclareSmallTichu):
        return "declare small Tichu"
    if isinstance(action, PlayAction):
        text = "play " + _format_combination(action.combination)
        if action.combination.contains_special(Special.MAHJONG):
            if action.wish_rank is None:
                text += " with no wish"
            else:
                text += f" and wish for {_rank_label(action.wish_rank)}"
        return text
    if isinstance(action, PassAction):
        return "pass"
    if isinstance(action, BombAction):
        if action.combination is None:
            return "decline bomb"
        return "bomb with " + _format_combination(action.combination)
    if isinstance(action, DragonGiftAction):
        return f"give Dragon trick to P{action.recipient}"
    return str(action)


def _format_combination(combo: Combination) -> str:
    if combo.kind is CombinationKind.DOG:
        return f"dog ({_format_combination_cards(combo)})"
    kind = _kind_label(combo.kind)
    rank = _rank_label(combo.primary_rank)
    cards = _format_combination_cards(combo)
    phoenix = ""
    if combo.phoenix_as is not None:
        phoenix = f", Phoenix as {_rank_label(combo.phoenix_as)}"
    return f"{kind} {rank} ({cards}{phoenix})"


def _kind_label(kind: CombinationKind) -> str:
    labels = {
        CombinationKind.DOG: "dog",
        CombinationKind.SINGLE: "single",
        CombinationKind.PAIR: "pair",
        CombinationKind.TRIPLE: "triple",
        CombinationKind.FULL_HOUSE: "full house",
        CombinationKind.PAIR_SEQUENCE: "pair sequence",
        CombinationKind.STRAIGHT: "straight",
        CombinationKind.FOUR_BOMB: "four-of-a-kind bomb",
        CombinationKind.STRAIGHT_FLUSH_BOMB: "straight-flush bomb",
    }
    return labels[kind]


def _rank_label(rank: int | float) -> str:
    if isinstance(rank, float) and not rank.is_integer():
        return str(rank)
    value = int(rank)
    labels = {1: "Mahjong", 11: "J", 12: "Q", 13: "K", 14: "A", 15: "Dragon"}
    return labels.get(value, str(value))


def _format_combination_cards(combo: Combination) -> str:
    card_ids = sorted(combo.card_ids, key=lambda card_id: _combination_card_key(combo, card_id))
    return " ".join(_card_label(card_id) for card_id in card_ids)


def _combination_card_key(combo: Combination, card_id: int) -> tuple[float, str]:
    card = CARD_BY_ID[card_id]
    if card.is_normal:
        return float(card.normal_rank), card.normal_suit.value
    if card.special is Special.MAHJONG:
        return 1.0, card.label
    if card.special is Special.PHOENIX:
        return float(combo.phoenix_as) if combo.phoenix_as is not None else 1.5, card.label
    if card.special is Special.DRAGON:
        return 15.0, card.label
    return 0.0, card.label


def _hand_card_key(card_id: int) -> tuple[float, str]:
    card = CARD_BY_ID[card_id]
    if card.is_normal:
        return float(card.normal_rank), card.normal_suit.value
    if card.special is Special.MAHJONG:
        return 1.0, card.label
    if card.special is Special.DOG:
        return 1.25, card.label
    if card.special is Special.PHOENIX:
        return 1.5, card.label
    if card.special is Special.DRAGON:
        return 15.0, card.label
    return 0.0, card.label


def _card_label(card_id: int) -> str:
    return CARD_BY_ID[card_id].label


def _actor_label(value: object, human: int) -> str:
    if isinstance(value, int):
        return "You" if value == human else f"P{value}"
    return "Unknown"


def _int_event(event: Mapping[str, object], key: str) -> int:
    value = event[key]
    if not isinstance(value, int):
        raise TypeError(f"Expected integer event field {key!r}")
    return value


def _dict_event(event: Mapping[str, object], key: str) -> dict[str, Any]:
    value = event[key]
    if not isinstance(value, dict):
        raise TypeError(f"Expected object event field {key!r}")
    return {str(field): raw for field, raw in value.items()}
