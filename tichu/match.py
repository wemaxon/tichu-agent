from __future__ import annotations

from dataclasses import dataclass, field, replace
from .config import TichuRoundConfig
from .round import TichuRound


@dataclass(frozen=True, slots=True)
class TichuMatch:
    """Immutable cumulative match state for standard play to 1,000 points."""

    target_score: int = 1000
    scores: tuple[int, int] = (0, 0)
    dealer: int = 0
    round_number: int = 0
    round_config: TichuRoundConfig = field(default_factory=TichuRoundConfig)
    rounds: tuple[TichuRound, ...] = ()

    def __post_init__(self) -> None:
        scores = tuple(self.scores)
        if len(scores) != 2:
            raise ValueError("Match scores must contain exactly two team scores")
        object.__setattr__(self, "scores", (int(scores[0]), int(scores[1])))
        object.__setattr__(self, "rounds", tuple(self.rounds))

    @property
    def terminal(self) -> bool:
        if max(self.scores) < self.target_score:
            return False
        return self.scores[0] != self.scores[1]

    @property
    def winning_team(self) -> int | None:
        if not self.terminal:
            return None
        return 0 if self.scores[0] > self.scores[1] else 1

    def new_round(self, seed: int) -> TichuRound:
        if self.terminal:
            raise RuntimeError("The match has already ended")
        return TichuRound.new(seed=seed, dealer=self.dealer, config=self.round_config)

    def record_round(self, round_: TichuRound) -> "TichuMatch":
        if not round_.terminal or round_.round_scores is None:
            raise ValueError("Only a completed round can be recorded")
        scores = (
            self.scores[0] + round_.round_scores[0],
            self.scores[1] + round_.round_scores[1],
        )
        dealer = round_.finish_order[0] if round_.finish_order else self.dealer
        return replace(
            self,
            scores=scores,
            rounds=self.rounds + (round_,),
            round_number=self.round_number + 1,
            dealer=dealer,
        )
