from __future__ import annotations

from dataclasses import dataclass, field

from .config import TichuRoundConfig
from .round import TichuRound


@dataclass
class TichuMatch:
    """Cumulative match state for standard play to 1,000 points."""

    target_score: int = 1000
    scores: list[int] = field(default_factory=lambda: [0, 0])
    dealer: int = 0
    round_number: int = 0
    round_config: TichuRoundConfig = field(default_factory=TichuRoundConfig)
    rounds: list[TichuRound] = field(default_factory=list)

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

    def record_round(self, round_: TichuRound) -> None:
        if not round_.terminal or round_.round_scores is None:
            raise ValueError("Only a completed round can be recorded")
        self.scores[0] += round_.round_scores[0]
        self.scores[1] += round_.round_scores[1]
        self.rounds.append(round_)
        self.round_number += 1
        if round_.finish_order:
            self.dealer = round_.finish_order[0]
