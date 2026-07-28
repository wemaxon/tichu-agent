from __future__ import annotations

import os


def pytest_addoption(parser) -> None:
    parser.addoption(
        "--random-games",
        action="store",
        default=os.environ.get("TICHU_RANDOM_GAMES", "16"),
        help="Number of complete random rounds for tests marked 'random'.",
    )
