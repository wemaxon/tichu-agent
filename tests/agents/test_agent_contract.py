"""Tests for the public agent boundary."""

from __future__ import annotations

import subprocess
import sys


def test_agents_do_not_import_round_state() -> None:
    code = (
        "import sys; "
        "import agents; "
        "print('\\n'.join(sorted(name for name in sys.modules if name.startswith('tichu')))); "
        "raise SystemExit(1 if 'tichu.round' in sys.modules else 0)"
    )
    result = subprocess.run(
        [sys.executable, "-B", "-c", code],
        check=False,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
