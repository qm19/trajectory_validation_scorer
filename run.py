"""Convenience entry: edit settings.py, then run python run.py."""
from pathlib import Path

from trajectory_scorer.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["annotate", "--config", str(Path(__file__).with_name("settings.py"))]))
