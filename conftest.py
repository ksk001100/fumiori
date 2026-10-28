"""Keep pytest temporary files usable on a fresh checkout."""
from pathlib import Path


def pytest_configure(config):
    # pytest creates basetemp itself, but requires its parent to exist.
    if config.option.basetemp is not None:
        Path(config.option.basetemp).parent.mkdir(parents=True, exist_ok=True)
