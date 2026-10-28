"""Canonical E2E targets autoregressive novelty, and fails rather than promoting a rejected model."""
import os
from pathlib import Path
import subprocess
import sys


def main():
    env = dict(os.environ, PYTHONUTF8='1', PYTHONIOENCODING='utf-8', CARGO_INCREMENTAL='0')
    commands = [
        ['cargo', 'fmt', '--all', '--', '--check'],
        ['cargo', 'clippy', '--workspace', '--all-targets', '--locked', '--', '-D', 'warnings'],
        ['cargo', 'test', '--workspace', '--locked'],
        [sys.executable, '-m', 'pytest', '-q'],
        [sys.executable, 'scripts/stories_pipeline.py', '--activate'],
    ]
    cli = str(Path('target/release/fumiori-cli' + ('.exe' if os.name == 'nt' else '')).resolve())
    commands += [[cli, 'parity', '--model', 'build/' + model] for model in ('model', 'model-int8')]
    commands.append([cli, 'generate', '--model', 'build/model', '--count', '10', '--seed', '3432'])
    for command in commands:
        subprocess.run(command, env=env, check=True)

if __name__ == "__main__":
    main()
