"""Audit the current standard model and its documented generation command."""
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.generalization import temporal_prefixes


def measure(cli, directory, difficulty, count):
    output = subprocess.check_output([str(cli), 'generate', '--model', str(directory), '--count', str(count),
        '--seed', '3432', '--difficulty', difficulty], encoding='utf-8')
    rows = [json.loads(line) for line in output.splitlines()]
    manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
    decoder_hash = hashlib.sha256(json.dumps(manifest.get('generation_constraints'), sort_keys=True).encode()).hexdigest()
    return dict(model=str(directory), weights_sha256=manifest['weights_sha256'], decoder_sha256=decoder_hash, difficulty=difficulty,
                seed=3432, count=count, temporal_prefix_count=sum(any(r['text'].startswith(p) for p in temporal_prefixes()) for r in rows),
                unique_texts=len({r['text'] for r in rows}), sources=dict(Counter(r['source'] for r in rows)),
                initial_tokens=dict(Counter(r['segments'][0]['surface'] for r in rows)), samples=rows)


def main():
    cli = Path('target/release/fumiori-cli' + ('.exe' if os.name == 'nt' else '')).resolve()
    current = json.loads(Path('build/model/manifest.json').read_text(encoding='utf-8'))
    activation = json.loads(Path('build/current-model.json').read_text(encoding='utf-8'))
    source = Path('build') / activation.get('fp32_source', 'stories-v2-diverse-3mb')
    source_manifest = json.loads((source / 'manifest.json').read_text(encoding='utf-8'))
    if any(current.get(key) != source_manifest.get(key) for key in
           ('weights_sha256', 'generation_constraints', 'vocab_sha256', 'generation_mode',
            'allow_static_fallback','generation_defaults','lexical_boundary_guard','polite_boundary_guard','reading_normalization')) or any(
            (Path('build/model') / name).read_bytes() != (source / name).read_bytes()
            for name in ('model-config.json', 'vocab.json')):
        raise ValueError('build/model is stale; run python -m training.activate')
    result = dict(normal=measure(cli, Path('build/model'), 'normal', 10),
                  mixed=measure(cli, Path('build/model'), 'mixed', 10),
                  larger_sample=measure(cli, Path('build/model'), 'mixed', 200))
    Path('docs/current-model-check.json').write_bytes(json.dumps(result, ensure_ascii=False, indent=2).encode())
    print(json.dumps({name: {k: v for k, v in record.items() if k != 'samples'} for name, record in result.items()}, ensure_ascii=True))


if __name__ == '__main__':
    main()
