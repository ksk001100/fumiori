"""Measure seed collisions without history, output filtering, or sentence templates."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess

from scripts.semantic_keys import core_key
from scripts.generalization import evaluation_text


def summarize(report, training_texts):
    requested = report['requested']
    samples = report['samples']
    if requested < 1 or len(samples) != report['success']:
        raise ValueError('invalid independent-seed report counts')
    if report['history_size'] != 0:
        raise ValueError('independent-seed evaluation requires history_size=0')
    seeds = [row['seed'] for row in samples]
    if len(set(seeds)) != len(seeds):
        raise ValueError('each output must have a different seed')
    failed_seeds = [row['seed'] for row in report['failures']]
    if (len(seeds) + len(failed_seeds) != requested
            or len(set(seeds + failed_seeds)) != requested
            or set(seeds + failed_seeds) != set(range(report['seed_start'], report['seed_start'] + requested))):
        raise ValueError('seed range must account for every success and failure exactly once')
    def normalized(text):
        return core_key(evaluation_text(text))

    known = {normalized(text) for text in training_texts}

    def metrics(rows, denominator):
        texts = Counter(row['text'] for row in rows)
        cores = Counter(normalized(row['text']) for row in rows)
        return dict(requested=denominator, success=len(rows),
                    unique_texts=len(texts), unique_ratio=len(texts) / denominator,
                    duplicate_count=len(rows) - len(texts),
                    unique_cores=len(cores), core_duplicate_count=len(rows) - len(cores),
                    novel_core_count=sum(normalized(row['text']) not in known for row in rows),
                    novel_core_rate=sum(normalized(row['text']) not in known for row in rows) / denominator)

    result = metrics(samples, requested)
    result['by_difficulty'] = {}
    # Failures retain their requested seed; the cycle is defined by the CLI.
    levels = report.get('difficulty_cycle', ['easy', 'normal', 'hard'])
    for i, level in enumerate(levels):
        count = len(range(i, requested, len(levels)))
        if count:
            rows = [r for r in samples if (r['seed'] - report['seed_start']) % len(levels) == i]
            result['by_difficulty'][level] = metrics(rows, count)
    result['novelty_scope'] = 'Normalized body absent from task-training file; upstream pretraining membership is unknown. Quality is unjudged.'
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cli', type=Path, default=Path('target/release/fumiori-cli' + ('.exe' if os.name == 'nt' else '')))
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--training-data', type=Path, required=True)
    parser.add_argument('--comparison-training-data', type=Path, help='Also measure against an unchanged baseline training reference')
    parser.add_argument('--samples', type=int, default=600)
    parser.add_argument('--seed', type=int, default=420000)
    parser.add_argument('--temperature', type=float, default=.5)
    parser.add_argument('--top-p', type=float, default=.9)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    # Import here to keep summary-only use free of torch dependencies.
    from scripts.open_generation import runtime_identity
    identity = runtime_identity(args.cli.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    raw = args.output.with_suffix('.raw.json')
    subprocess.run([str(args.cli.resolve()), 'evaluate-seeds', '--model', str(args.model),
                    '--samples', str(args.samples), '--seed', str(args.seed),
                    '--temperature', str(args.temperature), '--top-p', str(args.top_p),
                    '--report', str(raw)], check=True, stdout=subprocess.DEVNULL)
    report = json.loads(raw.read_text(encoding='utf-8'))
    report['difficulty_cycle'] = ['easy', 'normal', 'hard']
    files = ['manifest.json', 'model-config.json', 'model.safetensors', 'vocab.json']
    report['model_sha256'] = {name: hashlib.sha256((args.model / name).read_bytes()).hexdigest() for name in files}
    manifest = json.loads((args.model / 'manifest.json').read_text(encoding='utf-8'))
    decoder = {k: manifest.get(k) for k in ('generation_constraints', 'generation_mode', 'allow_static_fallback',
               'lexical_boundary_guard', 'polite_boundary_guard', 'reading_normalization')}
    for key in ('subword_start_temperature', 'subword_difficulty_prefix'):
        if key in manifest:
            decoder[key] = manifest[key]
    report['decoder_sha256'] = hashlib.sha256(json.dumps(decoder, sort_keys=True).encode()).hexdigest()
    report['runtime_identity'] = identity
    payload = args.training_data.read_bytes()
    report['training_file_sha256'] = hashlib.sha256(payload).hexdigest()
    report['metrics'] = summarize(report, [json.loads(line)['text'] for line in payload.decode('utf-8').splitlines()])
    if args.comparison_training_data is not None:
        comparison = args.comparison_training_data.read_bytes()
        report['common_reference'] = dict(path=str(args.comparison_training_data),
            sha256=hashlib.sha256(comparison).hexdigest(),
            metrics=summarize(report,[json.loads(line)['text'] for line in comparison.decode('utf-8').splitlines()]))
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report['metrics'], ensure_ascii=False))


if __name__ == '__main__':
    main()
