"""Reproducible Phase-1 baseline and separately attributed review collection."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import subprocess

from scripts.novelty_metrics import TrainingIndex, normalized, split_audit


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def review_selection(rows, count, seed):
    return [dict(sample_id=i, sample_sha256=hashlib.sha256(
        json.dumps(rows[i], ensure_ascii=False, sort_keys=True).encode()).hexdigest(), **rows[i])
        for i in sorted(random.Random(seed).sample(range(len(rows)), min(count, len(rows))))]


def summarize_review(selected, ratings):
    """Reject changed samples and mixed reviewer attribution; never fill missing labels."""
    expected = {str(r['sample_id']): r for r in selected}
    seen, rated = set(), []
    for row in ratings:
        identity = row['sample_id']
        if identity in seen or identity not in expected:
            raise ValueError('unexpected or duplicate review sample')
        seen.add(identity)
        target = expected[identity]
        if any(row[k] != str(target[k]) for k in ('sample_sha256', 'text', 'kana', 'status')):
            raise ValueError('review sample was modified')
        labels = [row.get(k, '') for k in ('natural', 'reading_ok')]
        if not any(labels):
            continue
        if labels not in [['0', '0'], ['0', '1'], ['1', '0'], ['1', '1']]:
            raise ValueError('both labels must be 0 or 1')
        if target['status'] != 'success':
            raise ValueError('failed requests cannot be rated as generated text')
        if not row.get('reviewer') or row.get('reviewer_kind') not in ('independent_human', 'agent'):
            raise ValueError('reviewer and explicit reviewer_kind required')
        if not row.get('notes', '').strip():
            raise ValueError('review rationale required, including ambiguous readings')
        rated.append(row)
    if seen != expected.keys():
        raise ValueError('review file must retain every selected request')
    groups = {}
    for kind in ('independent_human', 'agent'):
        rows = [r for r in rated if r['reviewer_kind'] == kind]
        groups[kind] = dict(rated=len(rows), natural_rate=sum(r['natural'] == '1' for r in rows)/len(rows) if rows else None,
            reading_accuracy=sum(r['reading_ok'] == '1' for r in rows)/len(rows) if rows else None,
            joint_rate=sum(r['natural'] == r['reading_ok'] == '1' for r in rows)/len(rows) if rows else None)
    return dict(selected=len(selected), unrated=len(selected)-len(rated), by_reviewer_kind=groups,
                scope='Rates among explicitly rated successes only; incomplete review is not a quality gate')


def validate_requests(raw, seed, requested):
    successes, failures = raw['samples'], raw['failures']
    ids = [r['seed'] for r in successes+failures]
    if (raw['requested'] != requested or raw['success'] != len(successes)
            or len(ids) != requested or set(ids) != set(range(seed, seed+requested))):
        raise ValueError('every requested seed must have exactly one outcome')
    if raw['history_size'] != 0 or any(r['source'] != 'model' for r in successes):
        raise ValueError('baseline requires history=0 and model outputs only')


def run(config_path, output):
    from scripts.open_generation import runtime_identity
    config_path, output = Path(config_path), Path(output)
    config = json.loads(config_path.read_text(encoding='utf-8'))
    if output.exists():
        raise ValueError('output must be a new directory; existing results are preserved')
    model, data = Path(config['model']), Path(config['data'])
    manifest = json.loads((data/'manifest.json').read_text(encoding='utf-8'))
    if sha(data/'manifest.json') != config['data_manifest_sha256']:
        raise ValueError('unexpected data manifest')
    for name, expected in manifest['split_sha256'].items():
        if sha(data/name) != expected:
            raise ValueError('split hash mismatch')
    if sha(data/'vocab.json') != manifest['vocab_sha256']:
        raise ValueError('data vocabulary hash mismatch')
    model_manifest = json.loads((model/'manifest.json').read_text(encoding='utf-8'))
    for name, field in [('model.safetensors', 'weights_sha256'), ('vocab.json', 'vocab_sha256'),
                        ('model-config.json', 'config_sha256')]:
        if sha(model/name) != model_manifest[field]:
            raise ValueError('model hash mismatch')
    if model_manifest['weights_sha256'] != config['weights_sha256']:
        raise ValueError('unexpected model weights')
    if model_manifest.get('allow_static_fallback') is not False or 'generation_constraints' in model_manifest:
        raise ValueError('baseline requires autoregressive model without sentence constraints')
    cli = Path('target/release/fumiori-cli'+('.exe' if os.name == 'nt' else '')).resolve()
    identity = runtime_identity(cli)
    splits = {name.removesuffix('.jsonl'): read_rows(data/name) for name in manifest['split_sha256']}
    split_report = split_audit(splits)
    if not split_report['exact_disjoint']:
        raise ValueError('evaluation splits overlap; inspect with scripts.novelty_metrics.split_audit')
    index = TrainingIndex(splits['train'])
    output.mkdir(parents=True)
    results = []
    for condition in config['conditions']:
        name = condition['name']
        if not name or Path(name).name != name or name in ('.', '..') or '/' in name or '\\' in name:
            raise ValueError('condition name must be a directory name')
        directory = output/name
        directory.mkdir()
        command = [str(cli), 'evaluate-seeds', '--model', str(model), '--samples', str(condition['samples']),
                   '--seed', str(condition['seed']), '--difficulty', condition['difficulty'],
                   '--temperature', str(condition['temperature']), '--top-p', str(condition['top_p']),
                   '--report', str(directory/'raw.json')]
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
        raw = json.loads((directory/'raw.json').read_text(encoding='utf-8'))
        validate_requests(raw, condition['seed'], condition['samples'])
        metrics = index.evaluate(raw['samples'], config['near_threshold'])
        successes = {r['seed']: r for r in raw['samples']}
        failures = {r['seed']: r for r in raw['failures']}
        requests = []
        for seed in range(condition['seed'], condition['seed']+condition['samples']):
            if seed in successes:
                requests.append(dict(status='success', **successes[seed]))
            else:
                requests.append(dict(status='failure', text='', kana='', **failures[seed]))
        selected = review_selection(requests, config['review_count'], config['review_seed'])
        with (directory/'review.csv').open('w', encoding='utf-8-sig', newline='') as file:
            fields = ['sample_id', 'sample_sha256', 'status', 'text', 'kana', 'natural', 'reading_ok',
                      'reviewer', 'reviewer_kind', 'notes']
            writer = csv.DictWriter(file, fieldnames=fields, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(selected)
        structural = sum(''.join(s['surface'] for s in r['segments']) == r['text']
                         and ''.join(s['kana'] for s in r['segments']) == r['kana'] for r in raw['samples'])
        result = dict(condition=condition, requested=raw['requested'], success=raw['success'], failures=raw['failures'],
            metrics=metrics, latency=dict(p50_ms=raw['p50_ms'], p95_ms=raw['p95_ms'],
                scope='CPU generate + serialization + result bookkeeping; all requests including failures; '
                      'first generation included; load/download/evaluation excluded; not WASM'),
            structural_roundtrip_count=structural, naturalness=None, contextual_reading_accuracy=None,
            review_selection=selected, review_status='pending; structural roundtrip is not reading accuracy',
            raw_sha256=sha(directory/'raw.json'), environment=raw['environment'])
        write_json(directory/'report.json', result)
        results.append(result)
    report = dict(schema_version=1, config=config, config_sha256=sha(config_path),
        evaluator_sha256={name: sha(Path('scripts')/name) for name in ('novelty_metrics.py', 'novelty_evaluate.py')},
        runtime_identity=identity, python=platform.python_version(),
        data_manifest=manifest, data_manifest_sha256=sha(data/'manifest.json'), split_audit=split_report,
        model_sha256={name: sha(model/name) for name in ('manifest.json', 'model.safetensors', 'model-config.json', 'vocab.json', 'LICENSE')},
        model_bytes=sum((model/name).stat().st_size for name in ('manifest.json', 'model.safetensors', 'model-config.json', 'vocab.json', 'LICENSE')),
        protocol=dict(normalization='NFKC and terminal Japanese period removal; retain all other wording',
            near_duplicate='maximum training character-bigram set Dice >= threshold; includes exact matches; not paraphrase detection',
            lcs='longest contiguous substring within one training sentence; not subsequence',
            ngram='occurrence-micro and type novelty n=1..4; word=existing lexical surface segments excluding punctuation',
            self_bleu='leave-one-observation-out sentence BLEU4; effective order; add-one smoothing n>=2; [0,1]',
            denominators='novelty/diversity among returned model outputs; requested/success/failures reported separately',
            limitations='No metric certifies semantics, naturalness or contextual reading; all task training sources must be included for future distilled models'),
        results=results)
    write_json(output/'report.json', report)
    with (output/'summary.csv').open('w', encoding='utf-8-sig', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['condition', 'requested', 'success', 'exact', 'near', 'lcs_fraction', 'word_self_bleu4', 'p50_ms', 'p95_ms'])
        for result in results:
            m = result['metrics']
            writer.writerow([result['condition']['name'], result['requested'], result['success'], m['exact_match_rate'],
                m['near_duplicate_rate'], m['mean_lcs_fraction'], m['self_bleu4']['word'], result['latency']['p50_ms'], result['latency']['p95_ms']])
    print(json.dumps(dict(output=str(output), conditions=len(results), model_bytes=report['model_bytes'])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    generation = sub.add_parser('run')
    generation.add_argument('--config', required=True)
    generation.add_argument('--output', required=True)
    review = sub.add_parser('review')
    review.add_argument('--report', required=True)
    review.add_argument('--ratings', required=True)
    review.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.command == 'run':
        run(args.config, args.output)
    else:
        if Path(args.output).exists():
            raise ValueError('review output already exists')
        report = json.loads(Path(args.report).read_text(encoding='utf-8'))
        with Path(args.ratings).open(encoding='utf-8-sig', newline='') as file:
            result = summarize_review(report['review_selection'], list(csv.DictReader(file)))
        write_json(args.output, dict(report_sha256=sha(args.report), ratings_sha256=sha(args.ratings), **result))


if __name__ == '__main__':
    main()
