"""Measure v1 generation without filtering memorized or unsupported outputs."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
from scripts.semantic_keys import core_key
from scripts.generalization import evaluation_text
from training.train import read_rows


def runtime_identity(cli, root=Path('.')):
    sources = [root / 'Cargo.toml', root / 'Cargo.lock',
               *sorted((root / 'crates').glob('*/Cargo.toml')),
               *sorted((root / 'crates').glob('*/src/**/*.rs'))]
    if any(path.stat().st_mtime_ns > cli.stat().st_mtime_ns for path in sources):
        raise ValueError('evaluation CLI is older than Rust sources; run cargo build --release --locked -p fumiori-cli')
    return dict(cli_sha256=hashlib.sha256(cli.read_bytes()).hexdigest(),
                rust_source_sha256={p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})


def audit(samples, train, domain, requested):
    texts = {evaluation_text(r['text']) for r in train}
    cores = {core_key(t) for t in texts}
    supported = {(evaluation_text(r['text']), r['kana']) for r in domain}
    model = [r for r in samples if r['source'] == 'model']
    exact = sum(evaluation_text(r['text']) in texts for r in model)
    novel = [r for r in model if core_key(evaluation_text(r['text'])) not in cores]
    novel_supported = [r for r in novel if (evaluation_text(r['text']), r['kana']) in supported]
    supported_count = sum((evaluation_text(r['text']), r['kana']) in supported for r in model)
    return dict(requested=requested, returned=len(samples), model_count=len(model),
        sources=dict(Counter(r['source'] for r in samples)), training_exact_count=exact,
        training_exact_rate=exact / requested, novel_core_count=len(novel),
        novel_core_rate=len(novel) / requested, author_supported_count=supported_count,
        author_supported_rate=supported_count / requested,
        novel_author_supported_count=len(novel_supported), novel_author_supported_rate=len(novel_supported) / requested,
        unique_novel_author_supported=len({r['text'] for r in novel_supported}),
        outside_reference_count=len(model) - supported_count,
        unique_texts=len({r['text'] for r in model}),
        unique_ratio=len({r['text'] for r in model}) / requested,
        warning='author-support is exact offline reference membership, not an independent quality rating; outside-reference texts are unjudged')


def measure(cli, directory, train, domain, seeds, temperature=.8, top_p=.9, report_dir=Path('build/stories-v1-3mb-evaluation'), skip_near_statistics=False):
    identity = runtime_identity(cli)
    report_dir.mkdir(parents=True, exist_ok=True)
    samples, runs = [], []
    for seed in seeds:
        # There is no filtering by train membership, support or quality after generation.
        command = [str(cli), 'evaluate', '--model', str(directory), '--samples', '200',
            '--seed', str(seed), '--history-size', '16', '--unconstrained',
            '--temperature', str(temperature), '--top-p', str(top_p),
            '--training-data', str(train), '--report', str(report_dir / f'{directory.name}-{seed}.json')]
        if skip_near_statistics:
            command.append('--skip-near-statistics')
        output = subprocess.check_output(command, encoding='utf-8')
        result = json.loads(output)
        runs.append(dict(seed=seed, failures=result['failures'], success=result['success']))
        samples.extend(result['samples'])
    return dict(runtime_identity=identity, runs=runs, metrics=audit(samples, read_rows(train), domain, 200 * len(seeds)), samples=samples)
