"""Matched A/B experiments; training references are used only AFTER generation."""
import argparse
import csv
import json
import os
from pathlib import Path
import random
import subprocess

from scripts.build_wasm import verify_runtime, RUNTIME_FILES
from scripts.novelty_evaluate import read_rows, review_selection, sha, write_json
from scripts.novelty_metrics import TrainingIndex, split_audit, normalized
from scripts.open_generation import runtime_identity


def validate_pool_report(raw, requested):
    for variant in raw['variants']:
        ids = [r['request_id'] for r in variant['samples']+variant['failures']]
        if (len(ids) != requested or set(ids) != set(range(requested))
                or len(variant['samples']) != variant['success']):
            raise ValueError('candidate report must account for all requests exactly once')


def load_prepared(data):
    data=Path(data)
    manifest=json.loads((data/'manifest.json').read_text(encoding='utf-8'))
    for filename,expected in manifest['split_sha256'].items():
        if sha(data/filename)!=expected:raise ValueError('split checksum mismatch')
    if sha(data/'vocab.json')!=manifest['vocab_sha256']:raise ValueError('data vocabulary mismatch')
    parts={name.removesuffix('.jsonl'):read_rows(data/name) for name in manifest['split_sha256']}
    audit=split_audit(parts)
    if not audit['exact_disjoint']:raise ValueError('overlapping evaluation splits')
    return manifest,parts,audit


def validate_common_reference(reference,training_groups):
    expected={normalized(r['text']) for rows in training_groups for r in rows}
    actual={normalized(r['text']) for r in reference}
    if actual!=expected:
        raise ValueError('common reference must equal the union of actual condition training texts')


def run(config_path, output):
    config = json.loads(Path(config_path).read_text(encoding='utf-8'))
    output = Path(output)
    if output.exists():
        raise ValueError('output must be a new directory')
    cli = Path('target/release/fumiori-cli'+('.exe' if os.name == 'nt' else '')).resolve()
    identity = runtime_identity(cli)
    data = Path(config['data'])
    datasets={str(data):load_prepared(data)}
    for condition in config['conditions']:
        directory=str(Path(condition.get('data',str(data))))
        if directory not in datasets:datasets[directory]=load_prepared(directory)
    reference_data=Path(config.get('common_reference_data',str(data)))
    if str(reference_data) not in datasets:datasets[str(reference_data)]=load_prepared(reference_data)
    reference_manifest,reference_parts,_=datasets[str(reference_data)]
    validate_common_reference(reference_parts['train'],[datasets[str(Path(c.get('data',str(data))))][1]['train'] for c in config['conditions']])
    if any(m['vocab_sha256']!=reference_manifest['vocab_sha256'] for m,_,_ in datasets.values()):
        raise ValueError('comparison requires one frozen vocabulary')
    manifest,parts,audit=datasets[str(data)]
    index = TrainingIndex(reference_parts['train'])
    output.mkdir(parents=True)
    results, review_cases, blind_map = [], [], []
    for condition in config['conditions']:
        model = Path(condition['model'])
        condition_data=Path(condition.get('data',str(data)))
        condition_manifest,condition_parts,_=datasets[str(condition_data)]
        verify_runtime(model)
        # The full export manifest binds the packaged weights to the exact train reference.
        runtime = json.loads((model/'manifest.json').read_text(encoding='utf-8'))
        stem = model.name.removesuffix('-3mb')
        export_manifest = Path('build')/(stem+'-int8')/'manifest.json'
        full = json.loads(export_manifest.read_text(encoding='utf-8'))
        if (sha(export_manifest) != runtime['source_manifest_sha256']
                or full['data_manifest']['split_sha256'] != condition_manifest['split_sha256']
                or full['data_manifest']['vocab_sha256'] != condition_manifest['vocab_sha256']):
            raise ValueError('runtime model does not match evaluation training reference')
        raw_path = output/(condition['name']+'.raw.json')
        subprocess.run([str(cli), 'evaluate-candidates', '--model', str(model), '--samples', str(config['samples']),
            '--seed', str(config['seed']), '--temperature', str(condition['temperature']), '--top-p', str(condition['top_p']),
            '--max-candidates', str(condition['max_candidates']), '--report', str(raw_path)], check=True, stdout=subprocess.DEVNULL)
        raw = json.loads(raw_path.read_text(encoding='utf-8'))
        validate_pool_report(raw, config['samples'])
        for variant in raw['variants']:
            name = f"{condition['name']}-n{variant['candidate_count']}"
            requests = [dict(status='failure', text='', kana='', request_id=i) for i in range(config['samples'])]
            for row in variant['samples']:
                requests[row['request_id']] = dict(status='success', **row)
            selected = review_selection(requests, config['review_count'], config['review_seed'])
            metrics = index.evaluate(variant['samples'], config['near_threshold'])
            own_texts={normalized(r['text']) for r in condition_parts['train']}
            result = dict(name=name, condition=condition, candidate_count=variant['candidate_count'],
                requested=config['samples'], success=variant['success'], failures=variant['failures'], metrics=metrics,
                p50_ms=variant['p50_ms'], p95_ms=variant['p95_ms'], latency_scope=raw['latency_scope'],
                normal_p95_ms=percentile([v for i,v in enumerate(variant['durations_ms']) if i % 3 == 1], .95),
                review_selection=selected, naturalness=None, contextual_reading_accuracy=None,
                raw_sha256=sha(raw_path), model_sha256={n:sha(model/n) for n in RUNTIME_FILES},
                model_bytes=sum((model/n).stat().st_size for n in RUNTIME_FILES), environment=raw['environment'])
            result['own_training_reference']=dict(path=str(condition_data),manifest_sha256=sha(condition_data/'manifest.json'),
                exact_match_rate=sum(normalized(r['text']) in own_texts for r in variant['samples'])/variant['success'] if variant['success'] else None)
            write_json(output/(name+'.json'), result)
            results.append(result)
            for row in selected:
                key = f"{name}:{row['sample_id']}"
                blind_map.append(dict(key=key, condition=name, sample_id=row['sample_id']))
                review_cases.append(dict(key=key, **row))
    random.Random(config['review_seed']+1).shuffle(review_cases)
    # Pool assignment remains in a separate mapping; the rating sheet hides model and N.
    with (output/'blind-review.csv').open('w', encoding='utf-8-sig', newline='') as file:
        fields = ['blind_id','status','text','kana','natural','reading_ok','reviewer','reviewer_kind','notes']
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(dict(blind_id=i, **row) for i,row in enumerate(review_cases))
    write_json(output/'review-map.json', [dict(blind_id=i,key=row['key']) for i,row in enumerate(review_cases)])
    report = dict(config=config, config_sha256=sha(config_path), runtime_identity=identity,
        code_sha256={n:sha(Path('scripts')/n) for n in ('evaluate_candidates.py','novelty_metrics.py','novelty_evaluate.py')},
        data_manifest_sha256=sha(data/'manifest.json'), data_manifest=manifest, split_audit=audit,
        common_reference=dict(path=str(reference_data),manifest_sha256=sha(reference_data/'manifest.json'),
            scope='Union of actual task-training texts of all compared conditions; no validation/test membership'),
        data_audits={directory:dict(manifest_sha256=sha(Path(directory)/'manifest.json'),split_audit=a) for directory,(_,_,a) in datasets.items()},
        protocol=raw['protocol'], quality='pending independent human review; student likelihood is not quality',
        review_seed=config['review_seed'], review_count=config['review_count'], results=results)
    write_json(output/'report.json', report)
    with (output/'summary.csv').open('w',encoding='utf-8-sig',newline='') as file:
        writer=csv.writer(file)
        writer.writerow(['condition','success','unique','exact','near','word_bleu','normal_p95_ms'])
        for r in results:
            m=r['metrics']
            writer.writerow([r['name'],r['success'],m['unique_texts'],m['exact_match_rate'],m['near_duplicate_rate'],m['self_bleu4']['word'],r['normal_p95_ms']])


def percentile(values, fraction):
    import math
    return sorted(values)[math.ceil((len(values)-1)*fraction)] if values else None


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/everyday-v14-evaluation.json')
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    run(args.config,args.output)
