"""Prepare reviewed, training-seed-only lexical repair groups without new IDs."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from scripts.generalization import evaluation_text
from scripts.semantic_keys import core_key


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(data, annotations, output):
    from sudachipy import Dictionary, SplitMode
    from scripts.prepare_tatoeba import analyze
    from scripts.prepare_teacher_prose import orthographic_readings
    data, annotations, output = Path(data), Path(annotations), Path(output)
    if output.exists():
        raise ValueError('repair output must be a new directory')
    manifest = json.loads((data/'manifest.json').read_text(encoding='utf-8'))
    vocab = json.loads((data/'vocab.json').read_text(encoding='utf-8'))
    if sha(data/'vocab.json') != manifest['vocab_sha256']:
        raise ValueError('vocabulary hash mismatch')
    key = lambda s: (s['surface'], s['kana'], s['pos'])
    mapping = {key(t):t['id'] for t in vocab['tokens'][7:]}
    protected = set()
    for filename, expected in manifest['split_sha256'].items():
        if sha(data/filename) != expected:
            raise ValueError('parent split hash mismatch')
        if filename != 'train.jsonl':
            protected.update(core_key(evaluation_text(json.loads(line)['text']))
                             for line in (data/filename).read_text(encoding='utf-8').splitlines())
    source_paths = dict(quality=Path('build/v12-training-rollouts.json'), ce=Path('build/v12-ce-training-rollouts.json'))
    reports = {name:json.loads(path.read_text(encoding='utf-8')) for name,path in source_paths.items()}
    for name, start, count in [('quality',810000,256), ('ce',811000,384)]:
        r = reports[name]
        if r['seed_start'] != start or r['requested'] != count or r['history_size'] != 0:
            raise ValueError('wrong training seed source')
        if {x['seed'] for x in r['samples']} != set(range(start,start+count)):
            raise ValueError('training rollouts incomplete')
    tokenizer = Dictionary().tokenizer()
    stats, groups = Counter(), []
    edits = json.loads(annotations.read_text(encoding='utf-8'))
    for source, index, prefix, suffixes in edits:
        row = reports[source]['samples'][index]
        text = ''
        prefix_ids = [0, 3 if len(row['kana'])<=12 else 4 if len(row['kana'])<=22 else 5]
        for segment in row['segments']:
            if text == prefix:
                break
            text += segment['surface']
            prefix_ids.append(mapping[key(segment)])
        if text != prefix or not row['text'].startswith(prefix):
            stats['prefix_not_token_boundary'] += 1
            print(json.dumps(dict(rejected=index,source=source,reason='prefix boundary'),ensure_ascii=False))
            continue
        offset = len(prefix_ids)-2
        rejected = mapping[key(row['segments'][offset])] if offset < len(row['segments']) else 1
        accepted = []
        for suffix in suffixes:
            positive = prefix + suffix
            if core_key(evaluation_text(positive)) in protected:
                stats['heldout_body_excluded'] += 1
                continue
            segments = analyze(positive, tokenizer, SplitMode.C)
            if segments is None:
                stats['dictionary_rejected'] += 1
                continue
            segments = orthographic_readings(segments)
            if any(key(s) not in mapping for s in segments):
                stats['out_of_vocabulary'] += 1
                continue
            ids = [0,prefix_ids[1],*[mapping[key(s)] for s in segments],1]
            if ids[:len(prefix_ids)] != prefix_ids or len(ids)>64:
                stats['retokenized_prefix_changed'] += 1
                continue
            if ids[len(prefix_ids)] == rejected:
                stats['rejected_token_still_positive'] += 1
                continue
            accepted.append(dict(text=positive,ids=ids))
        unique = {tuple(p['ids']):p for p in accepted}
        if len(unique)<2:
            stats['too_few_alternatives'] += 1
            continue
        group = dict(source=source,source_seed=row['seed'],original=row['text'],prefix=prefix,
                     prefix_ids=prefix_ids,rejected_token=rejected,positives=list(unique.values()),
                     reviewer='Codex; model-authored and model-reviewed, not independent human review')
        groups.append(group)
    train, validation = [], []
    for group in groups:
        bucket = int(hashlib.sha256(group['prefix'].encode()).hexdigest()[:8],16)%5
        (validation if bucket==0 else train).append(group)
    if not train or not validation:
        raise ValueError('need both training and held-out repair groups')
    output.mkdir(parents=True)
    checksums = {}
    for split, rows in [('train',train),('validation',validation)]:
        path = output/(split+'.jsonl')
        path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
        checksums[path.name] = sha(path)
    result = dict(version=1,review='Codex; not independent human review',vocab_sha256=sha(data/'vocab.json'),
                  parent_manifest_sha256=sha(data/'manifest.json'),annotation_sha256=sha(annotations),
                  source_script_sha256=sha(__file__),sources={name:dict(path=str(path),sha256=sha(path),
                  seed_start=reports[name]['seed_start'],seed_end=reports[name]['seed_end']) for name,path in source_paths.items()},
                  split_sha256=checksums,groups=dict(train=len(train),validation=len(validation)),
                  positive_sequences=dict(train=sum(len(g['positives']) for g in train),validation=sum(len(g['positives']) for g in validation)),
                  exclusions=dict(stats),inference_use='none; no repair table or corpus shipped')
    (output/'manifest.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', default='build/everyday-v11-data')
    p.add_argument('--annotations', default='data/repairs-v12/edits.json')
    p.add_argument('--output', default='build/everyday-v12-repairs')
    a = p.parse_args()
    print(json.dumps(prepare(a.data,a.annotations,a.output),ensure_ascii=False))
