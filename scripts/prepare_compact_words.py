"""Retokenize an existing segmented corpus into train-only paired word IDs.

No new sentences, sentence rules, or held-out vocabulary enter training.
"""
import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path


def prepare(source, output):
    source, output = Path(source), Path(output)
    if output.exists():
        raise ValueError('output must be a new directory')
    parent_bytes = (source / 'manifest.json').read_bytes()
    parent = json.loads(parent_bytes)
    parts = {}
    for name, expected in parent['split_sha256'].items():
        payload = (source / name).read_bytes()
        if hashlib.sha256(payload).hexdigest() != expected:
            raise ValueError(f'parent split hash mismatch: {name}')
        parts[name] = [json.loads(line) for line in payload.decode().splitlines() if line]
    key = lambda s: (s['surface'], s['kana'], s['pos'])
    counts = Counter(key(s) for r in parts['train.jsonl'] for s in r['segments'])
    tokens = [dict(id=i, surface=s, kana='', pos='special') for i, s in enumerate(
        ['BOS', 'EOS', 'PAD', 'EASY', 'NORMAL', 'HARD', 'UNK'])]
    for word in sorted(counts, key=lambda k: (-counts[k], k)):
        tokens.append(dict(id=len(tokens), surface=word[0], kana=word[1], pos=word[2],
                           metadata=dict(count=counts[word])))
    if len(tokens) > 16000:
        raise ValueError('word vocabulary exceeds runtime limit')
    ids = {key(t): t['id'] for t in tokens[7:]}
    vocab = json.dumps(dict(version='0.1.0', encoding='UTF-8', tokens=tokens),
                       ensure_ascii=False, separators=(',', ':')).encode()
    result = copy.deepcopy(parent)
    result.update(description='Train-only paired word vocabulary for compact neural generation',
                  parent_manifest_sha256=hashlib.sha256(parent_bytes).hexdigest(),
                  source_generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  vocab_sha256=hashlib.sha256(vocab).hexdigest(), splits={}, split_sha256={},
                  compact_word_stats={})
    prepared = {}
    for name, rows in parts.items():
        accepted, rejected = [], Counter()
        for row in rows:
            if ''.join(s['surface'] for s in row['segments']) != row['text'] or ''.join(s['kana'] for s in row['segments']) != row['kana']:
                raise ValueError('segmentation roundtrip mismatch')
            if any(key(s) not in ids for s in row['segments']):
                rejected['unseen_word'] += 1
                continue
            sequence = [0, row['ids'][1], *[ids[key(s)] for s in row['segments']], 1]
            if len(sequence) > 64:
                rejected['too_long'] += 1
                continue
            accepted.append(dict(row, ids=sequence))
        payload = ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in accepted).encode()
        prepared[name] = payload
        result['splits'][Path(name).stem] = len(accepted)
        result['split_sha256'][name] = hashlib.sha256(payload).hexdigest()
        result['compact_word_stats'][name] = dict(input=len(rows), kept=len(accepted), rejected=dict(rejected))
    output.mkdir(parents=True)
    (output / 'vocab.json').write_bytes(vocab)
    for name, payload in prepared.items():
        (output / name).write_bytes(payload)
    (output / 'manifest.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    return dict(words=len(tokens)-7, splits=result['splits'], stats=result['compact_word_stats'])


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    print(json.dumps(prepare(a.source, a.output), ensure_ascii=False))
