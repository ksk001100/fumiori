"""Authored-frame decoding domain, explicitly independent of the training split.

This contains held-out combinations too. It is a hybrid-generation constraint,
not evidence that the neural model learned those combinations unaided.
"""
import json
from pathlib import Path


def attach_constraints(directory, rows):
    directory = Path(directory)
    vocab = json.loads((directory / 'vocab.json').read_text(encoding='utf-8'))
    config = json.loads((directory / 'model-config.json').read_text(encoding='utf-8'))
    ids = {(t['surface'], t['kana'], t['pos']): t['id'] for t in vocab['tokens'][7:]}
    sequences, skipped_oov, skipped_structural = set(), 0, 0
    relations = {}
    kinds = {}
    for row in rows:
        segments = row['segments']
        keys = [(s['surface'], s['kana'], s['pos']) for s in segments]
        if any(key not in ids for key in keys):
            skipped_oov += 1
            continue
        if (not row['text'].endswith('。') or row['text'].count('。') != 1
                or len(segments) < 2 or segments[-2]['pos'] in ('助詞', '名詞')
                or any(a['pos'] == b['pos'] == '助詞' for a, b in zip(segments, segments[1:]))
                or not 5 <= len(row['kana']) <= 40 or len(segments) + 3 > config['max_seq_len']):
            skipped_structural += 1
            continue
        difficulty = 3 if len(row['kana']) <= 12 else 4 if len(row['kana']) <= 22 else 5
        sequence = (difficulty, *(ids[key] for key in keys), 1)
        sequences.add(sequence)
        g = row.get('generation') or {}
        relations[sequence] = g.get('event_family', g.get('family', row['text']))
        kinds[sequence] = ['prose','idiom','proverb','dialogue','classic'].index(g.get('topic')) if g.get('topic') in ['idiom','proverb','dialogue','classic'] else 0
    if not sequences:
        raise ValueError('no eligible authored frames')
    constraints = dict(version='semantic_frames_v1', origin='authored slot rules, not training-reference exclusion',
                       completion_prior=1.0,
                       includes_heldout_combinations=True, sequences=sorted(sequences),
                       skipped_oov=skipped_oov, skipped_structural=skipped_structural)
    if any((r.get('generation') or {}).get('event_family') for r in rows):
        labels = sorted(set(relations.values()))
        label_ids = {label: i for i, label in enumerate(labels)}
        constraints.update(relation_labels=labels, relation_ids=[label_ids[relations[tuple(seq)]] for seq in constraints['sequences']], relation_history_size=24, probability_mix=0.35,
            kind_labels=['prose','idiom','proverb','dialogue','classic'], kind_weights=[.75,.1,.1,.04,.01],
            kind_ids=[kinds[tuple(seq)] for seq in constraints['sequences']])
    manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
    manifest['generation_constraints'] = constraints
    if any(r.get('generation', {}).get('topic') == 'classic' for r in rows):
        manifest['data_sources'] = [dict(author='夏目漱石', work='吾輩は猫である', license='public-domain', url='https://www.aozora.gr.jp/cards/000148/files/789_14547.html'), dict(source='Traditional Japanese idioms/proverbs; agent-authored readings', license='public-domain'), dict(source='Original relations and dialogue', license='CC0-1.0')]

    payload = json.dumps(manifest, ensure_ascii=False, separators=(',', ':')).encode()
    if len(payload) > 1_048_576:
        raise ValueError('frame manifest exceeds runtime byte limit')
    (directory / 'manifest.json').write_bytes(payload)
    return dict(frames=len(sequences), manifest_bytes=len(payload), skipped_oov=skipped_oov, skipped_structural=skipped_structural)


