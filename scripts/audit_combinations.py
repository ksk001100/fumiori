"""Audit lexical combination overlap and checkpoint preferences, offline only."""
import argparse
from collections import Counter
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def predicate_key(morphs, index):
    morph = morphs[index]
    if (morph.dictionary_form() == 'する' and index > 0
            and 'サ変可能' in morphs[index-1].part_of_speech()):
        return morphs[index-1].normalized_form() + 'する'
    return morph.normalized_form()


def lexical_triples(text, tokenizer, mode):
    """Conservative co-occurrence, NOT dependency parsing or semantic equivalence.

    Retain noun + adjacent case particle with every subsequent lexical verb in
    the same punctuation-delimited span. Dictionary forms cover inflection;
    normalized forms cover dictionary spelling variants, but not synonyms.
    """
    active, triples = [], set()
    morphs = list(tokenizer.tokenize(text, mode))
    for i, morph in enumerate(morphs):
        pos = morph.part_of_speech()[0]
        if pos == '補助記号':
            active = []
        if pos in ('名詞', '代名詞') and i + 1 < len(morphs):
            next_morph = morphs[i + 1]
            if next_morph.part_of_speech()[0] == '助詞':
                active.append((morph.normalized_form(), next_morph.surface()))
        if pos == '動詞':
            triples.update((noun, case, predicate_key(morphs, i)) for noun, case in active)
    return triples


def load_pairs(path, tokenizer, mode):
    with Path(path).open(encoding='utf-8', newline='') as file:
        rows = list(csv.DictReader(file, delimiter='\t'))
    cases, seen = [], set()
    for i, row in enumerate(rows):
        prefix = row['名詞'] + row['格']
        texts = [prefix + row[field] + '。' for field in ('自然な述語', '不自然な述語')]
        keys = []
        for text in texts:
            triples = lexical_triples(text, tokenizer, mode)
            # A suru noun contributes an additional adjacent noun; select the
            # argument at the start, not the verbal noun.
            noun = list(tokenizer.tokenize(row['名詞'], mode))[-1].normalized_form()
            selected = sorted(t for t in triples if t[:2] == (noun, row['格']))
            if len(selected) != 1:
                raise ValueError(f'ambiguous lexical key: {text}: {selected}')
            keys.append(selected[0])
        if keys[0] == keys[1] or keys[0] in seen:
            raise ValueError('duplicate or indistinguishable positive combination')
        seen.add(keys[0])
        cases.append(dict(id=f'comb-{i+1:03}', prefix=prefix, positive=texts[0], negative=texts[1],
                          positive_key=keys[0], negative_key=keys[1]))
    return cases


def overlap(cases, sources, tokenizer, mode):
    targets = {tuple(c[name]) for c in cases for name in ('positive_key', 'negative_key')}
    hits = {key: Counter() for key in targets}
    source_info = []
    for path in map(Path, sources):
        rows = 0
        for line in path.read_text(encoding='utf-8').splitlines():
            row = json.loads(line)
            # Prepared repair groups carry every completed positive plus the
            # original rollout. Count at most once per source record.
            texts = [row['text']] if 'text' in row else [row['original'], *[p['text'] for p in row['positives']]]
            found = set().union(*(lexical_triples(t, tokenizer, mode) for t in texts))
            for key in found & targets:
                hits[key][path.as_posix()] += 1
            rows += 1
        source_info.append(dict(path=path.as_posix(), sha256=sha(path), rows=rows))
    return hits, source_info


def encode_cases(cases, vocab, tokenizer, mode):
    from scripts.prepare_tatoeba import analyze
    from scripts.prepare_teacher_prose import orthographic_readings
    mapping = {(t['surface'], t['kana'], t['pos']): t['id'] for t in vocab['tokens'][7:]}
    encoded, exclusions = [], []
    for case in cases:
        parts = [analyze(case[name], tokenizer, mode) for name in ('positive', 'negative')]
        if any(p is None for p in parts):
            exclusions.append(dict(id=case['id'], reason='dictionary_or_grammar'))
            continue
        parts = [orthographic_readings(p) for p in parts]
        missing = sorted({s['surface'] for p in parts for s in p if (s['surface'], s['kana'], s['pos']) not in mapping})
        if missing:
            exclusions.append(dict(id=case['id'], reason='out_of_vocabulary', missing=missing))
            continue
        # Fix the same difficulty for both alternatives, using positive kana.
        # Negative length must not change the conditioning token.
        kana_len = sum(len(s['kana']) for s in parts[0])
        difficulty = 3 if kana_len <= 12 else 4 if kana_len <= 22 else 5
        ids = [[0, difficulty, *[mapping[(s['surface'], s['kana'], s['pos'])] for s in p], 1] for p in parts]
        prefix = 0
        while prefix < min(map(len, ids)) and ids[0][prefix] == ids[1][prefix]:
            prefix += 1
        if not 2 <= prefix < min(map(len, ids)) or max(map(len, ids)) > 64:
            raise ValueError('invalid contrast encoding')
        encoded.append(dict(id=case['id'], ids=ids, prefix_length=prefix, difficulty=difficulty))
    return encoded, exclusions


def score_checkpoint(checkpoint, encoded, vocab_hash):
    import torch
    from training.train import load_checkpoint
    model, ck = load_checkpoint(checkpoint)
    if ck['data_manifest']['vocab_sha256'] != vocab_hash:
        raise ValueError('checkpoint vocabulary differs')
    records = []
    with torch.inference_mode():
        for case in encoded:
            a, b = case['ids']
            n = case['prefix_length']
            logp = model(torch.tensor([a[:n]]))[0, -1].log_softmax(-1)
            margin = float(logp[a[n]] - logp[b[n]])
            prior = model(torch.tensor([a[:2]]))[0, -1].log_softmax(-1)
            prior_margin = float(prior[a[n]] - prior[b[n]])
            # First divergent token has exactly the same conditioning context.
            # It is not a complete-sentence score or a free-generation test.
            records.append(dict(id=case['id'], log_odds=margin, prior_log_odds=prior_margin,
                                context_gain=margin-prior_margin,
                                positive_probability=float(logp[a[n]].exp()),
                                negative_probability=float(logp[b[n]].exp())))
    return dict(path=str(checkpoint), sha256=sha(checkpoint), records=records,
                preferred=sum(r['log_odds'] > 0 for r in records), total=len(records),
                prior_preferred=sum(r['prior_log_odds'] > 0 for r in records),
                positive_context_gain=sum(r['context_gain'] > 0 for r in records),
                mean_log_odds=sum(r['log_odds'] for r in records)/len(records) if records else None)


def run(pairs, data, sources, checkpoints):
    from sudachipy import Dictionary, SplitMode
    tokenizer, mode = Dictionary().tokenizer(), SplitMode.C
    cases = load_pairs(pairs, tokenizer, mode)
    data = Path(data)
    manifest = json.loads((data/'manifest.json').read_text(encoding='utf-8'))
    if sha(data/'vocab.json') != manifest['vocab_sha256']:
        raise ValueError('vocabulary hash mismatch')
    parent_sources = []
    for name, expected in manifest['split_sha256'].items():
        if sha(data/name) != expected:
            raise ValueError('parent split hash mismatch')
        parent_sources.append(data/name)
    hits, provenance = overlap(cases, [*parent_sources, *sources], tokenizer, mode)
    for case in cases:
        case['overlap'] = {name: dict(hits[tuple(case[name+'_key'])]) for name in ('positive', 'negative')}
    vocab = json.loads((data/'vocab.json').read_text(encoding='utf-8'))
    encoded, exclusions = encode_cases(cases, vocab, tokenizer, mode)
    if checkpoints:
        import torch
        torch.set_num_threads(2)
    return dict(version=1, scope='Model-authored literal everyday contrasts; exploratory checkpoint diagnostic, not unseen-combination evaluation or naturalness accuracy',
                pairs_sha256=sha(pairs), script_sha256=sha(__file__), parent_manifest_sha256=sha(data/'manifest.json'),
                vocab_sha256=sha(data/'vocab.json'),
                analyzer={name: importlib.metadata.version(name) for name in ('SudachiPy', 'SudachiDict-core')},
                overlap_rule='normalized noun + adjacent particle + subsequent verb in punctuation span; inflections and dictionary variants, no synonym/dependency guarantee',
                sources=provenance, cases=cases, encoded=encoded, exclusions=exclusions,
                checkpoints=[score_checkpoint(c, encoded, sha(data/'vocab.json')) for c in checkpoints])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pairs', default='data/combinations-v13/pairs.tsv')
    parser.add_argument('--data', default='build/everyday-v11-data')
    parser.add_argument('--source', action='append', default=[])
    parser.add_argument('--checkpoint', action='append', default=[])
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise ValueError('output must be a new file')
    result = run(args.pairs, args.data, args.source, args.checkpoint)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
