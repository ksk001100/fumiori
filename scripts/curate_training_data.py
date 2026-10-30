"""Create a new, traceable TRAIN-only curated corpus; never overwrite sources."""
import argparse
from collections import Counter, defaultdict
import copy
import importlib.metadata
import json
from pathlib import Path
import shutil
import unicodedata

from scripts.audit_data_diversity import lexical_sequence
from scripts.novelty_evaluate import sha, write_json
from scripts.prepare_combination_holdout import prepare as prepare_holdout


def style_key(text, tokenizer, mode):
    """Remove punctuation and polite masu; retain tense, negation and voice."""
    key = []
    for morph in tokenizer.tokenize(unicodedata.normalize('NFKC', text), mode):
        grammar = morph.part_of_speech()
        pos = grammar[0]
        lemma = morph.dictionary_form()
        if morph.surface() in ('?', '!'):
            key.append(('発話',morph.surface()))
            continue
        if pos in ('補助記号', '記号', '空白'):
            continue
        mood = grammar[5] if grammar[5] in ('命令形','意志推量形') else None
        if pos == '助動詞' and lemma == 'ます':
            if mood:
                key.append(('活用',mood))
            continue
        if grammar[4] == '助動詞-タ':
            lemma = 'た'
        if pos == '助動詞' and lemma == 'です':
            if key and key[-1][0] == '形容詞':
                continue
            lemma = 'だ'
        key.append((pos, lemma))
        if grammar[4] == '助動詞-ダ' and grammar[5] in ('連用形-ニ','連体形-一般'):
            key.append(('修飾',grammar[5]))
        if mood:
            key.append(('活用',mood))
    return tuple(key)


def coverage(row):
    segments = row['segments']
    content = {'名詞', '動詞', '形容詞', '副詞', '形状詞'}
    features = {('word', s['surface'], s['pos']) for s in segments if s['pos'].split('-')[0] in content}
    predicates = [s['surface'] for s in segments if s['pos'].startswith('動詞')]
    if predicates:
        for a, b in zip(segments, segments[1:]):
            if a['pos'] == '名詞' and b['pos'] == '助詞':
                features.add(('role', a['surface'], b['surface'], predicates[-1]))
    return features


def cap_group(indices, features, ranks, limit):
    """Greedy lexical/role coverage, then diversity cap; explicit exceptions."""
    if len(indices) <= limit:
        return list(indices), 0
    pool = set(indices)
    missing = set().union(*(features[i] for i in indices))
    selected = []
    while pool and missing:
        best = min(pool, key=lambda i: (-len(features[i] & missing), ranks[i]))
        if not features[best] & missing:
            break
        selected.append(best)
        pool.remove(best)
        missing.difference_update(features[best])
    needed = max(0, limit-len(selected))
    selected.extend(sorted(pool, key=lambda i:ranks[i])[:needed])
    return selected, max(0, len(selected)-limit)


def curate(config_path):
    from sudachipy import Dictionary, SplitMode
    config = json.loads(Path(config_path).read_text(encoding='utf-8'))
    base, merged, output = (Path(config[k]) for k in ('base','merged','output'))
    if merged.exists() or output.exists():
        raise ValueError('curation requires new output directories')
    parent = json.loads((base/'manifest.json').read_text(encoding='utf-8'))
    if sha(base/'vocab.json') != parent['vocab_sha256']:
        raise ValueError('vocabulary changed')
    for name, digest in parent['split_sha256'].items():
        if sha(base/name) != digest:
            raise ValueError('source split changed')
    if sha(base/'holdout.json') != parent['combination_holdout']['report_sha256']:
        raise ValueError('source holdout changed')
    if importlib.metadata.version('SudachiDict-core') != parent['dictionary']['version']:
        raise ValueError('frozen dictionary version required')
    lines = (base/'train.jsonl').read_text(encoding='utf-8').splitlines()
    rows = [json.loads(s) for s in lines]
    exclusions = json.loads(Path(config['exclusions']).read_text(encoding='utf-8'))
    by_text = {r['text']:i for i,r in enumerate(rows)}
    manual = {}
    for item in exclusions:
        i = by_text[item['text']]
        if sha_bytes(lines[i]) != item['row_sha256']:
            raise ValueError('manually reviewed TRAIN row changed')
        manual[i] = item['reason']
    tokenizer, mode = Dictionary().tokenizer(), SplitMode.C
    ranks = [sha_bytes(line) for line in lines]
    removed, styles = {}, defaultdict(list)
    for i, row in enumerate(rows):
        if i in manual:
            removed[i] = dict(reason='reviewed_train_issue', detail=manual[i])
        elif any(term in row['text'] for term in config['quarantine_terms']):
            removed[i] = dict(reason='topic_quarantine', detail='Explicit self-harm topic excluded from this typing-training subset; not a grammar label')
        else:
            styles[style_key(row['text'],tokenizer,mode)].append(i)
    candidates = []
    for indices in styles.values():
        # Prefer an existing short representative, then immutable source hash.
        kept = min(indices, key=lambda i:(len(rows[i]['text']),ranks[i]))
        candidates.append(kept)
        for i in indices:
            if i != kept:
                removed[i] = dict(reason='punctuation_or_politeness', representative_line=kept+1)
    features = {i:coverage(rows[i]) for i in candidates}
    groups = defaultdict(list)
    for i in candidates:
        row = rows[i]
        eligible = sum(s['pos']=='名詞' for s in row['segments']) >= 2 and len(lexical_sequence(row)) >= 6
        key = (row['training_group'],row['ids'][1],lexical_sequence(row,True)) if eligible else ('unique',i)
        groups[key].append(i)
    kept, exceptions = [], 0
    for indices in groups.values():
        chosen, extra = cap_group(indices,features,ranks,config['pattern_limit'])
        kept.extend(chosen)
        exceptions += extra
        for i in set(indices)-set(chosen):
            removed[i] = dict(reason='pattern_concentration', pattern_limit=config['pattern_limit'])
    kept = sorted(kept)
    before_features = set().union(*(features[i] for i in candidates))
    after_features = set().union(*(features[i] for i in kept))
    if before_features != after_features or set(kept) & removed.keys() or len(kept)+len(removed) != len(rows):
        raise ValueError('coverage or row-accounting invariant failed')
    report = dict(config_sha256=sha(config_path),script_sha256=sha(__file__),exclusions_sha256=sha(config['exclusions']),
        source_train_sha256=sha(base/'train.jsonl'),parent_manifest_sha256=sha(base/'manifest.json'),
        before=len(rows),after=len(kept),removed_reasons=dict(Counter(r['reason'] for r in removed.values())),
        before_groups=dict(Counter(r['training_group'] for r in rows)),after_groups=dict(Counter(rows[i]['training_group'] for i in kept)),
        pattern_limit=config['pattern_limit'],coverage_exceptions=exceptions,
        candidate_features=len(before_features),retained_features=len(after_features),
        coverage_scope='Surface content words and adjacent noun/particle/final-verb features AFTER style deduplication, not dependency or semantic coverage; frozen vocabulary can contain inactive IDs',
        kept_source_lines=[i+1 for i in kept],
        removed=[dict(source_line=i+1,text=rows[i]['text'],row_sha256=ranks[i],**r) for i,r in sorted(removed.items())])
    merged.mkdir(parents=True)
    for name in parent['split_sha256']:
        if name != 'train.jsonl':
            shutil.copyfile(base/name,merged/name)
    shutil.copyfile(base/'vocab.json',merged/'vocab.json')
    (merged/'train.jsonl').write_text(''.join(lines[i]+'\n' for i in kept),encoding='utf-8',newline='\n')
    write_json(merged/'curation.json',report)
    manifest = copy.deepcopy(parent)
    manifest.pop('combination_holdout')
    manifest['splits']['train'] = len(kept)
    manifest['split_sha256']['train.jsonl'] = sha(merged/'train.jsonl')
    manifest['data_curation'] = dict(parent_manifest_sha256=sha(base/'manifest.json'),report_sha256=sha(merged/'curation.json'),
        script_sha256=sha(__file__),policy='TRAIN-only immutable-row filtering; punctuation/polite-style representatives, lexical/role-preserving pattern cap, explicit reviewed/topic quarantines; no quality certification')
    write_json(merged/'manifest.json',manifest)
    final = prepare_holdout(merged,config['pairs'],output)
    shutil.copyfile(merged/'curation.json',output/'curation.json')
    if final['splits']['train'] != len(kept):
        raise ValueError('unexpected holdout removals')
    return report


def sha_bytes(value):
    import hashlib
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    args = parser.parse_args()
    result = curate(args.config)
    print(json.dumps({k:v for k,v in result.items() if k not in ('removed','kept_source_lines')},ensure_ascii=False))
