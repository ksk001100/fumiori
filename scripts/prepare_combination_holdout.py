"""Prepare a conservative combination holdout for future FROM-SCRATCH controls."""
import argparse
from collections import Counter
import copy
import json
from pathlib import Path

from scripts.audit_combinations import lexical_triples, load_pairs, predicate_key, sha


def prepare(data, pairs, output):
    from sudachipy import Dictionary, SplitMode
    data, pairs, output = Path(data), Path(pairs), Path(output)
    if output.exists():
        raise ValueError('output must be a new directory')
    tokenizer, mode = Dictionary().tokenizer(), SplitMode.C
    cases = load_pairs(pairs, tokenizer, mode)
    targets = {tuple(c[name]) for c in cases for name in ('positive_key', 'negative_key')}
    parent = json.loads((data/'manifest.json').read_text(encoding='utf-8'))
    if sha(data/'vocab.json') != parent['vocab_sha256']:
        raise ValueError('vocabulary hash mismatch')
    # Validate every input before creating output. No partial snapshot on bad hash.
    for filename, expected in parent['split_sha256'].items():
        if sha(data/filename) != expected:
            raise ValueError('parent split hash mismatch')
    output.mkdir(parents=True)
    (output/'vocab.json').write_bytes((data/'vocab.json').read_bytes())
    manifest = copy.deepcopy(parent)
    manifest['parent_manifest_sha256'] = sha(data/'manifest.json')
    manifest['description'] = 'Conservative combination holdout; frozen inherited vocabulary; retrain from random initialization only'
    manifest['splits'], manifest['split_sha256'] = {}, {}
    removals, nouns, verbs = {}, Counter(), Counter()
    for filename in parent['split_sha256']:
        kept, removed = [], []
        for line_number, line in enumerate((data/filename).read_text(encoding='utf-8').splitlines(), 1):
            row = json.loads(line)
            found = lexical_triples(row['text'], tokenizer, mode) & targets
            if found:
                removed.append(dict(line=line_number, id=row.get('id'), matches=sorted(found)))
            else:
                kept.append(line)
                if filename == 'train.jsonl':
                    morphs = list(tokenizer.tokenize(row['text'], mode))
                    for i, morph in enumerate(morphs):
                        pos = morph.part_of_speech()[0]
                        if pos in ('名詞', '代名詞'):
                            nouns[morph.normalized_form()] += 1
                        elif pos == '動詞':
                            verbs[predicate_key(morphs, i)] += 1
        path = output/filename
        path.write_text(''.join(line+'\n' for line in kept), encoding='utf-8')
        manifest['splits'][filename.removesuffix('.jsonl')] = len(kept)
        manifest['split_sha256'][filename] = sha(path)
        removals[filename] = removed
    coverage = []
    for case in cases:
        noun, _, positive = case['positive_key']
        negative = case['negative_key'][2]
        coverage.append(dict(id=case['id'], noun_count=nouns[noun], positive_verb_count=verbs[positive],
                             negative_verb_count=verbs[negative], components_observed=all((nouns[noun], verbs[positive], verbs[negative]))))
    report = dict(pairs_sha256=sha(pairs), script_sha256=sha(__file__),
                  extractor_sha256=sha(Path(__file__).with_name('audit_combinations.py')),
                  removals=removals, coverage=coverage,
                  restrictions=['No prior task-trained checkpoints or v12 repair auxiliary data',
                                'All additional future training sources must pass the same exclusion',
                                'Dictionary normalization is not synonym or semantic leakage detection',
                                'Inherited vocabulary preserves prior corpus word identities and counts; only sequence-combination holdout',
                                'Existing corpus validation/test also filtered; not comparable to original validation loss'])
    report_path = output/'holdout.json'
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    manifest['combination_holdout'] = dict(pairs_sha256=sha(pairs), report_sha256=sha(report_path),
                                          components_observed=sum(c['components_observed'] for c in coverage),
                                          total=len(coverage), initialization='random only; no prior task-trained checkpoint')
    (output/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default='build/everyday-v11-data')
    parser.add_argument('--pairs', default='data/combinations-v13/pairs.tsv')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.data, args.pairs, args.output)['combination_holdout']))
