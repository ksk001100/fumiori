"""Offline corpus redundancy audit; never remove data or change inference."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random

from scripts.novelty_evaluate import sha, write_json
from scripts.novelty_metrics import normalized


def lexical_sequence(row, mask_nouns=False):
    return tuple('<名詞>' if mask_nouns and s['pos'] == '名詞' else s['surface']
                 for s in row['segments'] if s['pos'] != '記号')


def replacement_groups(rows):
    """At least four tokens, differing at one SAME position, no insertions."""
    signatures = defaultdict(list)
    for i, row in enumerate(rows):
        sequence = lexical_sequence(row)
        if len(sequence) < 4:
            continue
        for position in range(len(sequence)):
            signatures[position, sequence[:position]+sequence[position+1:]].append(i)
    # Different punctuation or duplicate rows alone do not prove substitution.
    return [indices for indices in signatures.values()
            if len({lexical_sequence(rows[i]) for i in indices}) > 1]


class NeighborIndex:
    def __init__(self, rows):
        self.texts = [normalized(r['text']) for r in rows]
        self.grams = [set(zip(t, t[1:])) for t in self.texts]
        self.postings = defaultdict(list)
        self.exact = defaultdict(list)
        for i, (text, grams) in enumerate(zip(self.texts, self.grams)):
            self.exact[text].append(i)
            for gram in grams:
                self.postings[gram].append(i)

    def nearest(self, text, exclude=None):
        text = normalized(text)
        exact = [i for i in self.exact.get(text, []) if i != exclude]
        if exact:
            return 1.0, exact[0]
        query = set(zip(text, text[1:]))
        intersections = Counter()
        for gram in query:
            intersections.update(self.postings.get(gram, []))
        best, best_i = 0.0, None
        for i, overlap in intersections.items():
            if i == exclude:
                continue
            score = 2*overlap/(len(query)+len(self.grams[i]))
            if score > best or (score == best and (best_i is None or i < best_i)):
                best, best_i = score, i
        return best, best_i


def audit(data, configs, samples, seed):
    data = Path(data)
    manifest = json.loads((data/'manifest.json').read_text(encoding='utf-8'))
    parts = {}
    for filename, digest in manifest['split_sha256'].items():
        if sha(data/filename) != digest:
            raise ValueError('split hash mismatch')
        parts[filename.removesuffix('.jsonl')] = [json.loads(s) for s in (data/filename).read_text(encoding='utf-8').splitlines()]
    rows = parts['train']
    by_group = defaultdict(list)
    texts = defaultdict(list)
    skeletons = defaultdict(list)
    for i, row in enumerate(rows):
        by_group[row['training_group']].append(i)
        texts[normalized(row['text'])].append(i)
        skeletons[lexical_sequence(row, True)].append(i)
    replacements = replacement_groups(rows)
    changed = {i for group in replacements for i in group}
    masked = {i for indices in skeletons.values() if len(indices) > 1 for i in indices}
    index = NeighborIndex(rows)
    rng = random.Random(seed)
    diagnostics = []
    for group, indices in sorted(by_group.items()):
        for i in sorted(rng.sample(indices, min(samples, len(indices)))):
            score, other = index.nearest(rows[i]['text'], exclude=i)
            diagnostics.append(dict(group=group, row_index=i, text=rows[i]['text'], nearest_dice=score,
                neighbor_index=other, neighbor_text=rows[other]['text'] if other is not None else None,
                neighbor_group=rows[other]['training_group'] if other is not None else None))
    def near_summary(items):
        return dict(samples=len(items), counts={str(t):sum(r['nearest_dice'] >= t for r in items) for t in (.8,.9,.95)})
    groups = {}
    for name, indices in sorted(by_group.items()):
        groups[name] = dict(rows=len(indices), one_token_variant_rows=sum(i in changed for i in indices),
            noun_mask_shared_rows=sum(i in masked for i in indices),
            near_sample=near_summary([r for r in diagnostics if r['group'] == name]))
    heldout = {}
    for split, split_rows in sorted(parts.items()):
        if split == 'train':
            continue
        values = []
        for i in sorted(rng.sample(range(len(split_rows)), min(samples, len(split_rows)))):
            score, other = index.nearest(split_rows[i]['text'])
            values.append(dict(row_index=i, text=split_rows[i]['text'], nearest_dice=score,
                neighbor_index=other, neighbor_text=rows[other]['text'] if other is not None else None))
        heldout[split] = dict(rows=len(split_rows), train_exact_rows=sum(normalized(r['text']) in texts for r in split_rows),
            summary=near_summary(values), details=values)
    import torch
    from training.train import sampling_weights
    torch.set_num_threads(2)
    sampling = {}
    for config_path in configs:
        config = json.loads(Path(config_path).read_text(encoding='utf-8'))
        if Path(config['data_dir']).resolve() != data.resolve():
            raise ValueError('config must use the audited data')
        weights = sampling_weights(rows, config.get('lexical_balance'), config.get('topic_balance'), config.get('corpus_mixture'), config.get('corpus_group_map'))
        weights = (weights/weights.sum()).tolist()
        largest = sorted(skeletons.items(), key=lambda kv:-sum(weights[i] for i in kv[1]))[:10]
        sampling[config_path] = dict(config_sha256=sha(config_path),
            group_mass={name:sum(weights[i] for i in indices) for name,indices in by_group.items()},
            one_token_variant_mass=sum(weights[i] for i in changed),
            top_noun_mask_groups=[dict(pattern=' '.join(pattern),rows=len(indices),mass=sum(weights[i] for i in indices),
                examples=[rows[i]['text'] for i in indices[:4]]) for pattern,indices in largest])
    return dict(data=str(data),manifest_sha256=sha(data/'manifest.json'),split_sha256=manifest['split_sha256'],
        script_sha256=sha(__file__),seed=seed,samples_per_group_or_split=samples,
        limitations='Character bigram Dice and exact token skeletons are surface diagnostics, not semantic duplicate/quality labels. '
            'Near rates are stratified samples; do not pool as a uniform corpus sample. One-token groups overlap; no transitive semantic clustering.',
        rows=len(rows),exact_extra_rows=len(rows)-len(texts),exact_duplicate_texts=sum(len(v)>1 for v in texts.values()),
        one_token_variant_rows=len(changed),one_token_groups=len(replacements),noun_mask_shared_rows=len(masked),
        largest_one_token_groups=[dict(rows=len(v),examples=[rows[i]['text'] for i in v[:6]]) for v in sorted(replacements,key=len,reverse=True)[:10]],
        groups=groups,train_neighbor_details=diagnostics,heldout=heldout,sampling=sampling)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',required=True)
    parser.add_argument('--config',action='append',default=[])
    parser.add_argument('--samples',type=int,default=250)
    parser.add_argument('--seed',type=int,default=20261030)
    parser.add_argument('--output',required=True)
    args = parser.parse_args()
    if args.samples < 1 or Path(args.output).exists():
        raise ValueError('positive sample count and a new output path required')
    write_json(args.output,audit(args.data,args.config,args.samples,args.seed))
