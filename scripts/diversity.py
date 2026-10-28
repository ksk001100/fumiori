"""Observable lexical concentration, not a semantic or naturalness judgement."""
from collections import Counter, defaultdict
import math


def contains_excluded_tokens(row):
    return any(s['pos'] == '文字' or any(p in ('人名', '固有名詞') for p in s['pos'].split('-'))
               for s in row['segments'])


def require_generation_eligible(rows):
    if any(contains_excluded_tokens(row) for row in rows):
        raise ValueError('training data contains tokens excluded during generation; rebuild with scripts.prepare_stories')


def features(row):
    segments = row['segments']
    nouns = {s['surface'] for s in segments if s['pos'] == '名詞'}
    predicates = {s['surface'] for s in segments if s['pos'] in ('動詞', '形容詞')}
    words = [s for s in segments if s['pos'] != '記号']
    ending = ''.join(s['surface'] for s in words[-2:])
    return dict(nouns=nouns, predicates=predicates, endings={ending} if ending else set())


def difficulty(row):
    # Public outputs omit the final period; commas count towards kana length.
    n = len(row['kana'])
    return 'easy' if n <= 12 else 'normal' if n <= 22 else 'hard'


def concentration(rows):
    counts = {name: Counter() for name in ('nouns', 'predicates', 'endings')}
    for row in rows:
        for name, values in features(row).items():
            counts[name].update(values)
    result = {}
    for name, counter in counts.items():
        total = counter.total()
        result[name] = dict(
            distinct=len(counter), occurrences=total,
            effective_types=math.exp(-sum((n / total) * math.log(n / total) for n in counter.values())) if total else 0,
            top10_share=sum(n for _, n in counter.most_common(10)) / total if total else 0,
            top=[dict(surface=s, sentences=n, sentence_rate=n / len(rows)) for s, n in sorted(counter.items(), key=lambda x: (-x[1], x[0]))[:20]])
    return dict(sentences=len(rows), unique_texts=len({r['text'].rstrip('。') for r in rows}),
                generation_ineligible_sentences=sum(contains_excluded_tokens(r) for r in rows), **result)


def diversity_report(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[difficulty(row)].append(row)
    return dict(overall=concentration(rows),
        by_difficulty={d: concentration(groups[d]) for d in ('easy', 'normal', 'hard')},
        interpretation='Surface forms and existing POS only; no lemma or topic inference. '
        'Inflections count separately. Compare equal sample counts per difficulty. '
        'Lexical diversity does not establish naturalness, reading accuracy or topic fairness.')


def lexical_factors(rows, strength=.5, max_ratio=3.0):
    """Bounded row weights within difficulty; never edit or combine sentences.

    Average log sentence frequency over noun/predicate types gives common words
    less training mass. Geometric centering avoids dependence on corpus size.
    A final normalization is performed by the caller to preserve difficulty mass.
    """
    if not math.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError('lexical balance strength must be in [0, 1]')
    if not math.isfinite(max_ratio) or not 1 <= max_ratio <= 10:
        raise ValueError('lexical balance max_ratio must be in [1, 10]')
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[row['ids'][1]].append(i)
    factors = [1.0] * len(rows)
    for indices in groups.values():
        signatures = {}
        counts = Counter()
        for i in indices:
            f = features(rows[i])
            signatures[i] = {(name, word) for name in ('nouns', 'predicates') for word in f[name]}
            counts.update(signatures[i])
        scores = {i: -strength * sum(math.log(counts[t]) for t in ts) / len(ts)
                  for i, ts in signatures.items() if ts}
        if not scores:
            continue
        center = sum(scores.values()) / len(scores)
        # The largest/smallest factor ratio is at most max_ratio, even for outliers.
        limit = math.log(max_ratio) / 2
        for i, score in scores.items():
            factors[i] = math.exp(max(-limit, min(limit, score - center)))
    return factors


def topic_factors(rows, strength=.5, max_ratio=4.0):
    """Soften concentration in authored topic labels, using training rows only."""
    if not math.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError('topic balance strength must be in [0, 1]')
    if not math.isfinite(max_ratio) or not 1 <= max_ratio <= 10:
        raise ValueError('topic balance max_ratio must be in [1, 10]')
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        topic = (row.get('generation') or {}).get('topic')
        if not isinstance(topic, str) or not topic:
            raise ValueError('topic balance requires explicit authored topic labels')
        groups[row['ids'][1]].append((i, topic))
    factors = [1.0] * len(rows)
    for group in groups.values():
        counts = Counter(topic for _, topic in group)
        center = sum(math.log(counts[t]) for _, t in group) / len(group)
        limit = math.log(max_ratio) / 2
        for i, topic in group:
            score = strength * (center - math.log(counts[topic]))
            factors[i] = math.exp(max(-limit, min(limit, score)))
    return factors
