"""Visible narrative cues are diagnostics; genre labels require explicit review."""
from collections import Counter


TOPICS = ('animals', 'art', 'books', 'care', 'cooking', 'craft', 'everyday', 'exercise',
          'food', 'game', 'garden', 'health', 'home', 'learning', 'making', 'music',
          'nature', 'science', 'season', 'senses', 'shopping', 'social', 'sport',
          'thinking', 'town', 'travel', 'weather', 'work')
NARRATIVE_CUES = ('ある日', '少年', '少女', '彼女', '村', '夢', '魔法', '勇気', '幸せ')
EVERYDAY_POLICY = dict(max_fairy_tale=5, min_topics=8, max_topic_count=30,
                       topics=list(TOPICS), sample_count=100, min_acceptable=95,
                       min_novel_acceptable_rate=.9, min_novel_rate=.3)


def narrative_cues(rows):
    counts = Counter(cue for r in rows for cue in NARRATIVE_CUES if cue in r['text'])
    matching = sum(any(cue in r['text'] for cue in NARRATIVE_CUES) for r in rows)
    return dict(sentences=len(rows), matching_sentences=matching,
                matching_rate=matching / len(rows) if rows else 0,
                counts=dict(counts),
                interpretation='Literal substring diagnostics only. These cues also occur in ordinary prose; '
                'their presence or absence does not establish genre or quality.')


def everyday_gate(ratings, policy):
    allowed = set(policy['topics']) | {'other'}
    if any(type(r.get('fairy_tale')) is not bool or r.get('topic') not in allowed for r in ratings):
        raise ValueError('everyday review requires explicit fairy_tale booleans and valid topic labels')
    counts = Counter(r['topic'] for r in ratings)
    known = len(set(counts) - {'other'})
    fairy = sum(r['fairy_tale'] for r in ratings)
    largest = max(counts.values(), default=0)
    return dict(passed=len(ratings) == policy['sample_count'] and fairy <= policy['max_fairy_tale']
                and known >= policy['min_topics'] and largest <= policy['max_topic_count'],
                fairy_tale_count=fairy, distinct_topics=known, topic_counts=dict(sorted(counts.items())),
                largest_topic_count=largest, policy=policy)
