import math

import pytest

from scripts.content_audit import EVERYDAY_POLICY, everyday_gate, narrative_cues
from scripts.diversity import topic_factors
from training.train import sampling_weights


def rows():
    return [dict(ids=[0, d, 7, 1], generation=dict(topic=t))
            for d in (3, 4, 5) for t in ['nature'] * 90 + ['work'] * 10]


def test_topic_weights_reduce_concentration_and_preserve_difficulty_mass():
    data = rows()
    factors = topic_factors(data)
    weights = sampling_weights(data, topic_balance=dict(strength=.5, max_ratio=4))
    for offset in (0, 100, 200):
        assert factors[offset + 90] > factors[offset]
        assert max(factors[offset:offset+100]) / min(factors[offset:offset+100]) <= 4
        assert float(weights[offset:offset+100].sum()) == pytest.approx(1)
        assert float(weights[offset:offset+90].sum()) < .9
    assert sampling_weights(data, topic_balance=dict(strength=0)).tolist() == pytest.approx(sampling_weights(data).tolist())
    assert topic_factors([]) == []


@pytest.mark.parametrize('options', [dict(strength=math.nan), dict(strength=-1), dict(max_ratio=.5), dict(max_ratio=math.inf)])
def test_topic_configuration_is_bounded(options):
    with pytest.raises(ValueError):
        topic_factors(rows(), **options)


def test_missing_topic_is_rejected_instead_of_guessed():
    with pytest.raises(ValueError, match='explicit authored'):
        topic_factors([dict(ids=[0, 3, 1])])


def test_corpus_mixture_preserves_requested_mass_without_difficulty_drift():
    data = [dict(ids=[0, d, 7, 1], training_group=g) for d in (3, 4, 5)
            for g in ['paired'] * 20 + ['everyday'] * 5]
    weights = sampling_weights(data, corpus_mixture=dict(paired=.25, everyday=.75))
    for d in (3, 4, 5):
        for g, mass in [('paired', .25), ('everyday', .75)]:
            assert sum(float(w) for r, w in zip(data, weights)
                       if r['ids'][1] == d and r['training_group'] == g) == pytest.approx(mass)
    for mixture in [dict(paired=0, everyday=1), dict(paired=math.nan, everyday=1), dict(everyday=1)]:
        with pytest.raises(ValueError):
            sampling_weights(data, corpus_mixture=mixture)


def test_style_requires_review_and_multiple_topics_and_limits_concentration():
    ratings = [dict(fairy_tale=False, topic=EVERYDAY_POLICY['topics'][i % 10]) for i in range(100)]
    assert everyday_gate(ratings, EVERYDAY_POLICY)['passed']
    assert not everyday_gate([dict(r, topic='work') for r in ratings], EVERYDAY_POLICY)['passed']
    assert not everyday_gate([dict(r, fairy_tale=i < 6) for i, r in enumerate(ratings)], EVERYDAY_POLICY)['passed']
    assert not everyday_gate(ratings[:-1], EVERYDAY_POLICY)['passed']
    with pytest.raises(ValueError):
        everyday_gate([dict(topic='work')], EVERYDAY_POLICY)
    with pytest.raises(ValueError):
        everyday_gate([dict(fairy_tale=False, topic='invented')], EVERYDAY_POLICY)


def test_cues_do_not_claim_to_classify_ordinary_prose():
    result = narrative_cues([dict(text='村の図書館で働く'), dict(text='今日の会議を準備する')])
    assert result['matching_rate'] == .5
    assert result['counts'] == {'村': 1}
    assert narrative_cues([])['matching_rate'] == 0


def test_everyday_review_raises_quality_threshold_without_changing_legacy_gates():
    import hashlib
    import json
    from scripts.stories_gate import review_gate
    samples = [dict(text=f'例文{i}', kana='れいぶん') for i in range(600)]
    selected = [dict(sample_id=i, **samples[i]) for i in range(100)]
    payload = json.dumps(selected).encode()
    report = dict(weights_sha256='weights', decoder_sha256='decoder', samples=samples,
                  review_sample_ids=list(range(100)), structural_gate=dict(passed=True))
    ratings = [dict(sample_id=i, natural=i >= 9, reading_ok=True, fairy_tale=False,
                    topic=EVERYDAY_POLICY['topics'][i % 10]) for i in range(100)]
    review = dict(weights_sha256='weights', decoder_sha256='decoder',
                  sample_sha256=hashlib.sha256(payload).hexdigest(),
                  reviewer='fixture', reviewer_type='test fixture', ratings=ratings)
    assert review_gate(report, review, payload, [])['passed']
    report['content_review_policy'] = EVERYDAY_POLICY
    assert not review_gate(report, review, payload, [])['passed']
    review['ratings'] = [dict(r, natural=r['sample_id'] >= 4) for r in ratings]
    assert review_gate(report, review, payload, [])['passed']


def test_everyday_corpus_preserves_protected_splits_readings_and_unseen_combinations(tmp_path):
    import hashlib
    import json
    from pathlib import Path
    from scripts.prepare_everyday import prepare
    from scripts.semantic_keys import core_key
    from scripts.diversity import require_generation_eligible
    manifest = prepare(tmp_path)
    assert manifest['stats']['reserved_unseen_combination'] > 10000
    vocab = json.loads((tmp_path / 'vocab.json').read_text(encoding='utf-8'))['tokens']
    cores = {}
    for split in ('train', 'validation', 'test', 'family-test'):
        path = tmp_path / (split + '.jsonl')
        assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest['split_sha256'][path.name]
        data = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
        require_generation_eligible(data)
        for r in data:
            assert r['source'] == 'original-agent-authored'
            assert r['kana'] == ''.join(vocab[i]['kana'] for i in r['ids'][2:-1])
            assert r['text'] == ''.join(vocab[i]['surface'] for i in r['ids'][2:-1])
            assert 5 <= len(r['kana']) <= 40
        old_ids = {json.loads(line)['id'] for line in Path('data/paired-v1', split + '.jsonl').read_text(encoding='utf-8').splitlines()}
        assert old_ids <= {r['id'] for r in data}
        cores[split] = {core_key(r['text']) for r in data}
    assert not any(cores['train'] & cores[s] for s in ('validation', 'test', 'family-test'))
