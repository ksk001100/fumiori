import pytest

from scripts.independent_seeds import summarize


def test_independent_collisions_and_failures_are_not_hidden():
    report = dict(requested=6, success=4, history_size=0, seed_start=10,
                  failures=[dict(seed=13),dict(seed=15)],
                  samples=[dict(seed=10, text='猫が眠る'), dict(seed=11, text='猫が眠る'),
                           dict(seed=12, text='本を読む'), dict(seed=14, text='本を読んだ')])
    result = summarize(report, ['猫が眠る。'])
    assert result['unique_ratio'] == .5  # Failures remain in the denominator.
    assert result['duplicate_count'] == 1
    assert result['core_duplicate_count'] == 2  # Tense changes do not fake novelty.
    assert result['novel_core_rate'] == pytest.approx(2 / 6)
    assert result['by_difficulty']['easy']['success'] == 1
    assert result['by_difficulty']['normal']['duplicate_count'] == 0
    report['history_size'] = 16
    with pytest.raises(ValueError, match='history_size=0'):
        summarize(report, [])


def test_promotion_requires_independent_collisions_even_if_history_hides_them():
    from scripts.stories_gate import independent_gate
    report = dict(weights_sha256='weights', vocab_sha256='vocab', decoder_sha256='decoder',
                  runtime_identity={'cli': 'hash'}, sampling=dict(temperature=.5, top_p=.9),
                  content_review_policy=dict(independent_seed_evaluation=dict(samples=3,seed=10,min_unique_ratio=.99)))
    independent = dict(requested=3,success=3,seed_start=10,history_size=0,failures=[],
                       difficulty_cycle=['easy','normal','hard'],
                       model_sha256={'model.safetensors':'weights','vocab.json':'vocab'},
                       decoder_sha256='decoder',runtime_identity=report['runtime_identity'],sampling=report['sampling'],
                       samples=[dict(seed=i,text='猫が眠る。') for i in range(10,13)])
    assert not independent_gate(report,independent,[])['passed']
    independent['samples'][1]['text'] = '本を読む。'
    independent['samples'][2]['text'] = '犬が走る。'
    assert independent_gate(report,independent,[])['passed']
    report['content_review_policy']['min_novel_rate'] = .95
    assert not independent_gate(report,independent,[dict(text='猫が眠る。')])['passed']
    independent['decoder_sha256'] = 'changed'
    with pytest.raises(ValueError, match='decoder/runtime'):
        independent_gate(report,independent,[])
