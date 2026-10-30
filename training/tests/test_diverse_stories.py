import random

import pytest

from scripts.diverse_stories import NearFilter, standalone, reading_issue


@pytest.mark.parametrize('threshold',[.8,.85,.95,1.])
def test_prefix_filter_agrees_with_exhaustive_dice_including_incremental_inserts(threshold):
    rng=random.Random(341)
    references=[''.join(rng.choices('あいうえおかきくけこ',k=rng.randrange(8,28))) for _ in range(80)]
    index=NearFilter(references[:30],threshold)
    inserted=references[:30]
    for original in references[30:]:
        for text in (original,original[:-1]+'あ',inserted[-1],inserted[-1][:-2]+'いう'):
            query=index.grams(text)
            expected=any(2*len(query&index.grams(old))>=threshold*(len(query)+len(index.grams(old)))-1e-12 for old in inserted)
            assert index.contains_near(text)==expected
        index.add(original);inserted.append(original)


def test_new_rare_grams_do_not_change_global_order_and_single_character_exact():
    index=NearFilter(['あいうえお'])
    prefix=index.prefix(index.grams('かきくけこ'))
    index.add('かきくけこ')
    assert index.prefix(index.grams('かきくけこ'))==prefix
    assert index.contains_near('かきくけこ')
    index.add('あ')
    assert index.contains_near('あ')
    assert not index.contains_near('ん')


def test_context_and_quote_fragments_are_not_standalone_candidates():
    assert not standalone('すると彼は出かけた。')
    assert not standalone('彼は「こんにちは」と言った。')
    assert not standalone('しっかりと、山の上に小さな男の子がいました。')
    assert not standalone('それ以来、犬と鳥は毎日遊びました。')
    assert not standalone('ぎりぎりの場所で迷った男がいました。')
    assert not standalone('ほぼ毎日、海の近くに住む小さな魚がいました。')
    assert not standalone('静かな森の中で、一人の女の子がいました。')
    assert not standalone('やがて、森の中に大きな生物が住んでいました。')
    assert standalone('森の中で、少女が楽しく遊んでいました。')
    assert standalone('窓から山が見える。')


def test_reading_guard_handles_split_compounds_and_does_not_ban_all_jitsu():
    assert reading_issue(dict(text='宇宙人に会う',kana='うちゅうにんにあう',segments=[]))
    assert reading_issue(dict(text='何より大切',kana='なんよりたいせつ',segments=[]))
    assert reading_issue(dict(text='甘い実',kana='あまいじつ',segments=[dict(surface='甘い',kana='あまい'),dict(surface='実',kana='じつ')]))
    assert not reading_issue(dict(text='実は',kana='じつは',segments=[dict(surface='実',kana='じつ'),dict(surface='は',kana='は')]))
    assert not reading_issue(dict(text='実験',kana='じっけん',segments=[dict(surface='実験',kana='じっけん')]))


def test_additional_source_notice_is_verified_and_not_discarded():
    import hashlib
    from training.export import additional_data_attribution
    text='MIT License\nCopyright The SimpleStories Project\n'
    source=dict(dataset='SimpleStories/SimpleStories-JA',license='MIT',authors='The SimpleStories Project',dataset_url='https://example.invalid',license_text=text,license_text_sha256=hashlib.sha256(text.encode()).hexdigest())
    sources,notice=additional_data_attribution(dict(additional_sources=[source]))
    assert sources[0]['dataset']==source['dataset'] and text in notice
    assert 'license_text' not in sources[0]
    with pytest.raises(ValueError,match='verified'):
        additional_data_attribution(dict(additional_sources=[dict(source,license_text=text+'changed')]))
