"""Copy diagnostics must preserve sentence boundaries and evaluation attribution."""
from collections import Counter
import math
import random

import pytest

from scripts.novelty_evaluate import review_selection, summarize_review, validate_requests
from scripts.novelty_metrics import SubstringIndex, TrainingIndex, grams, normalized, self_bleu, split_audit, words


def row(*tokens):
    return dict(text=''.join(tokens), kana=''.join(tokens),
                segments=[dict(surface=t, kana=t, pos='名詞') for t in tokens])


def test_copy_and_known_word_combinations_are_separate():
    index = TrainingIndex([row('猫', '走る'), row('犬', '眠る')])
    result = index.evaluate([row('猫', '眠る'), row('猫', '走る'), row('鳥', '走る')])
    assert result['exact_match_rate'] == 1/3
    assert result['ngram_novelty']['word']['1']['unseen'] == 1
    assert result['known_word_novel_bigrams'] == dict(total=2, novel=1, rate=.5)
    assert result['details'][0]['longest_common_substring'] == '眠る'
    assert result['details'][1]['lcs_fraction'] == 1


def test_substring_index_matches_exhaustive_search_and_never_crosses_sentences():
    rng = random.Random(12)
    for _ in range(20):
        texts = [''.join(rng.choices('abc', k=6)) for _ in range(4)]
        index = SubstringIndex(texts)
        for _ in range(20):
            query = ''.join(rng.choices('abcd', k=8))
            expected = max((len(query[i:j]) for i in range(len(query)) for j in range(i+1, len(query)+1)
                            if any(query[i:j] in t for t in texts)), default=0)
            actual = index.longest(query)
            assert len(actual) == expected
            assert any(actual in t for t in texts)
    assert len(SubstringIndex(['abc', 'def']).longest('cde')) == 2


def test_indexed_dice_is_exhaustive_not_a_top_k_approximation():
    rows = [row('abcdef'), row('xyzabc'), row('日本語'), row('猫')]
    index = TrainingIndex(rows)
    for text in ['abcxyz', '本語', '猫', '犬', 'ababc']:
        query = set(grams(text, 2))
        scores = [2*len(query & ref)/(len(query)+len(ref)) if query or ref else 0
                  for ref in index.bigrams]
        expected = 1 if text in index.exact else max(scores)
        assert index.near(text)[0] == expected


def brute_bleu(sequences):
    scores = []
    for i, sequence in enumerate(sequences):
        refs = [s for j, s in enumerate(sequences) if i != j]
        ps = []
        for n in range(1, min(4, len(sequence))+1):
            counts = grams(sequence, n)
            max_ref = Counter()
            for ref in refs:
                max_ref |= grams(ref, n)
            matches = sum(min(count, max_ref[g]) for g, count in counts.items())
            ps.append(matches/counts.total() if n == 1 else (matches+1)/(counts.total()+1))
        length = min(map(len, refs), key=lambda n: (abs(n-len(sequence)), n))
        scores.append(math.exp(min(0, 1-length/len(sequence)) + sum(math.log(p) for p in ps)/len(ps))
                      if all(ps) else 0)
    return sum(scores)/len(scores)


@pytest.mark.parametrize('sequences', [['abc', 'abc'], ['abc', 'xyz'], ['ababa', 'ababa', 'abaca', 'ab'], ['猫', '猫', '犬']])
def test_self_bleu_excludes_one_observation_and_keeps_duplicates(sequences):
    assert self_bleu(sequences) == pytest.approx(brute_bleu(sequences))
    assert self_bleu(['abc']) is None


def test_normalization_and_split_leakage():
    assert normalized('Ａ猫。') == 'A猫'
    assert normalized('今日、猫') != normalized('猫')
    audit = split_audit(dict(train=[row('Ａ猫。'), row('A猫')], test=[row('A猫。')]))
    assert audit['within']['train']['duplicate_rows'] == 1
    assert audit['between'][0]['shared_texts'] == 1
    assert not audit['exact_disjoint']
    with pytest.raises(ValueError):
        normalized('a\0b')


def test_word_metrics_reject_invalid_segmentation_instead_of_inventing_words():
    bad = row('猫')
    bad['text'] = '犬'
    with pytest.raises(ValueError):
        words(bad)
    bad = row('ね', 'こ')
    bad['segments'][0]['pos'] = '文字'
    with pytest.raises(ValueError):
        words(bad)


def test_empty_ngrams_are_undefined_not_perfectly_novel():
    result = TrainingIndex([row('猫')]).evaluate([row('犬')])
    assert result['ngram_novelty']['word']['2']['occurrence_rate'] is None
    assert result['self_bleu4']['word'] is None
    assert TrainingIndex([row('猫')]).evaluate([])['exact_match_rate'] is None


def test_request_accounting_keeps_failures_and_rejects_duplicate_seed():
    raw = dict(requested=2, success=1, history_size=0, samples=[dict(seed=10, source='model')], failures=[dict(seed=11)])
    validate_requests(raw, 10, 2)
    raw['failures'][0]['seed'] = 10
    with pytest.raises(ValueError):
        validate_requests(raw, 10, 2)


def test_review_binds_samples_and_separates_human_from_agent():
    selected = review_selection([dict(status='success', text='猫', kana='ねこ'),
                                 dict(status='failure', text='', kana='')], 2, 9)
    ratings = [dict(r, sample_id=str(r['sample_id']), natural='', reading_ok='') for r in selected]
    assert summarize_review(selected, ratings)['unrated'] == 2
    ratings[0].update(natural='1', reading_ok='0', reviewer='Codex', reviewer_kind='agent', notes='読み違い')
    result = summarize_review(selected, ratings)
    assert result['by_reviewer_kind']['independent_human']['reading_accuracy'] is None
    assert result['by_reviewer_kind']['agent']['reading_accuracy'] == 0
    ratings[0]['text'] = '犬'
    with pytest.raises(ValueError):
        summarize_review(selected, ratings)


def test_failed_request_cannot_be_rated_or_omitted():
    selected = review_selection([dict(status='failure', text='', kana='')], 1, 9)
    with pytest.raises(ValueError):
        summarize_review(selected, [])
    rating = dict(selected[0], sample_id='0', natural='1', reading_ok='1', reviewer='A',
                  reviewer_kind='independent_human', notes='test')
    with pytest.raises(ValueError):
        summarize_review(selected, [rating])
