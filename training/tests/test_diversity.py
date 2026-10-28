import math
import json

import pytest

from scripts.diversity import diversity_report, lexical_factors
from training.train import sampling_weights


def row(noun='本', verb='読む', level=3):
    return dict(text=noun + 'を' + verb, kana='ほんをよむ' if level == 3 else 'あ' * (15 if level == 4 else 25),
                ids=[0, level, 7, 8, 9, 1], segments=[
                    dict(surface=noun, pos='名詞'), dict(surface='を', pos='助詞'), dict(surface=verb, pos='動詞')])


def test_balancing_reduces_concentration_without_changing_difficulty_mass():
    rows = [row()] * 90 + [row('空', '見る')] * 10 + [row(level=4)] * 2 + [row(level=5)]
    before = [dict(r) for r in rows]
    weights = sampling_weights(rows, dict(strength=.5, max_ratio=3))
    for d in (3, 4, 5):
        subset = [float(w) for r, w in zip(rows, weights) if r['ids'][1] == d]
        assert sum(subset) == pytest.approx(1)
        assert max(subset) / min(subset) <= 3 + 1e-10
    assert weights[90] > weights[0]
    assert float(weights[:90].sum()) < .9
    assert rows == before
    legacy = sampling_weights(rows)
    assert sampling_weights(rows, dict(strength=0, max_ratio=3)).tolist() == pytest.approx(legacy.tolist())


def test_report_shows_repeated_words_even_for_distinct_sentences_and_keeps_empty_levels():
    report = diversity_report([row('本', '読む'), row('本', '開く')])
    assert report['overall']['unique_texts'] == 2
    assert report['overall']['nouns']['distinct'] == 1
    assert report['overall']['nouns']['top'][0]['sentence_rate'] == 1
    assert report['overall']['predicates']['distinct'] == 2
    assert report['by_difficulty']['hard']['sentences'] == 0
    assert diversity_report([])['overall']['nouns']['effective_types'] == 0


def test_sentence_frequency_counts_a_repeated_noun_once():
    sample = row()
    sample['segments'].append(dict(surface='本', pos='名詞'))
    assert diversity_report([sample])['overall']['nouns']['occurrences'] == 1


@pytest.mark.parametrize('options', [dict(strength=-.1), dict(strength=math.nan), dict(max_ratio=.5), dict(max_ratio=math.inf)])
def test_unsafe_balancing_settings_rejected(options):
    with pytest.raises(ValueError):
        lexical_factors([row()], **options)


def test_empty_and_missing_lexical_features_are_neutral():
    assert lexical_factors([]) == []
    blank = dict(ids=[0, 3, 1], segments=[])
    assert lexical_factors([row(), blank]) == [1, 1]


def test_audit_accepts_corpus_and_report_and_records_exact_input_hash(tmp_path):
    import hashlib
    from scripts.audit_diversity import audit_file
    rows = [row(), row('空', '見る')]
    for name, payload in [('corpus.jsonl', '\n'.join(json.dumps(r) for r in rows)),
                          ('report.json', json.dumps(dict(samples=rows))),
                          ('samples.json', json.dumps(rows))]:
        path = tmp_path / name
        path.write_text(payload, encoding='utf-8')
        report = audit_file(path)
        assert report['sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
        assert report['overall']['sentences'] == 2
        assert report['overall']['nouns']['distinct'] == 2


@pytest.mark.parametrize('pos', ['文字', '人名', '固有名詞', '名詞-固有名詞-人名'])
def test_training_rejects_tokens_that_cannot_be_generated(pos):
    from scripts.diversity import require_generation_eligible
    bad = row()
    bad['segments'].append(dict(surface='ま', pos=pos))
    with pytest.raises(ValueError, match='excluded during generation'):
        require_generation_eligible([row(), bad])
    assert diversity_report([row(), bad])['overall']['generation_ineligible_sentences'] == 1
    require_generation_eligible([row()])


def test_evaluation_rejects_stale_binary_and_records_runtime_hashes(tmp_path):
    import os
    from scripts.open_generation import runtime_identity
    (tmp_path / 'Cargo.toml').write_text('manifest')
    (tmp_path / 'Cargo.lock').write_text('lock')
    binary = tmp_path / 'cli.exe'
    binary.write_bytes(b'binary')
    os.utime(binary, ns=(1, 1))
    with pytest.raises(ValueError, match='older than Rust sources'):
        runtime_identity(binary, tmp_path)
    fresh = max(p.stat().st_mtime_ns for p in (tmp_path / 'Cargo.toml', tmp_path / 'Cargo.lock')) + 1_000_000_000
    os.utime(binary, ns=(fresh, fresh))
    result = runtime_identity(binary, tmp_path)
    assert len(result['cli_sha256']) == 64
    assert set(result['rust_source_sha256']) == {'Cargo.toml', 'Cargo.lock'}
