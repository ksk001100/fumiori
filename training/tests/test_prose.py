import json
import pytest

from scripts.prepare_tatoeba import hiragana, normalize_sentence, transcription


def test_furigana_preserves_orthographic_particles_and_word_readings():
    assert transcription('[私|わたし]は[図書館|と|しょ|かん]へ[行|い]く。') == ('私は図書館へ行く。', 'わたしはとしょかんへいく。')
    assert transcription('コーヒーを[飲|の]む。') == ('コーヒーを飲む。', 'こーひーをのむ。')
    assert transcription('[壊れた|こわれた') is None
    assert hiragana('カタカナとひらがな') == 'かたかなとひらがな'


def test_plain_prose_normalization_rejects_unsupported_characters():
    assert normalize_sentence(' 本を読みました！ ') == '本を読みました。'
    assert normalize_sentence('本を読みました。次も読む。') is None
    assert normalize_sentence('数字123を書く。') is None
    assert normalize_sentence('「引用」する。') is None


def test_real_dictionary_rejects_names_and_keeps_nonterminal_verb_forms():
    pytest.importorskip('sudachipy')
    pytest.importorskip('sudachidict_core')
    from sudachipy import Dictionary, SplitMode
    from scripts.prepare_tatoeba import analyze
    tokenizer = Dictionary().tokenizer()
    assert analyze('トムは東京へ行った。', tokenizer, SplitMode.C) is None
    segments = analyze('本を読んでいます。', tokenizer, SplitMode.C)
    assert ''.join(s['kana'] for s in segments) == 'ほんをよんでいます'
    assert segments[-2] == dict(surface='読んでいます', kana='よんでいます', pos='動詞')
    segments = analyze('本を取りに行く。', tokenizer, SplitMode.C)
    assert any(s['surface'] == '取り' and s['pos'] == '動詞-非終止' for s in segments)
