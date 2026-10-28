"""V1 evaluation denominators and paired-input integrity."""
import hashlib
import json
from pathlib import Path
from scripts.open_generation import audit


def test_paired_v1_source_files_match_manifest():
    root = Path('data/paired-v1')
    manifest = json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    assert manifest['splits'] == {'train': 2500, 'validation': 300, 'test': 300, 'family-test': 300}
    for name, expected in {**manifest['split_sha256'], 'vocab.json': manifest['vocab_sha256']}.items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest() == expected


def test_novelty_audit_keeps_memorized_fallback_and_unsupported_outputs_visible():
    train = [dict(text='本を読む。', kana='ほんをよむ')]
    domain = train + [dict(text='詩集を読む。', kana='ししゅうをよむ')]
    samples = [dict(train[0], source='model'), dict(domain[1], source='model'),
        dict(text='空を読む', kana='そらをよむ', source='model'),
        dict(domain[1], source='fallback')]
    result = audit(samples, train, domain, 5)  # One failed request also stays in the denominator.
    assert result['training_exact_rate'] == 1 / 5
    assert result['novel_core_rate'] == 2 / 5
    assert result['novel_author_supported_rate'] == 1 / 5
    assert result['outside_reference_count'] == 1
    assert result['model_count'] == 3
    assert result['returned'] == 4
    assert result['unique_texts'] == 3
