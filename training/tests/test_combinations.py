import hashlib
import json

import pytest

from scripts.audit_combinations import lexical_triples, load_pairs, overlap, encode_cases, score_checkpoint
from scripts.prepare_combination_holdout import prepare


@pytest.fixture
def analyzer():
    sudachi = pytest.importorskip('sudachipy')
    pytest.importorskip('sudachidict_core')
    return sudachi.Dictionary().tokenizer(), sudachi.SplitMode.C


def test_inflections_variants_and_punctuation(analyzer):
    tokenizer, mode = analyzer
    base = lexical_triples('水を飲む。', tokenizer, mode)
    assert base <= lexical_triples('水をゆっくり飲みました。', tokenizer, mode)
    assert base == lexical_triples('水を飲んだ。', tokenizer, mode)
    assert not (base & lexical_triples('水を置く。お茶を飲む。', tokenizer, mode))
    # Intentionally conservative: do not claim syntactic dependency analysis.
    assert base <= lexical_triples('水を置いてお茶を飲む。', tokenizer, mode)
    suru = lexical_triples('部屋を掃除する。', tokenizer, mode)
    assert suru == lexical_triples('部屋を掃除しました。', tokenizer, mode)
    assert not (suru & lexical_triples('部屋を整理する。', tokenizer, mode))


def write_pairs(path):
    path.write_text('名詞\t格\t自然な述語\t不自然な述語\n水\tを\t飲む\t読む\n', encoding='utf-8')


def test_overlap_includes_every_repair_and_original(tmp_path, analyzer):
    tokenizer, mode = analyzer
    pairs = tmp_path/'pairs.tsv'
    write_pairs(pairs)
    cases = load_pairs(pairs, tokenizer, mode)
    source = tmp_path/'repairs.jsonl'
    source.write_text(json.dumps(dict(original='水を読んだ。', positives=[dict(text='水を飲みます。'), dict(text='水を飲んだ。')]), ensure_ascii=False)+'\n', encoding='utf-8')
    hits, provenance = overlap(cases, [source], tokenizer, mode)
    assert sum(hits[cases[0]['positive_key']].values()) == 1
    assert sum(hits[cases[0]['negative_key']].values()) == 1
    assert provenance[0]['rows'] == 1


def test_holdout_excludes_variants_checks_hash_and_counts_components(tmp_path, analyzer):
    data = tmp_path/'data'
    data.mkdir()
    pairs = tmp_path/'pairs.tsv'
    write_pairs(pairs)
    (data/'vocab.json').write_text('{}', encoding='utf-8')
    hashes = {}
    for split, texts in [('train', ['水を飲みます。', '水を読んだ。', '水が冷たい。', 'お茶を飲む。', '本を読む。']),
                         ('validation', ['水を飲んだ。', '本を読む。'])]:
        path = data/(split+'.jsonl')
        path.write_text(''.join(json.dumps(dict(text=t), ensure_ascii=False)+'\n' for t in texts), encoding='utf-8')
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = dict(split_sha256=hashes, vocab_sha256=hashlib.sha256((data/'vocab.json').read_bytes()).hexdigest())
    (data/'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    output = tmp_path/'heldout'
    result = prepare(data, pairs, output)
    assert result['splits'] == dict(train=3, validation=1)
    assert result['combination_holdout']['components_observed'] == 1
    for filename in hashes:
        assert hashlib.sha256((output/filename).read_bytes()).hexdigest() == result['split_sha256'][filename]
        for line in (output/filename).read_text(encoding='utf-8').splitlines():
            assert ('水', 'を', '飲む') not in lexical_triples(json.loads(line)['text'], *analyzer)
    with pytest.raises(ValueError, match='new directory'):
        prepare(data, pairs, output)
    (data/'train.jsonl').write_text('changed', encoding='utf-8')
    with pytest.raises(ValueError, match='hash'):
        prepare(data, pairs, tmp_path/'bad')
    assert not (tmp_path/'bad').exists()


def test_encoding_same_difficulty_and_explicit_oov(tmp_path, analyzer):
    from scripts.prepare_tatoeba import analyze
    tokenizer, mode = analyzer
    pairs = tmp_path/'pairs.tsv'
    write_pairs(pairs)
    cases = load_pairs(pairs, tokenizer, mode)
    words = {tuple(s[k] for k in ('surface', 'kana', 'pos')) for c in cases
             for field in ('positive', 'negative') for s in analyze(c[field], tokenizer, mode)}
    vocab = dict(tokens=[{}]*7+[dict(id=i, surface=s, kana=k, pos=p) for i, (s,k,p) in enumerate(sorted(words),7)])
    encoded, excluded = encode_cases(cases, vocab, tokenizer, mode)
    assert not excluded
    a,b = encoded[0]['ids']
    n = encoded[0]['prefix_length']
    assert a[:n] == b[:n] and a[n] != b[n]
    assert a[1] == b[1] == 3
    vocab['tokens'] = vocab['tokens'][:7]
    encoded, excluded = encode_cases(cases, vocab, tokenizer, mode)
    assert not encoded and excluded[0]['reason'] == 'out_of_vocabulary'


def test_checkpoint_score_compares_identical_context_and_removes_prior(tmp_path, monkeypatch):
    import torch
    import training.train
    calls = []
    class Model:
        def __call__(self, ids):
            calls.append(ids.tolist())
            logits = torch.zeros(1, ids.shape[1], 12)
            logits[0,-1,8] = 2 if ids.shape[1]>2 else 3
            logits[0,-1,9] = 1
            return logits
    monkeypatch.setattr(training.train, 'load_checkpoint', lambda _: (Model(), dict(data_manifest=dict(vocab_sha256='v'))))
    checkpoint = tmp_path/'model.pt'
    checkpoint.write_bytes(b'fixture')
    encoded = [dict(id='x', ids=[[0,3,7,8,1], [0,3,7,9,1]], prefix_length=3)]
    result = score_checkpoint(checkpoint, encoded, 'v')
    assert calls == [[[0,3,7]], [[0,3]]]
    assert result['records'][0]['log_odds'] == pytest.approx(1)
    assert result['records'][0]['context_gain'] == pytest.approx(-1)
    with pytest.raises(ValueError, match='vocabulary'):
        score_checkpoint(checkpoint, encoded, 'other')


def test_training_rejects_contaminated_initialization_and_changed_audit(tmp_path):
    from training.train import validate_combination_holdout
    payload = b'{}'
    (tmp_path/'holdout.json').write_bytes(payload)
    manifest = dict(combination_holdout=dict(report_sha256=hashlib.sha256(payload).hexdigest()))
    validate_combination_holdout({}, manifest, tmp_path)
    for config in [dict(initialize_from='prior.pt'), dict(repair_auxiliary=dict(data_dir='repairs'))]:
        with pytest.raises(ValueError, match='random initialization'):
            validate_combination_holdout(config, manifest, tmp_path)
    (tmp_path/'holdout.json').write_bytes(b'changed')
    with pytest.raises(ValueError, match='hash'):
        validate_combination_holdout({}, manifest, tmp_path)
    validate_combination_holdout(dict(initialize_from='prior.pt'), {}, tmp_path)
