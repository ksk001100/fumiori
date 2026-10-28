import json
import pytest
from training.train import learning_rate_at
from training.package import package
from training.tests.test_model import exported, cli


def test_warmup_cosine_and_legacy_constant_schedule():
    c = dict(learning_rate=.001, steps=100, warmup_steps=10, lr_schedule='cosine', min_lr_ratio=.1)
    assert learning_rate_at(c,0) == .0001
    assert learning_rate_at(c,9) == .001
    assert learning_rate_at(c,10) == .001
    assert abs(learning_rate_at(c,100)-.0001) < 1e-12
    assert learning_rate_at(c,50) > learning_rate_at(c,90)
    assert learning_rate_at(dict(learning_rate=.01,steps=0),0) == .01
    with pytest.raises(ValueError):
        learning_rate_at(dict(c,lr_schedule='wrong'),0)


def test_metadata_compaction_keeps_ids_and_parity(exported, tmp_path):
    path = exported/'vocab.json'
    original = json.loads(path.read_text(encoding='utf-8'))
    for token in original['tokens']:
        token['metadata'] = dict(provenance='training-only detail')
    path.write_text(json.dumps(original),encoding='utf-8')
    import hashlib
    manifest_path = exported/'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    manifest['vocab_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest),encoding='utf-8')
    target = tmp_path/'runtime'
    package(exported,target,strip_token_metadata=True)
    compact = json.loads((target/'vocab.json').read_text(encoding='utf-8'))
    assert [{k:v for k,v in t.items() if k!='metadata'} for t in original['tokens']] == compact['tokens']
    import shutil
    shutil.copyfile(exported/'parity.json',target/'parity.json')
    assert cli('parity','--model',target).returncode == 0


def test_story_alignment_rejects_unknown_kanji_and_preserves_readings():
    # Optional importer must not become a required training/runtime dependency.
    pytest.importorskip('pyarrow')
    from scripts.prepare_stories import lexicon, Aligner
    specials = [dict(id=i,surface=s,kana='',pos='special',metadata={}) for i,s in enumerate(['BOS','EOS','PAD','EASY','NORMAL','HARD','UNK'])]
    tokens = specials + [dict(id=7+i,surface=s,kana=k,pos=p,metadata={}) for i,(s,k,p) in enumerate([
        ('猫','ねこ','名詞'),('が','が','助詞'),('眠る','ねむる','動詞'),('行く','いく','動詞'),('。','','記号')])]
    vocab = lexicon(dict(version='0.1.0',encoding='UTF-8',tokens=tokens))
    aligner = Aligner(vocab)
    row = aligner.align('猫が眠りました。')
    assert row['kana'] == 'ねこがねむりました'
    assert row['text'] == ''.join(s['surface'] for s in row['segments'])
    assert aligner.align('象が眠りました。') is None
    assert aligner.align('猫が眠って。') is None
    assert aligner.align('猫が行きました。')['kana'] == 'ねこがいきました'
    for text in ('まりやが眠りました。', 'トモが眠りました。', '猫の名前はれいなでした。'):
        assert aligner.align(text) is None
    from scripts.prepare_stories import lexical_collision
    assert lexical_collision(dict(surface='風',pos='名詞'),dict(surface='船',pos='名詞'))
    assert lexical_collision(dict(surface='光',pos='名詞'),dict(surface='る',pos='文字'))
    assert not lexical_collision(dict(surface='風船',pos='名詞'),dict(surface='が',pos='助詞'))


def test_review_gate_rejects_stale_partial_or_low_quality_ratings():
    import hashlib
    from scripts.stories_gate import review_gate
    samples=[dict(text=f'新しい文{i}',kana='あたらしいぶん') for i in range(600)]
    subset=[dict(sample_id=i,**samples[i]) for i in range(100)]
    payload=json.dumps(subset,ensure_ascii=False).encode()
    report=dict(weights_sha256='weights',decoder_sha256='decoder',review_sample_ids=list(range(100)),
        samples=samples,structural_gate=dict(passed=True))
    review=dict(weights_sha256='weights',decoder_sha256='decoder',sample_sha256=hashlib.sha256(payload).hexdigest(),
        reviewer='fixture',reviewer_type='test fixture',ratings=[dict(sample_id=i,natural=True,reading_ok=True) for i in range(100)])
    assert review_gate(report,review,payload,[])['passed']
    with pytest.raises(ValueError,match='decoding policy'):
        review_gate(report,dict(review,decoder_sha256='stale'),payload,[])
    with pytest.raises(ValueError,match='exactly once'):
        review_gate(report,dict(review,ratings=review['ratings'][:-1]),payload,[])
    low=[dict(r,natural=r['sample_id']>=11) for r in review['ratings']]
    assert not review_gate(report,dict(review,ratings=low),payload,[])['passed']
