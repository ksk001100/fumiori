import json
from pathlib import Path

import pytest

from scripts.build_wasm import load_catalog
from scripts.evaluate_candidates import validate_pool_report
from scripts.review_candidates import summarize
from scripts.novelty_evaluate import review_selection


def test_catalog_retains_all_published_model_variants():
    catalog = load_catalog('configs/demo-models.json')
    assert list(catalog) == ['model-int8','model-experimental','model-v14-ce','model-v14-smooth',
                             'model-v15-base','model-v15-context','model-v15-quality','model-v15-quality-context',
                             'model-v16-branch-count','model-v16-branch-mass','model-v17-small','model-v17-expanded','model-v18-expanded-base','model-v19-curated','model-v19-refined',
                             'model-v20-control4','model-v20-layers4','model-v20-layers2','model-v20-layers8']


@pytest.mark.parametrize('identifier', ['../model','model-../../pkg','model-good/model-bad','pkg'])
def test_catalog_cannot_overwrite_assets(tmp_path,identifier):
    path = tmp_path/'catalog.json'
    path.write_text(json.dumps([dict(id=identifier,directory='models/stories-v2-diverse-3mb',label='label',description='limits')]))
    with pytest.raises(ValueError,match='invalid'):
        load_catalog(path)


def test_pool_accounting_keeps_generation_failures():
    raw=dict(variants=[dict(samples=[dict(request_id=0)],failures=[dict(request_id=1)],success=1)])
    validate_pool_report(raw,2)
    raw['variants'][0]['failures'][0]['request_id']=0
    with pytest.raises(ValueError):
        validate_pool_report(raw,2)


def test_holdout_controls_differ_only_in_label_smoothing_and_output():
    controls=[json.loads(Path(f'configs/everyday-v14-holdout-{name}.json').read_text()) for name in ('ce','smooth')]
    assert [c.pop('label_smoothing') for c in controls] == [0,.05]
    for control in controls:
        assert not control.get('initialize_from')
        assert not control.get('repair_auxiliary')
        control.pop('checkpoint')
    assert controls[0]==controls[1]


def test_blind_review_is_complete_and_sample_bound():
    selected=review_selection([dict(status='success',text='猫',kana='ねこ')],1,0)
    report=dict(results=[dict(name='ce',review_selection=selected)])
    mapping=[dict(blind_id=0,key='ce:0')]
    ratings=[dict(blind_id='0',status='success',text='猫',kana='ねこ',natural='1',reading_ok='1',
                  reviewer='Codex',reviewer_kind='agent',notes='語と読みが対応する')]
    assert summarize(report,mapping,ratings)['ce']['by_reviewer_kind']['agent']['joint_rate']==1
    with pytest.raises(ValueError):
        summarize(report,mapping,[])
    ratings[0]['kana']='いぬ'
    with pytest.raises(ValueError):
        summarize(report,mapping,ratings)


def test_semantic_probe_does_not_force_a_target_verb(tmp_path,monkeypatch):
    import torch
    import training.train
    from scripts.probe_combination_generation import probe

    class EndImmediately:
        config=type('Config',(),dict(max_seq_len=16))()

        def __call__(self,ids):
            logits=torch.zeros(1,ids.shape[1],10)
            logits[:,:,1]=10  # EOS ID 1, without a target predicate.
            return logits

    vocab=dict(tokens=[dict(surface='',kana='') for _ in range(7)]+[
        dict(surface='猫',kana='ねこ'),dict(surface='を',kana='を'),dict(surface='見る',kana='みる')])
    checkpoint=tmp_path/'checkpoint.pt'
    checkpoint.write_bytes(b'fixture')
    ck=dict(vocab=vocab,data_manifest=dict(vocab_sha256='fixture'))
    monkeypatch.setattr(training.train,'load_checkpoint',lambda path:(EndImmediately(),ck))
    audit=dict(vocab_sha256='fixture',cases=[dict(id='a',prefix='猫を',positive='猫を見る',negative='猫を飲む')],
               encoded=[dict(id='a',ids=[[0,3,7,8,9,1]])])
    path=tmp_path/'audit.json'
    path.write_text(json.dumps(audit),encoding='utf-8')
    result=probe(path,[str(checkpoint)])['results'][0]['samples'][0]
    assert result['text']=='猫を' and result['terminated']
    assert '見る' not in result['text']  # EOS is not silently repaired into the reference.
    audit['cases'][0]['prefix']='犬を'
    path.write_text(json.dumps(audit),encoding='utf-8')
    assert probe(path,[str(checkpoint)])['results'][0]['samples'][0]['error']=='prefix not on a token boundary'
