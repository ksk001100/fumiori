import json

import pytest

from scripts import prepare_word_teacher as prep
from scripts.teacher_prose import prompt_for


def segments():
    return [dict(surface=s,kana=k,pos=p) for s,k,p in [('花','はな','名詞'),('を','を','助詞'),('見る','みる','動詞'),('。','','記号')]]


@pytest.fixture
def encoding(monkeypatch):
    parts=segments()
    monkeypatch.setattr(prep,'analyze',lambda *args:parts)
    monkeypatch.setattr(prep,'lexical_triples',lambda *args:set())
    mapping={(s['surface'],s['kana'],s['pos']):i+7 for i,s in enumerate(parts)}
    return mapping,set()


def test_frozen_word_encoding_roundtrip_and_protected_bodies(encoding):
    row,reason=prep.encode_candidate('花を見る。',encoding,None,None,set(),set())
    assert reason is None and row['ids']==[0,3,7,8,9,10,1]
    assert row['text']==''.join(s['surface'] for s in row['segments'])
    assert row['kana']==''.join(s['kana'] for s in row['segments'])
    assert prep.encode_candidate('花を見る。',encoding,None,None,{prep.canonical('花を見る')},set())[1]=='duplicate_or_protected'
    assert prep.canonical('今日花を見る。')==prep.canonical('花を見る。')


def test_no_oov_guessing_or_ambiguous_reading_admission(encoding):
    mapping,_=encoding
    del mapping['花','はな','名詞']
    assert prep.encode_candidate('花を見る。',encoding,None,None,set(),set())[1]=='out_of_vocabulary'
    assert prep.encode_candidate('花を見る。',(mapping,{('花','名詞')}),None,None,set(),set())[1]=='ambiguous_lexeme_reading'


def test_combination_and_instruction_exclusions(encoding,monkeypatch):
    target=('花','を','見る')
    monkeypatch.setattr(prep,'lexical_triples',lambda *args:{target})
    assert prep.encode_candidate('花を見る。',encoding,None,None,set(),{target})[1]=='protected_combination'
    assert prep.encode_candidate('以下の例文です。',encoding,None,None,set(),set())[1]=='instruction_artifact'


def test_report_protection_reads_outputs_not_nearest_training_strings():
    report=dict(results=[dict(metrics=dict(details=[dict(text='花を見る',nearest_text='本を読む')]))])
    assert list(prep.report_texts(report))==['花を見る']


def test_fewshot_prompts_can_use_only_the_explicit_filtered_source(tmp_path):
    path=tmp_path/'train.jsonl'
    rows=[dict(text=f'見本{i}。',kana='あ'*length) for length in (6,15,30) for i in range(4)]
    path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
    _,prompt=prompt_for(0,42,examples_path=path)
    assert '見本' in prompt and '犬が' not in prompt


def test_common_reference_includes_new_training_but_never_heldout_rows():
    from scripts.evaluate_candidates import validate_common_reference
    old=[dict(text='花を見る')];new=old+[dict(text='本を読む。')]
    validate_common_reference(new,[old,new])
    with pytest.raises(ValueError):
        validate_common_reference(old,[old,new])
    with pytest.raises(ValueError):
        validate_common_reference(new+[dict(text='評価文')],[old,new])


def test_v17_changes_data_only_with_fixed_final_step_and_capacity():
    from pathlib import Path
    control=json.loads(Path('configs/everyday-v15-quality.json').read_text())
    for name in ('small','expanded'):
        config=json.loads(Path(f'configs/everyday-v17-{name}.json').read_text())
        assert config['data_dir']==f'build/everyday-v17-{name}-data'
        assert config['select_best_validation'] is False
        assert not config.get('initialize_from') and not config.get('repair_auxiliary')
        for key in ('data_dir','checkpoint','select_best_validation'):config.pop(key)
        expected=control.copy()
        for key in ('data_dir','checkpoint','select_best_validation'):expected.pop(key)
        assert config==expected


def test_preparation_rejects_fewshot_rows_from_heldout_data(tmp_path,monkeypatch):
    import hashlib
    import sys
    from types import SimpleNamespace
    monkeypatch.setitem(sys.modules,'sudachipy',SimpleNamespace(
        Dictionary=lambda:SimpleNamespace(tokenizer=lambda:None),SplitMode=SimpleNamespace(C=None)))
    monkeypatch.setattr(prep.importlib.metadata,'version',lambda name:'fixture')
    monkeypatch.setattr(prep,'load_pairs',lambda *args:[])
    base=tmp_path/'base';base.mkdir()
    train=json.dumps(dict(text='学習文。'))+'\n';heldout=json.dumps(dict(text='保留文。'))+'\n'
    (base/'train.jsonl').write_text(train,encoding='utf-8')
    (base/'vocab.json').write_text(json.dumps(dict(tokens=[])),encoding='utf-8')
    (base/'holdout.json').write_text('{}',encoding='utf-8')
    digest=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    manifest=dict(vocab_sha256=digest(base/'vocab.json'),split_sha256={'train.jsonl':digest(base/'train.jsonl')},
                  combination_holdout=dict(report_sha256=digest(base/'holdout.json')),dictionary=dict(version='fixture'))
    (base/'manifest.json').write_text(json.dumps(manifest),encoding='utf-8')
    examples=tmp_path/'examples.jsonl';examples.write_text(heldout,encoding='utf-8')
    raw=tmp_path/'raw.jsonl';raw.write_text('',encoding='utf-8')
    raw.with_suffix('.manifest.json').write_text(json.dumps(dict(raw_sha256=digest(raw),fewshot_training_path=str(examples),fewshot_training_sha256=digest(examples))),encoding='utf-8')
    config=tmp_path/'config.json'
    config.write_text(json.dumps(dict(base=str(base),protect_reports=[],pairs='fixture',raw=[str(raw)],
        variants=[dict(merged=str(tmp_path/'merged'),output=str(tmp_path/'output'))])),encoding='utf-8')
    with pytest.raises(ValueError,match='filtered train'):
        prep.prepare(config)
    assert not (tmp_path/'output').exists() and not (tmp_path/'merged').exists()
