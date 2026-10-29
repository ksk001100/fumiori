import hashlib
import json
import sys
from types import ModuleType, SimpleNamespace
import torch
from torch.nn import functional as F
import pytest

from training.repairs import RepairData, repair_loss
from scripts.prepare_repairs import prepare


def test_repair_loss_ignores_prefix_targets_and_padding():
    batch = torch.tensor([[0,4,7,8,1,2],[0,4,7,9,10,1]])
    lengths = torch.tensor([3,3])
    rejected = torch.tensor([11,11])
    logits = torch.randn(2,5,12,requires_grad=True)
    positive, negative = repair_loss(logits,batch,lengths,rejected)
    per = F.cross_entropy(logits.transpose(1,2),batch[:,1:],ignore_index=2,reduction='none')
    expected = (per[0,2:4].mean()+per[1,2:5].mean())/2
    torch.testing.assert_close(positive,expected)
    p = logits[:,2].softmax(-1)[:,11]
    torch.testing.assert_close(negative,-torch.log1p(-p).mean())
    negative.backward()
    assert (logits.grad[:,2,11]>0).all()
    assert torch.count_nonzero(logits.grad[:,:2])==0
    assert torch.count_nonzero(logits.grad[:,3:])==0


def test_repair_data_checks_hashes_and_uniform_group_mass(tmp_path):
    a = dict(prefix_ids=[0,4,7],rejected_token=11,positives=[dict(ids=[0,4,7,8,1]),dict(ids=[0,4,7,9,1])])
    b = dict(prefix_ids=[0,4,8],rejected_token=11,positives=[dict(ids=[0,4,8,7,1])])
    payload=''.join(json.dumps(g)+'\n' for g in [a,b]).encode()
    (tmp_path/'train.jsonl').write_bytes(payload)
    manifest=dict(vocab_sha256='v',parent_manifest_sha256='p',split_sha256={'train.jsonl':hashlib.sha256(payload).hexdigest()})
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    data=RepairData(tmp_path,'v','p','cpu')
    assert data.weights.tolist()==[.5,.5,1.]
    with pytest.raises(ValueError,match='provenance'):
        RepairData(tmp_path,'other','p','cpu')
    (tmp_path/'train.jsonl').write_text('changed')
    with pytest.raises(ValueError,match='hash'):
        RepairData(tmp_path,'v','p','cpu')


def test_preparation_excludes_parent_holdout_and_rejects_wrong_rollout_seeds(tmp_path,monkeypatch):
    sudachipy = ModuleType('sudachipy')
    sudachipy.Dictionary = lambda: SimpleNamespace(tokenizer=lambda: None)
    sudachipy.SplitMode = SimpleNamespace(C=object())
    monkeypatch.setitem(sys.modules, 'sudachipy', sudachipy)
    import scripts.prepare_tatoeba
    monkeypatch.chdir(tmp_path)
    roots=[chr(0x4e00+i) for i in range(100)]
    bucket=lambda s:int(hashlib.sha256((s+'は').encode()).hexdigest()[:8],16)%5
    train_root=next(s for s in roots if bucket(s)!=0)
    held_root=next(s for s in roots if bucket(s)==0)
    words=[(train_root,'ねこ','名詞'),(held_root,'いぬ','名詞'),('は','は','助詞'),
           ('食べる','たべる','動詞'),('寝る','ねる','動詞'),('鳴く','なく','動詞'),('走る','はしる','動詞'),('。','','記号')]
    tokens=[dict(id=i,surface=s,kana=k,pos=p) for i,(s,k,p) in enumerate(words,7)]
    segments={t['surface']:{k:t[k] for k in ('surface','kana','pos')} for t in tokens}
    def analyze(text,*_):
        return [segments[text[0]],segments['は'],segments[text[2:-1]],segments['。']]
    monkeypatch.setattr(scripts.prepare_tatoeba,'analyze',analyze)
    data=tmp_path/'data';data.mkdir()
    vocab=json.dumps(dict(tokens=[{}]*7+tokens),ensure_ascii=False).encode()
    (data/'vocab.json').write_bytes(vocab)
    hashes={}
    for name,rows in [('train',[]),('validation',[dict(text=train_root+'は寝る。')]),('test',[])]:
        payload=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows).encode()
        (data/(name+'.jsonl')).write_bytes(payload)
        hashes[name+'.jsonl']=hashlib.sha256(payload).hexdigest()
    (data/'manifest.json').write_text(json.dumps(dict(vocab_sha256=hashlib.sha256(vocab).hexdigest(),split_sha256=hashes)))
    build=tmp_path/'build';build.mkdir()
    for source,root,start,count,path in [('quality',train_root,810000,256,'v12-training-rollouts.json'),('ce',held_root,811000,384,'v12-ce-training-rollouts.json')]:
        samples=[dict(seed=start+i,text=root+'は食べる',kana='ねこはたべる',segments=[segments[root],segments['は'],segments['食べる']]) for i in range(count)]
        (build/path).write_text(json.dumps(dict(seed_start=start,seed_end=start+count-1,requested=count,history_size=0,samples=samples),ensure_ascii=False),encoding='utf-8')
    annotations=tmp_path/'edits.json'
    annotations.write_text(json.dumps([[s,0,r+'は',['寝る。','鳴く。','走る。']] for s,r in [('quality',train_root),('ce',held_root)]],ensure_ascii=False),encoding='utf-8')
    result=prepare(data,annotations,tmp_path/'prepared')
    assert result['groups']==dict(train=1,validation=1)
    assert result['exclusions']['heldout_body_excluded']==1
    records=[json.loads(s) for s in (tmp_path/'prepared/train.jsonl').read_text(encoding='utf-8').splitlines()]
    assert all(p['text']!=train_root+'は寝る。' for g in records for p in g['positives'])
    path=build/'v12-training-rollouts.json';r=json.loads(path.read_text(encoding='utf-8'));r['seed_start']=820000
    path.write_text(json.dumps(r),encoding='utf-8')
    with pytest.raises(ValueError,match='training seed'):
        prepare(data,annotations,tmp_path/'wrong')
