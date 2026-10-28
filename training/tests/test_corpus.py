"""Training-data weighting and checkpoint selection invariants."""
import hashlib
import json
from training.train import train, load_checkpoint, sampling_weights
from training.model import Transformer
from training.tests.test_model import small_config, exported
import torch


def test_difficulty_balancing_uses_only_observed_training_rows():
    rows = [dict(ids=[0, d, 7, 1]) for d in [3] * 10 + [4] * 30 + [5] * 2]
    weights = sampling_weights(rows)
    for d in (3, 4, 5):
        assert abs(sum(float(w) for r, w in zip(rows, weights) if r['ids'][1] == d) - 1) < 1e-12
    assert torch.isfinite(sampling_weights(rows[:10])).all()


def test_early_stopping_restores_best_checkpoint(tmp_path,exported):
    vocab_bytes=(exported/'vocab.json').read_bytes()
    (tmp_path/'vocab.json').write_bytes(vocab_bytes)
    row=dict(ids=[0,3,7,8,9,10,1],text='猫が眠る。',kana='ねこがねむる',trust='verified')
    payload=json.dumps(row,ensure_ascii=False).encode()
    hashes={}
    for split in ('train','validation','test'):
        (tmp_path/f'{split}.jsonl').write_bytes(payload)
        hashes[f'{split}.jsonl']=hashlib.sha256(payload).hexdigest()
    manifest=dict(vocab_sha256=hashlib.sha256(vocab_bytes).hexdigest(),split_sha256=hashes)
    (tmp_path/'manifest.json').write_text(json.dumps(manifest),encoding='utf-8')
    config=dict(seed=42,threads=2,data_dir=str(tmp_path),checkpoint=str(tmp_path/'selected.pt'),steps=10,batch_size=1,log_every=1,learning_rate=0.01,
        select_best_validation=True,early_stopping_patience=1,validation_min_delta=100,
        model={k:v for k,v in vars(small_config()).items() if k not in ('vocab_size','version')})
    config_path=tmp_path/'config.json'
    config_path.write_text(json.dumps(config),encoding='utf-8')
    summary=train(config_path)
    assert summary['stopped_early'] and summary['selected_step']==0
    assert summary['logs'][-1]['step']==1
    restored,_=load_checkpoint(config['checkpoint'])
    torch.manual_seed(42)
    initial=Transformer(small_config())
    for name,tensor in initial.state_dict().items():
        torch.testing.assert_close(tensor,restored.state_dict()[name],atol=0,rtol=0)
