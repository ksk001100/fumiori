import hashlib
import json
import torch
import pytest
from training.train import load_checkpoint
from training.average import average
from training.tests.test_model import small_config, exported


def test_aligned_averaging_preserves_sources_and_rejects_mismatched_initialization(tmp_path, exported):
    vocab = (exported / 'vocab.json').read_bytes()
    (tmp_path / 'vocab.json').write_bytes(vocab)
    row = dict(ids=[0,3,7,8,9,10,1], text='猫が眠る。', kana='ねこがねむる', trust='verified')
    payload = json.dumps(row, ensure_ascii=False).encode()
    hashes = {}
    for split in ('train', 'validation', 'test'):
        name = split + '.jsonl'; (tmp_path / name).write_bytes(payload)
        hashes[name] = hashlib.sha256(payload).hexdigest()
    (tmp_path / 'manifest.json').write_text(json.dumps(dict(vocab_sha256=hashlib.sha256(vocab).hexdigest(), split_sha256=hashes)), encoding='utf-8')
    paths = []
    base = dict(seed=17, threads=2, data_dir=str(tmp_path), steps=2, batch_size=1, log_every=2,
                learning_rate=.01, select_best_validation=True,
                model={k:v for k,v in vars(small_config()).items() if k not in ('vocab_size','version')})
    for i, rate in enumerate((.1, .25)):
        path = tmp_path / f'source-{i}.json'
        path.write_text(json.dumps(dict(base, checkpoint=str(tmp_path/f'source-{i}.pt'), noun_context_dropout=rate)), encoding='utf-8')
        paths.append(str(path))
    config = dict(base, checkpoint=str(tmp_path/'average.pt'), average_configs=paths, average_weights=[.5,.5])
    path = tmp_path / 'average.json'; path.write_text(json.dumps(config), encoding='utf-8')
    result = average(path)
    _, a = load_checkpoint(tmp_path/'source-0.pt')
    _, b = load_checkpoint(tmp_path/'source-1.pt')
    _, mean = load_checkpoint(config['checkpoint'])
    for name in mean['state_dict']:
        torch.testing.assert_close(mean['state_dict'][name], a['state_dict'][name]*.5+b['state_dict'][name]*.5, atol=0, rtol=0)
    assert len(result['averaging_sources']) == 2 and result['selected_step'] is None
    before = (tmp_path/'average.pt').read_bytes()
    path.write_text(json.dumps(dict(config, seed=18)), encoding='utf-8')
    with pytest.raises(ValueError, match='must match'):
        average(path)
    assert (tmp_path/'average.pt').read_bytes() == before
