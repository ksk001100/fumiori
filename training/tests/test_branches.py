from collections import Counter
import hashlib
import json

import pytest
import torch
from torch.nn import functional as F

from scripts.prepare_compact_words import prepare
from scripts.initialize_compact import initialize
from training.branches import branch_weights
from training.model import ModelConfig, Transformer, loss_for


def test_branch_balance_preserves_deterministic_steps_and_conditional_mass():
    rows = [dict(ids=[0, 4, 7, 9, 1])] * 9 + [dict(ids=[0, 4, 8, 9, 1])]
    weights, meta = branch_weights(rows, 6, strength=.5, max_ratio=100)
    torch.testing.assert_close(weights[:, 0], torch.ones(10))
    torch.testing.assert_close(weights[:, 2:4], torch.ones(10, 2))
    assert weights[-1, 1] > weights[0, 1]
    assert float(weights[:, 1].mean()) == pytest.approx(1.)
    assert (weights[:, 4] == 0).all()
    assert meta['branching_prefixes'] == 1
    unchanged, _ = branch_weights(rows, 6, strength=0)
    assert (unchanged[:, :4] == 1).all()


def test_prefixes_keep_difficulty_and_distant_context():
    rows = [dict(ids=[0, 3, 7, 9, 1]), dict(ids=[0, 4, 7, 8, 1]), dict(ids=[0, 4, 8, 7, 9, 1])]
    weights, _ = branch_weights(rows, 6)
    # Reusing the last word alone must not mix unrelated continuations.
    assert weights[0, 2] == weights[1, 2] == weights[2, 3] == 1
    for kwargs in [dict(strength=-1), dict(strength=float('nan')), dict(max_ratio=.5)]:
        with pytest.raises(ValueError):
            branch_weights(rows, 6, **kwargs)


def test_branch_distribution_uses_actual_sampling_mass_and_is_scale_invariant():
    rows=[dict(ids=[0,4,7,9,1]),dict(ids=[0,4,8,9,1])]
    uniform,_=branch_weights(rows,5,strength=.5,max_ratio=100)
    assert (uniform==1).all()
    corrected,metadata=branch_weights(rows,5,strength=.5,max_ratio=100,row_weights=[9.,1.])
    mass=torch.tensor([.9,.1])*corrected[:,1]
    torch.testing.assert_close(mass,torch.tensor([.75,.25]))
    torch.testing.assert_close(corrected[:,0],torch.ones(2))
    torch.testing.assert_close(corrected[:,2:],torch.ones(2,2))
    scaled,_=branch_weights(rows,5,strength=.5,max_ratio=100,row_weights=[90.,10.])
    torch.testing.assert_close(corrected,scaled)
    assert metadata['frequency_source']=='sampling_mass' and metadata['row_mass_sha256']
    for invalid in ([0.,1.],[-1.,2.],[float('nan'),1.],[1.]):
        with pytest.raises(ValueError,match='row weights'):
            branch_weights(rows,5,row_weights=invalid)


def test_v16_controls_only_change_branch_frequency_source():
    from pathlib import Path
    baseline=json.loads(Path('configs/everyday-v15-quality.json').read_text())
    for name,source in [('branch-count','uniform_rows'),('branch-mass','sampling_mass')]:
        config=json.loads(Path(f'configs/everyday-v16-{name}.json').read_text())
        branch=config.pop('branch_balance')
        assert branch==dict(strength=.5,max_ratio=4.,frequency_source=source)
        config.pop('checkpoint')
        expected=baseline.copy();expected.pop('checkpoint')
        assert config==expected


def test_weighted_loss_matches_explicit_token_objective_and_masks_padding():
    torch.set_num_threads(2)
    m = Transformer(ModelConfig(vocab_size=12, d_model=16, n_heads=4, n_layers=1, d_ff=32))
    batch = torch.tensor([[0, 3, 7, 1, 2], [0, 4, 8, 9, 1]])
    weights = torch.tensor([[1., 2., 3., 1000.], [1., 4., 2., 1.]])
    loss = loss_for(m, batch, token_weights=weights)
    per_token = F.cross_entropy(m(batch[:, :-1]).transpose(1, 2), batch[:, 1:], ignore_index=2, reduction='none')
    masked = weights.masked_fill(batch[:, 1:] == 2, 0)
    torch.testing.assert_close(loss, (per_token*masked).sum()/masked.sum())
    loss.backward()
    assert torch.isfinite(m.token_emb.weight.grad).all()
    with pytest.raises(ValueError):
        loss_for(m, batch, token_weights=torch.zeros_like(weights))


def test_compact_vocabulary_uses_train_only_and_preserves_source(tmp_path):
    source = tmp_path/'source'
    source.mkdir()
    def row(word):
        return dict(text=word, kana=word, ids=[0, 3, 7, 1], segments=[dict(surface=word, kana=word, pos='名詞')])
    checksums = {}
    for name, words in [('train', ['あ', 'い']), ('validation', ['あ', 'う']), ('test', ['い'])]:
        payload = ''.join(json.dumps(row(w), ensure_ascii=False)+'\n' for w in words).encode()
        (source/(name+'.jsonl')).write_bytes(payload)
        checksums[name+'.jsonl'] = hashlib.sha256(payload).hexdigest()
    (source/'manifest.json').write_text(json.dumps(dict(split_sha256=checksums, source=dict(license='test'))))
    output = tmp_path/'prepared'
    result = prepare(source, output)
    vocab = json.loads((output/'vocab.json').read_text(encoding='utf-8'))
    assert Counter(t['surface'] for t in vocab['tokens'][7:]) == Counter(['あ', 'い'])
    assert result['splits']['validation'] == 1
    manifest = json.loads((output/'manifest.json').read_text(encoding='utf-8'))
    assert manifest['source']['license'] == 'test'
    assert manifest['compact_word_stats']['validation.jsonl']['rejected'] == {'unseen_word': 1}
    with pytest.raises(ValueError, match='new directory'):
        prepare(source, output)


def test_compact_initialization_checks_vocabulary_and_records_parent_hash(tmp_path):
    data = tmp_path/'data'
    data.mkdir()
    vocab = b'{"tokens":[]}'
    (data/'vocab.json').write_bytes(vocab)
    checkpoint = tmp_path/'parent.pt'
    state = {'weight': torch.tensor([1., 2.])}
    torch.save(dict(state_dict=state, data_manifest=dict(vocab_sha256=hashlib.sha256(vocab).hexdigest())), checkpoint)
    output = tmp_path/'init.pt'
    source = initialize(checkpoint, data, output)
    assert source['revision'] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    ck = torch.load(output, weights_only=True)
    torch.testing.assert_close(ck['state_dict']['weight'], state['weight'])
    with pytest.raises(ValueError, match='already exists'):
        initialize(checkpoint, data, output)
    (data/'vocab.json').write_bytes(b'changed')
    with pytest.raises(ValueError, match='vocabulary differs'):
        initialize(checkpoint, data, tmp_path/'other.pt')
