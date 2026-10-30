import torch
import pytest

from training.train import noun_context, state_digest, training_generators
from training.model import ModelConfig, Transformer


def test_masking_does_not_advance_data_or_model_dropout_streams():
    batch=torch.tensor([[0,3,7,8,9,1]])
    nouns=torch.zeros(12,dtype=torch.bool)
    nouns[7]=True
    sequences=[]
    for mask in (False,True):
        torch.manual_seed(42)
        data,context=training_generators(dict(seed=42,independent_data_rng=True),'cpu')
        draws=[]
        for _ in range(5):
            draws.append(torch.multinomial(torch.ones(10),4,replacement=True,generator=data))
            if mask:
                original=batch.clone()
                noun_context(batch,nouns,.5,context)
                assert torch.equal(batch,original)
        sequences.append((torch.stack(draws),torch.rand(5)))
    torch.testing.assert_close(sequences[0][0],sequences[1][0],rtol=0,atol=0)
    torch.testing.assert_close(sequences[0][1],sequences[1][1],rtol=0,atol=0)


def test_legacy_rng_and_invalid_options():
    assert training_generators(dict(seed=42),'cpu')==(None,None)
    with pytest.raises(ValueError):
        training_generators(dict(seed=42,independent_data_rng=1),'cpu')


def test_initial_state_hash_tracks_weights_without_changing_them():
    torch.manual_seed(42)
    model=Transformer(ModelConfig(vocab_size=12,d_model=16,n_heads=4,n_layers=1,d_ff=32))
    initial=state_digest(model)
    assert initial==state_digest(model)
    with torch.no_grad():
        model.token_emb.weight[0,0]+=1
    assert initial!=state_digest(model)


def test_v15_factorial_controls_keep_validation_and_capacity_fixed():
    import json
    from pathlib import Path
    variants={name:json.loads(Path(f'configs/everyday-v15-{name}.json').read_text())
              for name in ('base','context','quality','quality-context')}
    for config in variants.values():
        assert config['independent_data_rng'] is True
        assert not config.get('initialize_from') and not config.get('repair_auxiliary')
    assert variants['base']['validation_corpus_mixture']==variants['quality']['validation_corpus_mixture']
    for left,right in (('base','context'),('quality','quality-context')):
        a,b=variants[left].copy(),variants[right].copy()
        assert a.pop('noun_context_dropout')==0
        assert b.pop('noun_context_dropout')==.1
        a.pop('checkpoint');b.pop('checkpoint')
        assert a==b
    a,b=variants['base'].copy(),variants['quality'].copy()
    assert a.pop('corpus_mixture')!=b.pop('corpus_mixture')
    a.pop('checkpoint');b.pop('checkpoint')
    assert a==b
