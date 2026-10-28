"""Map public GPT-2 tensors into the local word model; no remote code is loaded."""
import hashlib
import json
from pathlib import Path
import sentencepiece as spm
import torch
from safetensors.torch import load_file

from training.model import ModelConfig, Transformer


def transfer_body(weights, config, state):
    state['pos_emb.weight'] = weights['transformer.wpe.weight'][:64].clone()
    for i in range(config.n_layers):
        source_prefix = f'transformer.h.{i}.'
        target_prefix = f'blocks.{i}.'
        mappings = {'ln_1':'ln1','ln_2':'ln2','attn.c_attn':'qkv','attn.c_proj':'attn_out','mlp.c_fc':'fc1','mlp.c_proj':'fc2'}
        for old, new in mappings.items():
            for suffix in ('weight','bias'):
                tensor = weights[source_prefix+old+'.'+suffix]
                if suffix == 'weight' and old not in ('ln_1','ln_2'):
                    tensor = tensor.t()
                state[target_prefix+new+'.'+suffix] = tensor.contiguous()
    for suffix in ('weight','bias'):
        state['final_ln.'+suffix] = weights['transformer.ln_f.'+suffix]


def main():
    source = Path('build/pretrained/rinna-xsmall')
    provenance = json.loads((source/'source.json').read_text())
    for name, info in provenance['files'].items():
        if hashlib.sha256((source/name).read_bytes()).hexdigest() != info['sha256']:
            raise ValueError(f'pretrained source changed: {name}')
    raw = json.loads((source/'config.json').read_text())
    if raw['activation_function'] != 'gelu_new' or raw['layer_norm_epsilon'] != 1e-5:
        raise ValueError('unsupported pretrained architecture')
    vocab_path = Path('build/everyday-v4-data/vocab.json')
    vocab = json.loads(vocab_path.read_text(encoding='utf-8'))
    config = ModelConfig(vocab_size=len(vocab['tokens']), d_model=raw['n_embd'], n_heads=raw['n_head'],
                         n_layers=raw['n_layer'], d_ff=raw['n_inner'], max_seq_len=64,
                         dropout=.1, approximate_gelu=True)
    model = Transformer(config)
    weights = load_file(source/'model.safetensors')
    state = model.state_dict()
    sp = spm.SentencePieceProcessor(model_file=str(source/'spiece.model'))
    original_emb = weights['transformer.wte.weight']
    unknown = 0
    for token in vocab['tokens']:
        i = token['id']
        if i < 7:
            indices = [raw['eos_token_id']] if i == 1 else [raw['bos_token_id']]
        else:
            indices = sp.encode(token['surface'], out_type=int)
            unknown += int(sp.unk_id() in indices)
            if not indices:
                raise ValueError('empty pretrained word encoding')
        state['token_emb.weight'][i] = original_emb[indices].mean(0)
    transfer_body(weights, config, state)
    model.load_state_dict(state, strict=True)
    result = dict(state_dict=state, model_config=vars(config),
                  vocab_sha256=hashlib.sha256(vocab_path.read_bytes()).hexdigest(),
                  source=dict(**provenance, mapping='Mean of source SentencePiece embeddings for each dictionary word; '
                              'GPT-2 Conv1D tensors transposed; first 64 position embeddings retained.',
                              words_with_unknown_subpieces=unknown))
    path = Path('build/checkpoints/rinna-word-initialization.pt')
    path.parent.mkdir(parents=True,exist_ok=True)
    torch.save(result,path)
    print(json.dumps(dict(parameter_count=model.parameter_count(), words_with_unknown_subpieces=unknown, model_config=vars(config))))


if __name__ == '__main__':
    main()
