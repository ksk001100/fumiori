"""Retain pretrained subword embeddings and attach an explicit word lexicon."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sentencepiece as spm
import torch
from safetensors.torch import load_file

from scripts.initialize_pretrained import transfer_body
from scripts.lexical_rules import lexical_collision
from training.model import ModelConfig, Transformer


def main():
    source = Path('build/pretrained/rinna-xsmall')
    origin = json.loads((source/'source.json').read_text())
    for name, info in origin['files'].items():
        if hashlib.sha256((source/name).read_bytes()).hexdigest() != info['sha256']:
            raise ValueError('pretrained input changed')
    data = Path('build/everyday-v4-data')
    parent_bytes = (data/'manifest.json').read_bytes()
    manifest = json.loads(parent_bytes)
    for name, expected in manifest['split_sha256'].items():
        if hashlib.sha256((data/name).read_bytes()).hexdigest() != expected:
            raise ValueError('word data changed')
    word_vocab = json.loads((data/'vocab.json').read_text(encoding='utf-8'))
    sp = spm.SentencePieceProcessor(model_file=str(source/'spiece.model'))
    parts, frequency, counts, stats = {}, Counter(), Counter(), Counter()
    for split in ('train','validation','test','family-test'):
        parts[split] = []
        for line in (data/(split+'.jsonl')).read_text(encoding='utf-8').splitlines():
            row = json.loads(line)
            if any(lexical_collision(a,b) for a,b in zip(row['segments'],row['segments'][1:])):
                stats['lexical_boundary_'+split] += 1; continue
            raw_ids = sp.encode(row['text'],out_type=int)
            if sp.unk_id() in raw_ids or len(raw_ids)+3 > 64:
                stats['unrepresentable_'+split] += 1; continue
            row['raw_piece_ids'] = raw_ids
            parts[split].append(row)
            if split == 'train':
                frequency.update(raw_ids)
                counts.update((s['surface'],s['kana'],s['pos']) for s in row['segments'])
    required = set(frequency)
    for word in word_vocab['tokens'][7:]:
        required.update(sp.encode(word['surface'],out_type=int))
        required.update(sp.encode('。'+word['surface'],out_type=int))
    required -= set(range(7))
    if len(required) > 15993:
        raise ValueError('required subwords exceed runtime budget')
    kept = list(sorted(required))
    for i in range(7,sp.get_piece_size()):
        piece = sp.id_to_piece(i)
        if i not in required and len(kept) < 15993 and all(c == '▁' or '\u3041' <= c <= '\u30ff' or '\u4e00' <= c <= '\u9fff' or c in '々、。' for c in piece):
            kept.append(i)
    remap = {old:i+7 for i,old in enumerate(kept)}
    specials = ['BOS','EOS','PAD','EASY','NORMAL','HARD','UNK']
    tokens = [dict(id=i,surface=s,kana='',pos='special') for i,s in enumerate(specials)]
    tokens += [dict(id=i+7,surface=sp.id_to_piece(old),kana='',pos='subword') for i,old in enumerate(kept)]
    lexicon = word_vocab['tokens'][7:]
    for word in lexicon:
        word['metadata'] = dict(count=counts[(word['surface'],word['kana'],word['pos'])])
    vocab = dict(version='0.1.0',encoding='sentencepiece_lexicon_v1',tokens=tokens,lexicon=lexicon)
    payload = json.dumps(vocab,ensure_ascii=False,separators=(',',':')).encode()
    out = Path('build/everyday-v6-data'); out.mkdir(parents=True,exist_ok=True)
    (out/'vocab.json').write_bytes(payload)
    result = dict(manifest, description='Pretrained subword language model with dictionary-only output spelling',
                  parent_manifest_sha256=hashlib.sha256(parent_bytes).hexdigest(),
                  vocab_sha256=hashlib.sha256(payload).hexdigest(), splits={}, split_sha256={},
                  subword_preparation_stats=dict(stats), source_generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    for split,rows in parts.items():
        accepted = []
        for row in rows:
            raw_ids = row.pop('raw_piece_ids')
            if any(i not in remap for i in raw_ids):
                continue
            row['ids'] = [0,row['ids'][1],*[remap[i] for i in raw_ids],1]
            accepted.append(row)
        blob = ''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in accepted).encode()
        (out/(split+'.jsonl')).write_bytes(blob)
        result['splits'][split] = len(accepted)
        result['split_sha256'][split+'.jsonl'] = hashlib.sha256(blob).hexdigest()
    (out/'manifest.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    raw_config = json.loads((source/'config.json').read_text())
    config = ModelConfig(vocab_size=len(tokens),d_model=512,n_heads=8,n_layers=6,d_ff=2304,
                         max_seq_len=64,dropout=.1,approximate_gelu=True)
    model = Transformer(config); state=model.state_dict()
    weights = load_file(source/'model.safetensors')
    for i, old in enumerate([raw_config['bos_token_id'],raw_config['eos_token_id'],3,1,1,1,0]+kept):
        state['token_emb.weight'][i] = weights['transformer.wte.weight'][old]
    transfer_body(weights,config,state)
    model.load_state_dict(state,strict=True)
    path = Path('build/checkpoints/rinna-subword-initialization.pt')
    torch.save(dict(state_dict=state,model_config=vars(config),vocab_sha256=result['vocab_sha256'],
                    source=dict(**origin,mapping='Retained exact pretrained subword embeddings and transformer; vocabulary pruned, first 64 positions retained.')),path)
    print(json.dumps(dict(splits=result['splits'],pieces=len(tokens),words=len(lexicon),stats=stats,parameter_count=model.parameter_count())),flush=True)


if __name__ == '__main__':
    main()
