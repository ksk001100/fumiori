"""Prune LLM-jp's unigram vocabulary; retain all 12 pretrained Llama layers."""
import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import re


def prepare(output, size=6000):
    import torch
    from safetensors.torch import load_file
    from tokenizers import Tokenizer, models
    from training.model import ModelConfig, make_model
    from transformers import AutoConfig, AutoModelForCausalLM
    torch.set_num_threads(2)
    source=Path('build/pretrained/llm-jp-150m')
    origin=json.loads((source/'source.json').read_text(encoding='utf-8'))
    for name,info in origin['files'].items():
        with (source/name).open('rb') as f:
            if hashlib.file_digest(f,'sha256').hexdigest()!=info['sha256']:
                raise ValueError('Japanese source changed')
    base=Path('build/everyday-v9-data')
    parent=json.loads((base/'manifest.json').read_text(encoding='utf-8'))
    vocab_bytes=(base/'vocab.json').read_bytes()
    if hashlib.sha256(vocab_bytes).hexdigest()!=parent['vocab_sha256']:
        raise ValueError('parent vocabulary changed')
    vocabulary=json.loads(vocab_bytes)
    parts={}
    for file,expected in parent['split_sha256'].items():
        payload=(base/file).read_bytes()
        if hashlib.sha256(payload).hexdigest()!=expected:
            raise ValueError('parent data changed')
        parts[file.removesuffix('.jsonl')]=[json.loads(line) for line in payload.decode('utf-8').splitlines()]
    original_json=json.loads((source/'tokenizer.json').read_text(encoding='utf-8'))
    original=Tokenizer.from_file(str(source/'tokenizer.json'))
    entries=original_json['model']['vocab']
    lookup={piece:i for i,(piece,_) in enumerate(entries)}
    chars=set('▁、。')
    for row in parts['train']:
        chars.update(row['text'])
    required={lookup[c] for c in chars if c in lookup}
    if len(required)>size-7:
        raise ValueError('required characters exceed vocabulary budget')
    frequency=Counter(i for r in parts['train'] for i in original.encode(r['text'],add_special_tokens=False).ids)
    def allowed(i):
        return i>=8 and re.fullmatch(r'[▁ぁ-ゖァ-ヺ一-龯々ー、。]+',entries[i][0]) is not None
    selected=set(required)
    for i,_ in frequency.most_common():
        if allowed(i) and len(selected)<size-7:
            selected.add(i)
    for i in range(8,len(entries)):
        if allowed(i) and len(selected)<size-7:
            selected.add(i)
    kept=sorted(selected)
    special_names=['BOS','EOS','PAD','EASY','NORMAL','HARD','UNK']
    specials=['<s>','</s>','<pad>','<easy>','<normal>','<hard>','<unk>']
    pruned_entries=[(s,-127.5) for s in specials]+[(entries[i][0],entries[i][1]) for i in kept]
    tokenizer=Tokenizer(models.Unigram(pruned_entries,unk_id=6,byte_fallback=False))
    tokenizer.normalizer=original.normalizer
    tokens=[dict(id=i,surface=s,kana='',pos='special') for i,s in enumerate(special_names)]
    tokens += [dict(id=i+7,surface=entries[old][0],kana='',pos='subword') for i,old in enumerate(kept)]
    result=copy.deepcopy(parent)
    result.update(description='All twelve LLM-jp-3-150m layers retained; unigram vocabulary retokenized after pruning',
                  parent_manifest_sha256=hashlib.sha256((base/'manifest.json').read_bytes()).hexdigest(),
                  splits={},split_sha256={},pretrained_source=origin,
                  source_generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    stats=Counter()
    counts=Counter()
    for split,rows in parts.items():
        accepted=[]
        for row in rows:
            ids=tokenizer.encode(row['text'],add_special_tokens=False).ids
            text=''.join(tokens[i]['surface'] for i in ids).replace('▁','')
            if 6 in ids or text!=row['text'] or len(ids)+3>64:
                stats['unrepresentable_'+split]+=1
                continue
            row=dict(row,ids=[0,row['ids'][1],*ids,1])
            accepted.append(row)
            if split=='train':
                counts.update((s['surface'],s['kana'],s['pos']) for s in row['segments'])
        parts[split]=accepted
    lexicon=copy.deepcopy(vocabulary['lexicon'])
    for word in lexicon:
        word['metadata']=dict(count=counts[(word['surface'],word['kana'],word['pos'])])
    payload=json.dumps(dict(version='0.1.0',encoding='sentencepiece_lexicon_v1',tokens=tokens,lexicon=lexicon),ensure_ascii=False,separators=(',',':')).encode()
    result['vocab_sha256']=hashlib.sha256(payload).hexdigest()
    result['retokenization_stats']=dict(stats)
    output.mkdir(parents=True,exist_ok=True)
    (output/'vocab.json').write_bytes(payload)
    tokenizer.save(str(output/'training-tokenizer.json'))
    for split,rows in parts.items():
        data=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows).encode()
        (output/(split+'.jsonl')).write_bytes(data)
        result['splits'][split]=len(rows)
        result['split_sha256'][split+'.jsonl']=hashlib.sha256(data).hexdigest()
    (output/'manifest.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    config=ModelConfig(vocab_size=len(tokens),d_model=512,n_heads=8,n_layers=12,d_ff=2048,
                       max_seq_len=64,architecture='llama',dropout=0.)
    weights=load_file(source/'model.safetensors')
    state={name.removeprefix('model.'):tensor.float() for name,tensor in weights.items()}
    mapping=[1,2,4,1,1,1,0]+kept
    state['embed_tokens.weight']=state['embed_tokens.weight'][mapping].clone()
    state['lm_head.weight']=state['lm_head.weight'][mapping].clone()
    model=make_model(config).eval()
    model.load_state_dict(state,strict=True)
    # Check retained logits against the original full-vocabulary model, before task training.
    hf_config=AutoConfig.from_pretrained(source,local_files_only=True,trust_remote_code=False)
    reference=AutoModelForCausalLM.from_config(hf_config,attn_implementation='eager').float().eval()
    reference.load_state_dict(weights,strict=True)
    max_error=0.
    with torch.no_grad():
        for ids in ([0,3],[0,4,7],[0,5,8,9,10]):
            expected=reference(torch.tensor([[mapping[i] for i in ids]])).logits[0,:,mapping]
            actual=model(torch.tensor([ids]))[0]
            error=float((expected-actual).abs().max())
            max_error=max(max_error,error)
            torch.testing.assert_close(actual,expected,rtol=2e-5,atol=3e-4)
    initialization=dict(state_dict=state,model_config=vars(config),vocab_sha256=result['vocab_sha256'],
                        source=dict(origin,mapping='Exact retained input/output embedding rows; all pretrained body layers unchanged.',
                                    retained_logit_max_error=max_error))
    torch.save(initialization,output/'initialization.pt')
    print(json.dumps(dict(splits=result['splits'],pieces=len(tokens),words=len(lexicon),
        parameter_count=model.parameter_count(),retained_logit_max_error=max_error,stats=stats)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('build/everyday-v10-data'))
    p.add_argument('--vocab-size',type=int,default=6000)
    args=p.parse_args()
    prepare(args.output,args.vocab_size)
