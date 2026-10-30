"""Small FP32 prefix-conditioned probe, separate from CPU/WASM free generation."""
import argparse
import json
from pathlib import Path

from scripts.novelty_evaluate import sha, write_json


def probe(audit_path, checkpoints):
    import torch
    from training.train import load_checkpoint
    audit=json.loads(Path(audit_path).read_text(encoding='utf-8'))
    cases={c['id']:c for c in audit['cases']}
    torch.set_num_threads(2)
    results=[]
    for checkpoint in checkpoints:
        model,ck=load_checkpoint(checkpoint)
        if ck['data_manifest']['vocab_sha256']!=audit['vocab_sha256']:
            raise ValueError('probe vocabulary mismatch')
        tokens=ck['vocab']['tokens']
        rows=[]
        with torch.inference_mode():
            for case in audit['encoded']:
                prefix=cases[case['id']]['prefix']
                ids=case['ids'][0][:2]
                text=''
                for identity in case['ids'][0][2:-1]:
                    text+=tokens[identity]['surface']
                    ids.append(identity)
                    if len(text)>=len(prefix):
                        break
                if text!=prefix:
                    rows.append(dict(id=case['id'],prefix=prefix,error='prefix not on a token boundary'))
                    continue
                terminated=False
                error=None
                while len(ids)<model.config.max_seq_len:
                    next_id=int(model(torch.tensor([ids]))[0,-1].argmax())
                    if next_id==1:
                        terminated=True
                        break
                    if next_id<7:
                        error='unexpected special token'
                        break
                    ids.append(next_id)
                segments=[tokens[i] for i in ids[2:]]
                text=''.join(t['surface'] for t in segments).removesuffix('。')
                kana=''.join('、' if t['surface']=='、' else t['kana'] for t in segments)
                rows.append(dict(id=case['id'],prefix=prefix,text=text,kana=kana,ids=ids,
                    terminated=terminated,error=error,positive=cases[case['id']]['positive'],negative=cases[case['id']]['negative']))
        results.append(dict(checkpoint=checkpoint,checkpoint_sha256=sha(checkpoint),samples=rows))
    return dict(audit_sha256=sha(audit_path),script_sha256=sha(__file__),
        scope='Greedy FP32 PyTorch completion conditioned only on noun+case prefix and fixed positive-derived difficulty. No slot template, no target-verb forcing. Not a WASM API, not full subject/object/location/tense control; no automatic semantic or reading certification.',
        results=results)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit',required=True)
    parser.add_argument('--checkpoint',action='append',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    if Path(args.output).exists():
        raise ValueError('probe output must be new')
    write_json(args.output,probe(args.audit,args.checkpoint))
