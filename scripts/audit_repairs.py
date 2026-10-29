"""Prefix diagnostics only: not a judgment of generated sentence naturalness."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from training.train import load_checkpoint


def audit(checkpoint, data):
    torch.set_num_threads(2)
    model, ck = load_checkpoint(checkpoint)
    result = dict(checkpoint_sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
                  scope='Unquantized checkpoint prefix probabilities; no generated-text quality claim',splits={})
    manifest = json.loads((Path(data)/'manifest.json').read_text(encoding='utf-8'))
    if ck['data_manifest']['vocab_sha256'] != manifest['vocab_sha256']:
        raise ValueError('vocabulary differs')
    for split in ('train','validation'):
        path=Path(data)/(split+'.jsonl')
        if hashlib.sha256(path.read_bytes()).hexdigest()!=manifest['split_sha256'][path.name]:
            raise ValueError('repair hash differs')
        records=[]
        for line in path.read_text(encoding='utf-8').splitlines():
            row=json.loads(line)
            with torch.inference_mode():
                p=model(torch.tensor([row['prefix_ids']]))[0,-1].softmax(-1)
            valid=sorted({a['ids'][len(row['prefix_ids'])] for a in row['positives']})
            records.append(dict(source_seed=row['source_seed'],bad_probability=float(p[row['rejected_token']]),
                                valid_first_word_mass=float(p[valid].sum()),valid_first_word_count=len(valid)))
        result['splits'][split]=dict(groups=len(records),bad_probability=sum(r['bad_probability'] for r in records)/len(records),
            valid_first_word_mass=sum(r['valid_first_word_mass'] for r in records)/len(records),records=records)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--data',default='build/everyday-v12-repairs-reviewed')
    p.add_argument('--output',required=True)
    a=p.parse_args()
    Path(a.output).write_text(json.dumps(audit(a.checkpoint,a.data),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
