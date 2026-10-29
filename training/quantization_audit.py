"""Compare task validation loss before/after INT4; this is not a language-quality review."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import torch
from torch.nn import functional as F
from safetensors.torch import load_file
from .model import ModelConfig, make_model
from .train import read_rows,tensor_batch


def restored_state(directory):
    manifest=json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
    for name,key in [('model.safetensors','weights_sha256'),('model-config.json','config_sha256'),('vocab.json','vocab_sha256')]:
        if hashlib.sha256((directory/name).read_bytes()).hexdigest()!=manifest[key]:
            raise ValueError('quantization audit source changed')
    weights=load_file(directory/'model.safetensors')
    if manifest['dtype']=='I4_BLOCK128_F32':
        decoded={k:v for k,v in weights.items() if not k.endswith(('.shape','.scale')) and v.dtype==torch.float32}
        for name in [k[:-6] for k in weights if k.endswith('.shape')]:
            packed=weights[name].to(torch.int16)
            values=torch.stack((packed&15,packed>>4),dim=1).flatten().float()-8
            shape=weights[name+'.shape'].tolist()
            decoded[name]=(values.reshape(-1,128)*weights[name+'.scale'][:,None]).reshape(shape)
        return decoded
    if manifest['dtype']!='F32':
        raise ValueError('audit supports FP32 and block INT4')
    return weights


def audit(fp32,int4,data,output):
    torch.set_num_threads(2)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for this diagnostic')
    rows=read_rows(data/'validation.jsonl')
    result={}
    for name,directory in [('fp32',fp32),('int4',int4)]:
        config=ModelConfig(**json.loads((directory/'model-config.json').read_text()))
        model=make_model(config)
        model.load_state_dict(restored_state(directory))
        model=model.to('cuda').eval()
        groups=defaultdict(lambda:[0.,0])
        with torch.no_grad():
            for start in range(0,len(rows),64):
                part=rows[start:start+64]
                batch=tensor_batch(part,'cuda')
                logits=model(batch[:,:-1])
                losses=F.cross_entropy(logits.flatten(0,1),batch[:,1:].flatten(),ignore_index=2,reduction='none').reshape(batch.shape[0],-1)
                for row,loss,n in zip(part,losses.sum(1).tolist(),(batch[:,1:]!=2).sum(1).tolist()):
                    for key in ('all',row['training_group'],str(row['ids'][1])):
                        groups[key][0]+=loss;groups[key][1]+=n
        result[name]={key:dict(nll=loss/n,tokens=n) for key,(loss,n) in groups.items()}
        del model
        torch.cuda.empty_cache()
    result['delta_nll']={key:result['int4'][key]['nll']-value['nll'] for key,value in result['fp32'].items()}
    result['validation_sha256']=hashlib.sha256((data/'validation.jsonl').read_bytes()).hexdigest()
    result['interpretation']='Same validation examples and tokenization; lower loss does not establish naturalness or reading accuracy.'
    output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fp32',type=Path,required=True)
    p.add_argument('--int4',type=Path,required=True)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();audit(a.fp32,a.int4,a.data,a.output)
