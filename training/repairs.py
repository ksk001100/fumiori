"""Sparse error suppression anchored to uniformly sampled valid repairs."""
import hashlib
import json
from pathlib import Path
import torch
from torch.nn import functional as F


def repair_loss(logits, batch, prefix_lengths, rejected_ids):
    targets = batch[:,1:]
    positions = torch.arange(targets.shape[1],device=batch.device)[None,:]
    keep = (positions >= prefix_lengths[:,None]-1) & (targets != 2)
    ce = F.cross_entropy(logits.float().transpose(1,2),targets,ignore_index=2,reduction='none')
    positive = ((ce*keep).sum(1)/keep.sum(1).clamp_min(1)).mean()
    selected = logits[torch.arange(len(batch),device=batch.device),prefix_lengths-1].float()
    probability = selected.softmax(-1).gather(1,rejected_ids[:,None]).squeeze(1)
    negative = -torch.log1p(-probability.clamp(max=1-1e-6)).mean()
    return positive,negative


class RepairData:
    def __init__(self, directory, vocab_hash, parent_hash, device):
        directory = Path(directory)
        self.manifest = json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
        if self.manifest['vocab_sha256'] != vocab_hash or self.manifest['parent_manifest_sha256'] != parent_hash:
            raise ValueError('repair data provenance differs')
        for filename, expected in self.manifest['split_sha256'].items():
            if hashlib.sha256((directory/filename).read_bytes()).hexdigest() != expected:
                raise ValueError('repair split hash mismatch')
        self.groups = [json.loads(s) for s in (directory/'train.jsonl').read_text(encoding='utf-8').splitlines()]
        # Uniform group mass, then uniform alternatives within each group.
        examples,weights = [],[]
        for group in self.groups:
            prefix = group['prefix_ids']
            for positive in group['positives']:
                ids = positive['ids']
                if not 2<=len(prefix)<len(ids)<=64 or ids[:len(prefix)]!=prefix or ids[len(prefix)]==group['rejected_token']:
                    raise ValueError('invalid repair prefix or alternative')
                examples.append((ids,len(prefix),group['rejected_token']))
                weights.append(1/len(group['positives']))
        if not examples:
            raise ValueError('empty repair data')
        length = max(len(e[0]) for e in examples)
        self.batch = torch.tensor([ids+[2]*(length-len(ids)) for ids,_,_ in examples],device=device)
        self.prefixes = torch.tensor([n for _,n,_ in examples],device=device)
        self.rejected = torch.tensor([n for _,_,n in examples],device=device)
        self.weights = torch.tensor(weights,device=device)

    def loss(self,model,count):
        indices = torch.multinomial(self.weights,count,replacement=True)
        batch = self.batch[indices]
        return repair_loss(model(batch[:,:-1]),batch,self.prefixes[indices],self.rejected[indices])
