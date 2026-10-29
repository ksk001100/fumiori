"""Wrap a local task checkpoint for same-vocabulary compact fine-tuning."""
import argparse
import hashlib
import json
from pathlib import Path
import torch


def initialize(checkpoint, data_dir, output):
    checkpoint, data_dir, output = Path(checkpoint), Path(data_dir), Path(output)
    if output.exists():
        raise ValueError('initialization output already exists')
    ck = torch.load(checkpoint, map_location='cpu', weights_only=True)
    vocab_hash = hashlib.sha256((data_dir/'vocab.json').read_bytes()).hexdigest()
    if ck['data_manifest']['vocab_sha256'] != vocab_hash:
        raise ValueError('checkpoint vocabulary differs from prepared data')
    source = dict(model='fumiori/' + checkpoint.stem,
                  revision=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                  license='MIT', authors='Fumiori contributors',
                  url='https://github.com/ksk001100/fumiori')
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(state_dict=ck['state_dict'], vocab_sha256=vocab_hash, source=source), output)
    return source


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--data-dir', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    print(json.dumps(initialize(a.checkpoint, a.data_dir, a.output)))
