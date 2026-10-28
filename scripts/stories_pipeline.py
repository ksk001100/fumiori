"""Reproduce the <=3 MB autoregressive model; never manufacture missing reviews."""
import argparse
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--skip-training',action='store_true')
    parser.add_argument('--activate',action='store_true')
    args=parser.parse_args()
    if not Path('build/simple-stories/source.json').exists() and not args.skip_training:
        raise RuntimeError('fetch the pinned train shard first: python scripts/fetch_stories.py')
    env=dict(os.environ,PYTHONUTF8='1',PYTHONIOENCODING='utf-8',CARGO_INCREMENTAL='0')
    commands=[['cargo','build','--release','--locked','-p','fumiori-cli','-p','fumiori-data']]
    if not args.skip_training:
        commands += [[sys.executable,'-m','scripts.prepare_stories','--out-dir','build/stories-v1-data','--train-limit','100000'],
            [sys.executable,'-m','training.train','--config','configs/stories-v1-3mb.json']]
    commands += [[sys.executable,'-m','training.export','--checkpoint','build/checkpoints/stories-v1-3mb.pt',
        '--output','build/stories-v1-3mb','--generation-mode','autoregressive','--polite-boundary-guard','--normalize-morph-readings'],
        [sys.executable,'-m','training.quantize','--model','build/stories-v1-3mb','--output','build/stories-v1-3mb-int8']]
    cli=str(Path('target/release/fumiori-cli'+('.exe' if os.name=='nt' else '')).resolve())
    commands += [[cli,'parity','--model','build/'+name] for name in ('stories-v1-3mb','stories-v1-3mb-int8')]
    evaluate=[sys.executable,'scripts/stories_evaluate.py','--model-name','stories-v1-3mb',
        '--data-dir','build/stories-v1-data','--temperature','.75','--top-p','.95']
    commands += [evaluate,[sys.executable,'scripts/stories_gate.py','--stage','development','--require-pass'],
        evaluate+['--confirmation'],[sys.executable,'scripts/stories_gate.py','--require-pass']+(['--activate'] if args.activate else [])]
    for command in commands:
        subprocess.run(command,env=env,check=True)


if __name__=='__main__':
    main()
