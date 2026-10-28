"""Reproduce the <=3 MB autoregressive model; never manufacture missing reviews."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--skip-training',action='store_true')
    parser.add_argument('--activate',action='store_true')
    parser.add_argument('--model-name', choices=['stories-v1-3mb', 'stories-v2-diverse-3mb'], default='stories-v2-diverse-3mb')
    args=parser.parse_args()
    name = args.model_name
    config_path = Path('configs') / (name + '.json')
    config = json.loads(config_path.read_text(encoding='utf-8'))
    data_dir = config['data_dir']
    temperature = '.8' if name == 'stories-v2-diverse-3mb' else '.75'
    if not Path('build/simple-stories/source.json').exists() and not args.skip_training:
        raise RuntimeError('fetch the pinned train shard first: python scripts/fetch_stories.py')
    env=dict(os.environ,PYTHONUTF8='1',PYTHONIOENCODING='utf-8',CARGO_INCREMENTAL='0')
    commands=[['cargo','build','--release','--locked','-p','fumiori-cli','-p','fumiori-data']]
    if not args.skip_training:
        commands += [[sys.executable,'-m','scripts.prepare_stories','--out-dir',data_dir,'--train-limit','100000'],
            [sys.executable,'-m','training.train','--config',str(config_path)]]
    commands += [[sys.executable,'-m','training.export','--checkpoint',config['checkpoint'],
        '--output','build/'+name,'--generation-mode','autoregressive','--polite-boundary-guard','--normalize-morph-readings'],
        [sys.executable,'-m','training.quantize','--model','build/'+name,'--output','build/'+name+'-int8']]
    cli=str(Path('target/release/fumiori-cli'+('.exe' if os.name=='nt' else '')).resolve())
    commands += [[cli,'parity','--model','build/'+variant] for variant in (name,name+'-int8')]
    evaluate=[sys.executable,'scripts/stories_evaluate.py','--model-name',name,
        '--data-dir',data_dir,'--temperature',temperature,'--top-p','.95']
    gate=[sys.executable,'scripts/stories_gate.py','--model-name',name]
    commands += [evaluate,gate+['--stage','development','--require-pass'],
        evaluate+['--confirmation'],gate+['--require-pass']+(['--activate'] if args.activate else [])]
    for command in commands:
        subprocess.run(command,env=env,check=True)


if __name__=='__main__':
    main()
