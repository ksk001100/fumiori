"""Export a new paired-token experiment; preserve existing exports and bundles."""
import argparse
import json
from pathlib import Path
import shutil


def export_runtime(checkpoint, prefix, output):
    import torch
    from training.export import export
    from training.quantize import quantize
    from training.package import package
    prefix, output = Path(prefix), Path(output)
    fp, qt, bundle = [prefix.with_name(prefix.name+suffix) for suffix in ('-fp32','-int8','-runtime')]
    if any(p.exists() for p in (fp,qt,bundle,output)):
        raise ValueError('experiment exports and runtime output must be new paths')
    torch.set_num_threads(2)
    export(checkpoint,fp,'autoregressive',True,True,True)
    path=fp/'manifest.json'
    manifest=json.loads(path.read_text(encoding='utf-8'))
    manifest['generation_defaults']=dict(temperature=.8,top_p=.95,history_size=16)
    path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    quantize(fp,qt)
    result=package(qt,bundle,True)
    output.mkdir(parents=True)
    for path in bundle.iterdir():
        shutil.copyfile(path,output/path.name)
    return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--prefix',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    print(json.dumps(export_runtime(args.checkpoint,args.prefix,args.output)))
