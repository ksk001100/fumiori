"""Measure autoregressive novelty without equating reference membership to quality.

The fixed random review sample is written before ratings. Naturalness requires
an explicit review file; metrics alone can never certify language quality.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.open_generation import measure
from scripts.diversity import diversity_report
from scripts.content_audit import EVERYDAY_POLICY, narrative_cues
from training.package import package
from training.train import read_rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--confirmation',action='store_true')
    parser.add_argument('--model-name',default='stories-v2-diverse-3mb')
    parser.add_argument('--data-dir',type=Path,default=Path('build/stories-v2-data'))
    parser.add_argument('--temperature',type=float,default=.8)
    parser.add_argument('--top-p',type=float,default=.95)
    parser.add_argument('--everyday-review', action='store_true', help='Require reviewed topic coverage and low fairy-tale share')
    parser.add_argument('--history-size', type=int, default=16)
    args=parser.parse_args()
    if Path(args.model_name).name != args.model_name or args.model_name in ('.','..'):
        raise ValueError('model name must be a directory name')
    model=Path('build')/(args.model_name+'-int8')
    stage='confirmation' if args.confirmation else 'development'
    directory=Path('build')/(args.model_name+'-evaluation')/stage
    directory.mkdir(parents=True,exist_ok=True)
    data=args.data_dir
    manifest=json.loads((model/'manifest.json').read_text(encoding='utf-8'))
    if 'generation_constraints' in manifest or manifest.get('allow_static_fallback') is not False:
        raise ValueError('model must have neither sentence constraints nor static fallback')
    if manifest['training_config']['data_dir'] != str(data).replace('\\','/'):
        raise ValueError('unexpected training data')
    for name,expected in manifest['data_manifest']['split_sha256'].items():
        if hashlib.sha256((data/name).read_bytes()).hexdigest()!=expected:
            raise ValueError('training data changed')
    cli=Path('target/release/fumiori-cli'+('.exe' if os.name=='nt' else '')).resolve()
    seeds=(3432,811,1907) if args.confirmation else (42,2026,73)
    references=sum((read_rows(data/f'{s}.jsonl') for s in ('train','validation','test','family-test')),[])
    if not 0 <= args.history_size <= 1024:
        raise ValueError('history size must be in [0, 1024]')
    result=measure(cli,model,data/'train.jsonl',references,seeds,args.temperature,args.top_p,directory,skip_near_statistics=True,history_size=args.history_size)
    result['diversity'] = diversity_report(result['samples'])
    result['narrative_cues'] = narrative_cues(result['samples'])
    if args.everyday_review or manifest['training_config'].get('everyday_review', False):
        result['content_review_policy'] = dict(EVERYDAY_POLICY, **manifest['training_config'].get('content_review_policy', {}))
        if result['content_review_policy'].get('max_runtime_bytes', 3_000_000) > 3_000_000:
            result['content_review_policy'].setdefault('independent_seed_evaluation',
                dict(samples=600, seed=530000, min_unique_ratio=.99))
    sampling = dict(temperature=args.temperature, top_p=args.top_p)
    if args.history_size != 16:
        sampling['history_size'] = args.history_size
    result.update(stage=stage,seeds=seeds,data_dir=str(data),sampling=sampling,
        weights_sha256=manifest['weights_sha256'],vocab_sha256=manifest['vocab_sha256'],
        quality='not inferred from novelty; explicit review required',
        review_policy=dict(seed=9012 if args.confirmation else 9011,count=100,
            selection='uniform sample of all 600 returned requests; no filtering by novelty, support or apparent quality'),
        runtime=package(model,Path('build')/('fumiori-'+args.model_name+'-experimental'),strip_token_metadata=True))
    decoder={k:manifest.get(k) for k in ('generation_constraints','generation_mode','allow_static_fallback','lexical_boundary_guard','polite_boundary_guard','reading_normalization')}
    if 'subword_start_temperature' in manifest:
        decoder['subword_start_temperature'] = manifest['subword_start_temperature']
    if 'subword_difficulty_prefix' in manifest:
        decoder['subword_difficulty_prefix'] = manifest['subword_difficulty_prefix']
    result['decoder_sha256']=hashlib.sha256(json.dumps(decoder,sort_keys=True).encode()).hexdigest()
    # Select by index before viewing any text or rating.
    indices=sorted(random.Random(result['review_policy']['seed']).sample(range(600),100))
    if len(result['samples'])!=600:
        raise ValueError('generation failures: cannot form the prespecified review sample')
    review=[dict(sample_id=i,text=result['samples'][i]['text'],kana=result['samples'][i]['kana']) for i in indices]
    (directory/'review-sample.json').write_text(json.dumps(review,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    with (directory/'review.csv').open('w',encoding='utf-8-sig',newline='') as file:
        fields = ['sample_id','text','kana','natural','reading_ok','reviewer','notes']
        if 'content_review_policy' in result:
            fields += ['fairy_tale', 'topic']
        writer=csv.DictWriter(file,fieldnames=fields)
        writer.writeheader(); writer.writerows(review)
    result['review_sample_ids']=indices
    min_novel = result.get('content_review_policy', {}).get('min_novel_rate', .5)
    min_unique = result.get('content_review_policy', {}).get('min_unique_ratio', .9)
    max_bytes = result.get('content_review_policy', {}).get('max_runtime_bytes', 3_000_000)
    result['structural_gate']=dict(passed=(result['metrics']['sources']=={'model':600}
        and result['metrics']['novel_core_rate']>=min_novel and result['metrics']['unique_ratio']>=min_unique
        and result['runtime']['runtime_bytes']<=max_bytes),
        criteria=f'600 model outputs, >={min_novel:.0%} novel normalized bodies, >={min_unique:.0%} distinct texts, <={max_bytes:,} runtime bytes')
    report=Path(f'docs/{args.model_name}-{stage}-measurements.json')
    report.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(metrics=result['metrics'],runtime=result['runtime'],structural_gate=result['structural_gate']),ensure_ascii=False))


if __name__=='__main__':
    main()
