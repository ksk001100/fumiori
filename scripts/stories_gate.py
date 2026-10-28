"""Require both prespecified review samples and byte/novelty checks to promote."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.generalization import evaluation_text
from scripts.semantic_keys import core_key
from training.train import read_rows
from training.package import package
from training.activate import activate


def review_gate(report, review, sample_bytes, training):
    if review['weights_sha256'] != report['weights_sha256'] or review['sample_sha256'] != hashlib.sha256(sample_bytes).hexdigest():
        raise ValueError('review does not match generated samples')
    if review.get('decoder_sha256')!=report.get('decoder_sha256'):
        raise ValueError('review does not match decoding policy')
    ratings=review['ratings']
    expected=report['review_sample_ids']
    supplied=json.loads(sample_bytes)
    actual=[dict(sample_id=i,text=report['samples'][i]['text'],kana=report['samples'][i]['kana']) for i in expected]
    if supplied!=actual:
        raise ValueError('sample file differs from measured outputs')
    if len(ratings)!=100 or sorted(r['sample_id'] for r in ratings)!=expected:
        raise ValueError('review must rate every prespecified sample exactly once')
    if any(type(r.get('natural')) is not bool or type(r.get('reading_ok')) is not bool for r in ratings):
        raise ValueError('ratings must be explicit booleans')
    cores={core_key(evaluation_text(r['text'])) for r in training}
    novel=[r for r in ratings if core_key(evaluation_text(report['samples'][r['sample_id']]['text'])) not in cores]
    correct=lambda r:r['natural'] and r['reading_ok']
    accepted=sum(correct(r) for r in ratings)
    novel_accepted=sum(correct(r) for r in novel)
    return dict(passed=report['structural_gate']['passed'] and accepted>=90 and len(novel)>=30 and novel_accepted/len(novel)>=.85,
        reviewed=100,acceptable=accepted,novel_reviewed=len(novel),novel_acceptable=novel_accepted,
        reviewer=review['reviewer'],reviewer_type=review['reviewer_type'],
        independent_human_review='pending',
        policy='>=90/100 acceptable overall; >=30 novel reviewed; >=85% acceptable among novel; structural gate also required')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-name',default='stories-v2-diverse-3mb')
    parser.add_argument('--activate',action='store_true')
    parser.add_argument('--stage',choices=['development','both'],default='both')
    parser.add_argument('--require-pass',action='store_true')
    args=parser.parse_args()
    name=args.model_name
    if Path(name).name!=name or name in ('.','..'):
        raise ValueError('invalid model name')
    if args.activate and args.stage!='both':
        raise ValueError('activation requires confirmation review')
    result={}
    stages=('development',) if args.stage=='development' else ('development','confirmation')
    for stage in stages:
        report=json.loads(Path(f'docs/{name}-{stage}-measurements.json').read_text(encoding='utf-8'))
        review=json.loads(Path(f'docs/{name}-{stage}-review.json').read_text(encoding='utf-8'))
        sample=(Path('build')/(name+'-evaluation')/stage/'review-sample.json').read_bytes()
        result[stage]=review_gate(report,review,sample,read_rows(Path(report['data_dir'])/'train.jsonl'))
        if stage=='development':
            previous=report
        elif any(report[k]!=previous[k] for k in ('weights_sha256','vocab_sha256','sampling','decoder_sha256')):
            raise ValueError('confirmation used a different model or sampling')
        elif report.get('runtime_identity') != previous.get('runtime_identity'):
            raise ValueError('confirmation used a different runtime')
    result['passed']=all(result[s]['passed'] for s in stages)
    if args.activate and result['passed']:
        for model in (name,name+'-int8'):
            path=Path('build')/model/'manifest.json'
            manifest=json.loads(path.read_text(encoding='utf-8'))
            if model.endswith('-int8') and manifest['weights_sha256']!=report['weights_sha256']:
                raise ValueError('exports changed since review')
            decoder={k:manifest.get(k) for k in ('generation_constraints','generation_mode','allow_static_fallback','lexical_boundary_guard','polite_boundary_guard','reading_normalization')}
            if hashlib.sha256(json.dumps(decoder,sort_keys=True).encode()).hexdigest()!=report['decoder_sha256'] or manifest['vocab_sha256']!=report['vocab_sha256']:
                raise ValueError('export vocabulary/decoding policy changed since review')
            manifest['generation_defaults']=report['sampling']
            path.write_text(json.dumps(manifest,ensure_ascii=False,separators=(',',':'))+'\n',encoding='utf-8')
        result['package']=package(Path('build')/(name+'-int8'),Path('build')/('fumiori-'+name),strip_token_metadata=True)
        if result['package']['runtime_bytes']>3_000_000:
            raise ValueError('final runtime exceeds 3,000,000 bytes')
        result['activation']=activate(fp32_source=name,int8_source=name+'-int8')
    suffix='development-gate' if args.stage=='development' else 'promotion'
    Path(f'docs/{name}-{suffix}.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))
    if args.activate and 'activation' not in result:
        raise RuntimeError('review/novelty/size gates rejected the candidate; defaults unchanged')
    if args.require_pass and not result['passed']:
        raise RuntimeError('prespecified review gate failed')


if __name__=='__main__':
    main()
