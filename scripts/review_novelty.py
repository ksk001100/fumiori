"""Offline joint quality/novelty audit of bound review sheets, never an inference filter."""
import argparse
import csv
import json
from pathlib import Path

from scripts.novelty_evaluate import sha, write_json
from scripts.novelty_metrics import normalized
from scripts.review_candidates import summarize as validate_reviews


def summarize(report, mapping, ratings):
    validate_reviews(report,mapping,ratings)
    labels={row['blind_id']:row for row in ratings}
    by_key={row['key']:labels[str(row['blind_id'])] for row in mapping}
    results={}
    for condition in report['results']:
        name=condition['name']
        classifications={normalized(row['text']):(row['exact'],row['near']) for row in condition['metrics']['details']}
        selected=condition['review_selection']
        groups={}
        for kind in ('agent','independent_human'):
            rows=[]
            for sample in selected:
                label=by_key[f"{name}:{sample['sample_id']}"]
                if not label.get('natural') or label.get('reviewer_kind')!=kind:
                    continue
                exact,near=classifications[normalized(sample['text'])]
                rows.append(dict(exact=exact,near=near,natural=label['natural']=='1',reading=label['reading_ok']=='1'))
            def breakdown(values):
                return dict(reviewed=len(values),natural=sum(v['natural'] for v in values),
                    reading=sum(v['reading'] for v in values),joint=sum(v['natural'] and v['reading'] for v in values))
            joint_novel=sum(v['natural'] and v['reading'] and not v['exact'] for v in rows)
            joint_not_near=sum(v['natural'] and v['reading'] and not v['near'] for v in rows)
            groups[kind]=dict(rated=len(rows),train_exact=breakdown([v for v in rows if v['exact']]),
                train_nonexact=breakdown([v for v in rows if not v['exact']]),
                joint_nonexact_count=joint_novel,joint_nonexact_rate_among_rated=joint_novel/len(rows) if rows else None,
                joint_not_near_count=joint_not_near,joint_not_near_rate_among_rated=joint_not_near/len(rows) if rows else None)
        results[name]=dict(selected=len(selected),by_reviewer_kind=groups)
    return dict(scope='NFKC/terminal-period training membership and character-bigram Dice only, not semantic novelty. '
                      'Joint means explicit naturalness AND reading labels. Rates among rated successes; incomplete reviews are not gates.',
                by_condition=results)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('report','mapping','ratings','output'):
        parser.add_argument('--'+name,required=True)
    args=parser.parse_args()
    if Path(args.output).exists():
        raise ValueError('joint review output must be new')
    report=json.loads(Path(args.report).read_text(encoding='utf-8'))
    mapping=json.loads(Path(args.mapping).read_text(encoding='utf-8'))
    with Path(args.ratings).open(encoding='utf-8-sig',newline='') as file:
        result=summarize(report,mapping,list(csv.DictReader(file)))
    write_json(args.output,dict(report_sha256=sha(args.report),mapping_sha256=sha(args.mapping),ratings_sha256=sha(args.ratings),
                               script_sha256=sha(__file__),**result))
