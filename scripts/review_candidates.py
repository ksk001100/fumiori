"""Bind a shuffled rating sheet back to matched candidate experiments."""
import argparse
import csv
import json
from pathlib import Path

from scripts.novelty_evaluate import sha, summarize_review, write_json


def summarize(report, mapping, ratings):
    targets={f"{r['name']}:{row['sample_id']}":row for r in report['results'] for row in r['review_selection']}
    by_id={str(r['blind_id']):r['key'] for r in mapping}
    if len(by_id)!=len(mapping) or set(by_id.values())!=set(targets) or len(mapping)!=len(targets):
        raise ValueError('blind mapping must cover each selected sample once')
    seen, restored=set(),{}
    for row in ratings:
        identity=row['blind_id']
        if identity in seen or identity not in by_id:
            raise ValueError('duplicate or unexpected blind ID')
        seen.add(identity)
        key=by_id[identity]
        target=targets[key]
        if any(row[k]!=target[k] for k in ('text','kana','status')):
            raise ValueError('blind sample text, kana or status modified')
        restored[key]=dict(row,sample_id=str(target['sample_id']),sample_sha256=target['sample_sha256'])
    if seen!=by_id.keys():
        raise ValueError('blind ratings must retain all samples')
    return {r['name']:summarize_review(r['review_selection'],
                [restored[f"{r['name']}:{row['sample_id']}"] for row in r['review_selection']]) for r in report['results']}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',required=True)
    parser.add_argument('--mapping',required=True)
    parser.add_argument('--ratings',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    if Path(args.output).exists():
        raise ValueError('review output must be new')
    report=json.loads(Path(args.report).read_text(encoding='utf-8'))
    mapping=json.loads(Path(args.mapping).read_text(encoding='utf-8'))
    with Path(args.ratings).open(encoding='utf-8-sig',newline='') as file:
        result=summarize(report,mapping,list(csv.DictReader(file)))
    write_json(args.output,dict(report_sha256=sha(args.report),mapping_sha256=sha(args.mapping),
                               ratings_sha256=sha(args.ratings),by_condition=result))
