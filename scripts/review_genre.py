"""Bind explicit narrative labels to the existing blind quality review."""
import argparse
import csv
import json
from pathlib import Path

from scripts.novelty_evaluate import sha, write_json
from scripts.novelty_metrics import normalized
from scripts.review_candidates import summarize as validate_quality


def summarize(report, mapping, ratings):
    validate_quality(report, mapping, ratings)
    labels = {m['key']: next(r for r in ratings if str(r['blind_id']) == str(m['blind_id']))
              for m in mapping}
    if any(r.get('fiction_or_narrative') not in ('0', '1') or not r.get('genre_notes', '').strip()
           for r in ratings):
        raise ValueError('explicit narrative label and genre rationale required for every selected request')
    result = {}
    for condition in report['results']:
        details = {normalized(r['text']): r for r in condition['metrics']['details']}
        groups = {}
        for kind in ('agent', 'independent_human'):
            rows = [(s, labels[f"{condition['name']}:{s['sample_id']}"])
                    for s in condition['review_selection']]
            rows = [(s, r) for s, r in rows if r.get('reviewer_kind') == kind]
            acceptable = [(s, r) for s, r in rows if r.get('natural') == r.get('reading_ok') == '1']
            daily = [(s, r) for s, r in acceptable if r['fiction_or_narrative'] == '0']
            groups[kind] = dict(reviewed=len(rows),
                narrative_count=sum(r['fiction_or_narrative'] == '1' for _, r in rows),
                acceptable_non_narrative_count=len(daily),
                acceptable_non_narrative_nonexact_count=sum(not details[normalized(s['text'])]['exact'] for s, _ in daily),
                acceptable_non_narrative_not_near_count=sum(not details[normalized(s['text'])]['near'] for s, _ in daily))
        result[condition['name']] = groups
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('report', 'mapping', 'ratings', 'protocol', 'output'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    if Path(args.output).exists():
        raise ValueError('review output must be new')
    with Path(args.ratings).open(encoding='utf-8-sig', newline='') as file:
        ratings = list(csv.DictReader(file))
    report, mapping = [json.loads(Path(p).read_text(encoding='utf-8')) for p in (args.report, args.mapping)]
    write_json(args.output, dict(by_condition=summarize(report, mapping, ratings),
        **{k + '_sha256': sha(getattr(args, k)) for k in ('report', 'mapping', 'ratings', 'protocol')},
        script_sha256=sha(__file__),
        limitation='Subjective explicit review of narrative framing, not keyword classification or independent human evidence.'))
