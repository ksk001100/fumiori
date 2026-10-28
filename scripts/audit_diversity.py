"""Audit a JSONL corpus or a generation report without changing its samples."""
import argparse
import hashlib
import json
from pathlib import Path

from scripts.diversity import diversity_report


def audit_file(path):
    payload = path.read_bytes()
    if path.suffix == '.jsonl':
        rows = [json.loads(line) for line in payload.decode('utf-8-sig').splitlines() if line.strip()]
    else:
        report = json.loads(payload)
        rows = report['samples'] if isinstance(report, dict) else report
    return dict(path=path.as_posix(), sha256=hashlib.sha256(payload).hexdigest(), **diversity_report(rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs', nargs='+', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    reports = [audit_file(path) for path in args.inputs]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(reports, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    for report in reports:
        print(json.dumps(dict(path=report['path'], sentences=report['overall']['sentences'],
            noun_top10_share=report['overall']['nouns']['top10_share'],
            predicate_top10_share=report['overall']['predicates']['top10_share']), ensure_ascii=True))


if __name__ == '__main__':
    main()
