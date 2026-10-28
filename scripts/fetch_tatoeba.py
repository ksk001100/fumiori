"""Fetch checksummed public Japanese Tatoeba exports; never execute remote code."""
import argparse
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import urllib.request


BASE = 'https://downloads.tatoeba.org/exports/per_language/jpn/'


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.names = []

    def handle_starttag(self, tag, attrs):
        href = dict(attrs).get('href', '')
        if tag == 'a' and href.startswith('jpn_') and '/' not in href:
            self.names.append(href)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--list', action='store_true')
    parser.add_argument('--out-dir', type=Path, default=Path('build/tatoeba-source'))
    args = parser.parse_args()
    listing = Links()
    with urllib.request.urlopen(BASE, timeout=45) as response:
        listing.feed(response.read().decode('utf-8'))
    if args.list:
        print('\n'.join(listing.names))
        return
    names = ('jpn_sentences_detailed.tsv.bz2', 'jpn_transcriptions.tsv.bz2')
    if any(name not in listing.names for name in names):
        raise ValueError(f'expected official exports missing: {listing.names}')
    args.out_dir.mkdir(parents=True, exist_ok=True)
    saved = args.out_dir / 'source.json'
    old = json.loads(saved.read_text(encoding='utf-8')) if saved.exists() else None
    files = {}
    for name in names:
        path = args.out_dir / name
        if not path.exists():
            temporary = path.with_suffix('.part')
            with urllib.request.urlopen(BASE + name, timeout=60) as response, temporary.open('wb') as target:
                total = 0
                while chunk := response.read(1024 * 1024):
                    total += len(chunk)
                    if total > 200_000_000:
                        raise ValueError('public export exceeds download limit')
                    target.write(chunk)
            temporary.replace(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if old and digest != old['files'][name]['sha256']:
            raise ValueError(f'saved source changed: {name}')
        files[name] = dict(url=BASE+name, bytes=path.stat().st_size, sha256=digest)
    source = dict(dataset='Tatoeba Japanese', dataset_url='https://tatoeba.org/en/downloads',
                  license='CC-BY-2.0-FR', attribution='Tatoeba.org and its contributors',
                  license_url='https://creativecommons.org/licenses/by/2.0/fr/', files=files,
                  note='Public weekly export; contributor names and sentence IDs retained in prepared data. '
                       'Transcriptions without a username are not marked reviewed by Tatoeba.')
    saved.write_text(json.dumps(source, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(source, ensure_ascii=True), flush=True)


if __name__ == '__main__':
    main()
