"""Fetch one pinned SimpleStories-JA training shard for optional data experiments.

Never executes repository code. Only the public train shard and dataset card
are read; the upstream test split is not downloaded for training.
"""
import hashlib
import json
from pathlib import Path
import urllib.request

REPO = 'SimpleStories/SimpleStories-JA'
REVISION = 'd25370316dd77ec0ee4fd9db57bfe57075cf4b90'


def get_json(url):
    with urllib.request.urlopen(url, timeout=45) as response:
        return json.load(response)


def main():
    out = Path('build/simple-stories'); out.mkdir(parents=True, exist_ok=True)
    saved = out / 'source.json'
    revision = json.loads(saved.read_text(encoding='utf-8'))['revision'] if saved.exists() else REVISION
    entries = get_json(f'https://huggingface.co/api/datasets/{REPO}/tree/{revision}/data')
    entry = next(x for x in entries if x['path'] == 'data/train-00000-of-00004.parquet')
    assert entry['size'] < 200_000_000
    path = out / 'train-00000.parquet'
    if not path.exists():
        temp = path.with_suffix('.part')
        with urllib.request.urlopen(f'https://huggingface.co/datasets/{REPO}/resolve/{revision}/{entry["path"]}', timeout=60) as source, temp.open('wb') as target:
            while chunk := source.read(1024 * 1024):
                target.write(chunk)
        if temp.stat().st_size != entry['size']:
            raise ValueError('download size mismatch')
        temp.replace(path)
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    expected = entry.get('lfs', {}).get('oid')
    if expected and expected.removeprefix('sha256:') != actual:
        raise ValueError('upstream hash mismatch')
    with urllib.request.urlopen(f'https://huggingface.co/datasets/{REPO}/raw/{revision}/README.md', timeout=45) as response:
        card = response.read()
    if b'license: mit' not in card:
        raise ValueError('unexpected dataset license metadata')
    (out / 'README.md').write_bytes(card)
    result = dict(dataset=REPO, revision=revision, upstream_path=entry['path'], sha256=actual,
        bytes=path.stat().st_size, license='MIT', authors='The SimpleStories Project',
        dataset_url=f'https://huggingface.co/datasets/{REPO}',
        paper='https://arxiv.org/abs/2504.09184', downloaded_split='train',
        note='synthetic gpt-4o-mini stories; not independently human-reviewed')
    saved.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
