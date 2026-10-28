"""Fetch an immutable public tensor model and tokenizer, never repository code."""
import hashlib
import json
from pathlib import Path
import urllib.request


def main():
    repo = 'rinna/japanese-gpt2-xsmall'
    out = Path('build/pretrained/rinna-xsmall')
    out.mkdir(parents=True, exist_ok=True)
    metadata = out / 'source.json'
    with urllib.request.urlopen('https://huggingface.co/api/models/'+repo, timeout=45) as response:
        info = json.load(response)
    revision = json.loads(metadata.read_text())['revision'] if metadata.exists() else info['sha']
    files = {}
    for name in ('config.json', 'spiece.model', 'model.safetensors', 'README.md', 'tokenizer_config.json', 'special_tokens_map.json'):
        path = out/name
        url = f'https://huggingface.co/{repo}/resolve/{revision}/{name}'
        if not path.exists():
            temporary = path.with_suffix('.part')
            with urllib.request.urlopen(url, timeout=90) as response, temporary.open('wb') as target:
                total = 0
                while chunk := response.read(1024*1024):
                    total += len(chunk)
                    if total > 250_000_000:
                        raise ValueError('pretrained file exceeds limit')
                    target.write(chunk)
            temporary.replace(path)
        files[name] = dict(url=url, sha256=hashlib.sha256(path.read_bytes()).hexdigest(), bytes=path.stat().st_size)
    if metadata.exists():
        old = json.loads(metadata.read_text())
        if any(old['files'][n]['sha256'] != files[n]['sha256'] for n in files):
            raise ValueError('saved pretrained input changed')
    source = dict(model=repo, revision=revision, license='MIT', authors='rinna Co., Ltd.',
                  url='https://huggingface.co/'+repo, files=files)
    metadata.write_text(json.dumps(source, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(source), flush=True)


if __name__ == '__main__':
    main()
