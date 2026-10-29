"""Fetch a pinned Japanese teacher for offline training; never fetch Python code."""
import hashlib
import json
from pathlib import Path
import urllib.request
import time

REPOSITORY = 'llm-jp/llm-jp-3.1-1.8b-instruct4'
REVISION = 'f19510db409090bb1737f24f868d17c4bdc86c8e'
WEIGHTS_SHA256 = 'a5608664cb63ed2ee32c26ecdcf88d12f7ed8b45230761b96b857aa64ef65ed3'
FILES = ('config.json', 'generation_config.json', 'model.safetensors', 'tokenizer.json',
         'tokenizer_config.json', 'special_tokens_map.json', 'README.md')


def download(url, path):
    temporary = path.with_suffix('.part')
    for attempt in range(5):
        offset = temporary.stat().st_size if temporary.exists() else 0
        request = urllib.request.Request(url, headers={'Range': f'bytes={offset}-'}) if offset else url
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                resumed = response.status == 206
                if resumed and not response.headers.get('Content-Range', '').startswith(f'bytes {offset}-'):
                    raise ValueError('unexpected resumed byte range')
                expected_total = (int(response.headers['Content-Range'].rsplit('/',1)[1]) if resumed
                                  else int(response.headers.get('Content-Length','0')))
                total = offset if resumed else 0
                last_report = total
                with temporary.open('ab' if resumed else 'wb') as target:
                    while chunk := response.read(1024 * 1024):
                        target.write(chunk)
                        total += len(chunk)
                        if total > 4_000_000_000:
                            raise ValueError('teacher file exceeds allowed download size')
                        if total-last_report >= 256*1024*1024:
                            print(json.dumps(dict(file=path.name, downloaded_bytes=total)),flush=True)
                            last_report = total
            if expected_total and total != expected_total:
                print(json.dumps(dict(file=path.name,partial_bytes=total,expected_bytes=expected_total)),flush=True)
                continue
            temporary.replace(path)
            return
        except OSError:
            if attempt == 4:
                raise
            print(json.dumps(dict(file=path.name,retry=attempt+1)),flush=True)
            time.sleep(2)
    raise OSError('teacher download remained incomplete after retries')


def main():
    out = Path('build/teacher/llm-jp-3.1-1.8b-instruct4')
    out.mkdir(parents=True, exist_ok=True)
    record = out / 'source.json'
    previous = json.loads(record.read_text()) if record.exists() else None
    entries = {}
    for name in FILES:
        path = out / name
        url = f'https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/{name}'
        if name == 'model.safetensors' and path.exists() and path.stat().st_size < 3_735_253_776:
            path.replace(path.with_suffix('.part'))
        if not path.exists():
            download(url, path)
        with path.open('rb') as file:
            sha = hashlib.file_digest(file, 'sha256').hexdigest()
        if (name == 'model.safetensors' and sha != WEIGHTS_SHA256
                or previous is not None and sha != previous['files'][name]['sha256']):
            raise ValueError(f'teacher checksum mismatch: {name}')
        entries[name] = dict(bytes=path.stat().st_size, sha256=sha, url=url)
        print(json.dumps(dict(file=name, **entries[name])), flush=True)
    record.write_text(json.dumps(dict(model=REPOSITORY, revision=REVISION, license='Apache-2.0',
        authors='LLM-jp', url='https://huggingface.co/' + REPOSITORY,
        role='offline training teacher only; never packaged for application inference',
        files=entries), indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
