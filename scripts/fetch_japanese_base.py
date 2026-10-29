"""Pinned LLM-jp-3-150m source tensors, without remote Python code."""
import hashlib
import json
from pathlib import Path
from scripts.fetch_teacher import download, FILES


def main():
    repo, revision = 'llm-jp/llm-jp-3-150m', 'b112feef602fff752e4dac4c30af6a2c2fa41c7a'
    expected = '257d0c9e56fe2b585cc13a0c9550234b30c00eca021ce23bda37989db95b7fb2'
    out = Path('build/pretrained/llm-jp-150m')
    out.mkdir(parents=True,exist_ok=True)
    record = out/'source.json'
    previous = json.loads(record.read_text(encoding='utf-8')) if record.exists() else None
    files = {}
    for name in FILES:
        path = out/name
        url = f'https://huggingface.co/{repo}/resolve/{revision}/{name}'
        if not path.exists():
            download(url,path)
        with path.open('rb') as f:
            sha = hashlib.file_digest(f,'sha256').hexdigest()
        if name=='model.safetensors' and sha!=expected or previous and sha!=previous['files'][name]['sha256']:
            raise ValueError(f'Japanese base checksum mismatch: {name}')
        files[name] = dict(bytes=path.stat().st_size,sha256=sha,url=url)
        print(json.dumps(dict(file=name,bytes=path.stat().st_size)),flush=True)
    record.write_text(json.dumps(dict(model=repo,revision=revision,license='Apache-2.0',authors='LLM-jp',
        url='https://huggingface.co/'+repo,files=files),indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':
    main()
