"""Build a minimal, checksummed runtime bundle, retaining provenance by hash.

Full training provenance and parity fixtures stay in the source export. The
application needs four runtime files and LICENSE, never a Python checkpoint.
"""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def compact_json(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':')).encode('utf-8')


def package(source, output, strip_token_metadata=False):
    source, output = Path(source), Path(output)
    if source.resolve() == output.resolve() or source.resolve() in output.resolve().parents:
        raise ValueError('runtime output must be outside source export')
    manifest_bytes = (source / 'manifest.json').read_bytes()
    manifest = json.loads(manifest_bytes)
    names = {'vocab.json': 'vocab_sha256', 'model-config.json': 'config_sha256',
             'model.safetensors': 'weights_sha256'}
    files = {}
    for name, key in names.items():
        payload = (source / name).read_bytes()
        if digest(payload) != manifest[key]:
            raise ValueError(f'checksum mismatch: {name}')
        files[name] = compact_json(json.loads(payload)) if name.endswith('.json') else payload
    minimal = {key: manifest[key] for key in ('version', 'dtype', 'parameter_count')}
    if strip_token_metadata:
        vocab = json.loads(files['vocab.json'])
        for token in vocab['tokens']:
            token.pop('metadata', None)
        files['vocab.json'] = compact_json(vocab)
    minimal.update({key: digest(files[name]) for name, key in names.items()})
    if 'data_sources' in manifest:
        minimal['data_sources'] = manifest['data_sources']
    if 'quantization' in manifest:
        minimal['quantization'] = manifest['quantization']
    if 'generation_constraints' in manifest:
        minimal['generation_constraints'] = manifest['generation_constraints']
    for key in ('generation_mode', 'allow_static_fallback', 'generation_defaults', 'lexical_boundary_guard', 'polite_boundary_guard', 'reading_normalization'):
        if key in manifest:
            minimal[key] = manifest[key]
    minimal['source_manifest_sha256'] = digest(manifest_bytes)
    minimal['quality'] = 'experimental; independent human review pending'
    files['manifest.json'] = compact_json(minimal)
    files['LICENSE'] = (Path(__file__).resolve().parents[1] / 'LICENSE').read_bytes()
    if output.exists() and any(output.iterdir()):
        # Avoid bundling stale checkpoints or untracked extras accidentally.
        if set(p.name for p in output.iterdir()) != set(files):
            raise ValueError('runtime output must be empty or contain only runtime files')
    output.mkdir(parents=True, exist_ok=True)
    for name, payload in files.items():
        (output / name).write_bytes(payload)
    archive = output.with_suffix('.zip')
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as handle:
        for name, payload in sorted(files.items()):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            handle.writestr(info, payload)
    return dict(files_bytes={name: len(payload) for name, payload in files.items()},
                runtime_bytes=sum(map(len, files.values())), zip_bytes=archive.stat().st_size,
                zip_sha256=digest(archive.read_bytes()), source_manifest_sha256=digest(manifest_bytes))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--strip-token-metadata', action='store_true')
    args = parser.parse_args()
    print(json.dumps(package(args.model, args.output, args.strip_token_metadata)))
