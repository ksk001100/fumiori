"""Install the checked current exports without retaining obsolete model bundles."""
import hashlib
import json
from pathlib import Path


FILES = ('model.safetensors', 'vocab.json', 'model-config.json', 'manifest.json', 'parity.json')


def checked_files(source):
    files = {name: (source / name).read_bytes() for name in FILES}
    manifest = json.loads(files['manifest.json'])
    for name, field in [('model.safetensors', 'weights_sha256'), ('vocab.json', 'vocab_sha256'), ('model-config.json', 'config_sha256')]:
        if hashlib.sha256(files[name]).hexdigest() != manifest[field]:
            raise ValueError(f'{source}: checksum mismatch: {name}')
    if not json.loads(files['parity.json']):
        raise ValueError('empty parity fixtures')
    return files, manifest


def activate(build='build', fp32_source='stories-v2-diverse-3mb', int8_source='stories-v2-diverse-3mb-int8'):
    build = Path(build)
    fp32, fm = checked_files(build / fp32_source)
    int8, qm = checked_files(build / int8_source)
    if (fm['dtype'] != 'F32' or qm['dtype'] != 'I8_ROW_F32'
            or fm['vocab_sha256'] != qm['vocab_sha256'] or fm['config_sha256'] != qm['config_sha256']
            or fm.get('generation_constraints') != qm.get('generation_constraints')
            or any(fm.get(key) != qm.get(key) for key in
                ('generation_mode', 'allow_static_fallback', 'generation_defaults', 'lexical_boundary_guard', 'polite_boundary_guard', 'reading_normalization', 'subword_start_temperature', 'subword_difficulty_prefix'))
            or qm['quantization']['source_weights_sha256'] != fm['weights_sha256']):
        raise ValueError('FP32 and INT8 exports do not describe the same model')
    updates = [('model', fp32, fm), ('model-int8', int8, qm)]
    # Validate all destinations before touching either public path.
    for name, files, _ in updates:
        target = build / name
        if target.is_symlink() or any((target / file).is_symlink() for file in files):
            raise ValueError('model destination must not contain symlinks')
        if target.exists() and not target.is_dir():
            raise ValueError('model destination must be a directory')
    backups = []
    for name, files, manifest in updates:
        target = build / name
        changed = not target.exists() or any(not (target / n).is_file() or (target / n).read_bytes() != b for n, b in files.items())
        if not changed:
            continue
        target.mkdir(parents=True, exist_ok=True)
        for file, payload in files.items():
            (target / file).write_bytes(payload)
    result = dict(model=str(build / 'model'), model_int8=str(build / 'model-int8'),
                  fp32_source=fp32_source, int8_source=int8_source,
                  weights_sha256=fm['weights_sha256'], int8_weights_sha256=qm['weights_sha256'],
                  backups=backups, training_data=fm.get('training_config', {}).get('data_dir'),
                  quality='experimental; independent human review pending')
    (build / 'current-model.json').write_bytes(json.dumps(result, ensure_ascii=False, indent=2).encode())
    return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', choices=['stories-v1-3mb', 'stories-v2-diverse-3mb'], default='stories-v2-diverse-3mb')
    args = parser.parse_args()
    source = args.source
    print(json.dumps(activate(fp32_source=source, int8_source=source + '-int8'), ensure_ascii=True))
