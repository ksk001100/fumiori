"""Build local Worker demo. Requires installed wasm target and wasm-bindgen 0.2.127."""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--models', nargs='+', choices=['model', 'model-int8'], default=['model-int8'])
    source.add_argument('--runtime-model', type=Path, help='Build INT8 demo from an existing five-file runtime bundle (no training exports required)')
    args = parser.parse_args()
    runtime_files = ('manifest.json', 'model-config.json', 'vocab.json', 'model.safetensors', 'LICENSE')
    if args.runtime_model:
        manifest = json.loads((args.runtime_model / 'manifest.json').read_text(encoding='utf-8'))
        for name, key in [('model-config.json', 'config_sha256'), ('vocab.json', 'vocab_sha256'),
                          ('model.safetensors', 'weights_sha256')]:
            if hashlib.sha256((args.runtime_model / name).read_bytes()).hexdigest() != manifest[key]:
                raise ValueError(f'checksum mismatch: {name}')
        for name in runtime_files:
            if not (args.runtime_model / name).is_file():
                raise ValueError(f'missing runtime file: {name}')
        args.models = ['model-int8']
    subprocess.run(["cargo", "build", "--locked", "--release", "-p", "fumiori-wasm",
                    "--target", "wasm32-unknown-unknown"], check=True)
    out = Path("build/web")
    shutil.copytree("web", out, dirs_exist_ok=True)
    subprocess.run(["wasm-bindgen", "target/wasm32-unknown-unknown/release/fumiori_wasm.wasm",
                    "--target", "web", "--out-dir", str(out / "pkg")], check=True)
    page = (out / 'index.html').read_text(encoding='utf-8')
    labels = {'model': 'FP32', 'model-int8': 'INT8保存'}
    start = page.index('<select id="model">') + len('<select id="model">')
    end = page.index('</select>', start)
    page = page[:start] + ''.join(f'<option value="{name}">{labels[name]}</option>' for name in args.models) + page[end:]
    (out / 'index.html').write_text(page, encoding='utf-8')
    for model in args.models:
        if args.runtime_model:
            target = out / model
            target.mkdir(exist_ok=True)
            for name in runtime_files:
                shutil.copyfile(args.runtime_model / name, target / name)
            continue
        runtime = {'model-int8': 'fumiori-stories-v2-diverse-3mb'}.get(model)
        if runtime:
            subprocess.run([sys.executable, '-m', 'training.package', '--model', f'build/{model}',
                            '--output', f'build/{runtime}', '--strip-token-metadata'], check=True)
        target = out / model
        target.mkdir(exist_ok=True)
        for name in ("manifest.json", "model-config.json", "vocab.json", "model.safetensors", "parity.json"):
            source = runtime if runtime and name != 'parity.json' else model
            shutil.copyfile(Path("build") / source / name, target / name)
        shutil.copyfile(Path('LICENSE'), target / 'LICENSE')
    print("Serve: python -m http.server 8000 --bind 127.0.0.1 --directory build/web")


if __name__ == "__main__":
    main()
