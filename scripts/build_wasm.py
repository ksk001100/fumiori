"""Build local Worker demo. Requires installed wasm target and wasm-bindgen 0.2.127."""
import argparse
import hashlib
from html import escape
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
    parser.add_argument('--experimental-model', type=Path, help='Add a separately selectable experimental five-file runtime bundle')
    parser.add_argument('--experimental-label', help='Override the label of the optional experimental model')
    parser.add_argument('--output', type=Path, default=Path('build/web'), help='Use a separate directory to test a candidate without replacing the running demo')
    args = parser.parse_args()
    runtime_files = ('manifest.json', 'model-config.json', 'vocab.json', 'model.safetensors', 'LICENSE')
    def verify_runtime(directory):
        manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
        for name, key in [('model-config.json', 'config_sha256'), ('vocab.json', 'vocab_sha256'),
                          ('model.safetensors', 'weights_sha256')]:
            if hashlib.sha256((directory / name).read_bytes()).hexdigest() != manifest[key]:
                raise ValueError(f'checksum mismatch: {name}')
        for name in runtime_files:
            if not (directory / name).is_file():
                raise ValueError(f'missing runtime file: {name}')
    if args.runtime_model:
        verify_runtime(args.runtime_model)
        args.models = ['model-int8']
    if args.experimental_model:
        verify_runtime(args.experimental_model)
    experimental_label = args.experimental_label
    if args.experimental_model and experimental_label is None:
        size = sum((args.experimental_model/name).stat().st_size for name in runtime_files)/1_000_000
        experimental_label = f'小型の日常文試作（約{size:.1f}MB）' if args.experimental_model.name == 'everyday-v3-3mb' else f'試作モデル（約{size:.1f}MB）'
    subprocess.run(["cargo", "build", "--locked", "--release", "-p", "fumiori-wasm",
                    "--target", "wasm32-unknown-unknown"], check=True)
    out = args.output
    shutil.copytree("web", out, dirs_exist_ok=True)
    subprocess.run(["wasm-bindgen", "target/wasm32-unknown-unknown/release/fumiori_wasm.wasm",
                    "--target", "web", "--out-dir", str(out / "pkg")], check=True)
    # The HTML, Worker protocol, JS glue and WASM must belong to the same build.
    # A browser can otherwise reuse an older Worker after refreshing the page.
    assets = [out / 'worker.js', out / 'pkg/fumiori_wasm.js', out / 'pkg/fumiori_wasm_bg.wasm']
    version = hashlib.sha256(b''.join(p.read_bytes() for p in assets)).hexdigest()[:20]
    page = (out / 'index.html').read_text(encoding='utf-8')
    page = page.replace("new Worker('./worker.js',", f"new Worker('./worker.js?v={version}',")
    labels = {'model': '既存モデル v2（FP32）', 'model-int8': '既存モデル v2（約2.9MB）',
              'model-experimental': experimental_label}
    descriptions = {'model': '現在の標準モデルです。', 'model-int8': '現在の標準モデルです。',
                    'model-experimental': '品質を検証中の試作です。学習文の再現や重複、不自然な文・読みが残ります。'}
    displayed = args.models + (['model-experimental'] if args.experimental_model else [])
    start = page.index('<select id="model">') + len('<select id="model">')
    end = page.index('</select>', start)
    page = page[:start] + ''.join(f'<option value="{name}" data-description="{escape(descriptions[name], quote=True)}">{escape(labels[name])}</option>' for name in displayed) + page[end:]
    (out / 'index.html').write_text(page, encoding='utf-8')
    for model in displayed:
        runtime_source = args.experimental_model if model == 'model-experimental' else args.runtime_model
        if runtime_source:
            target = out / model
            target.mkdir(exist_ok=True)
            for name in runtime_files:
                shutil.copyfile(runtime_source / name, target / name)
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
    print("Serve: python scripts/serve_demo.py")


if __name__ == "__main__":
    main()
