import argparse
import hashlib
import json
from pathlib import Path
from safetensors.torch import save_file
import torch
from .train import load_checkpoint


def export(checkpoint, output, generation_mode=None, polite_boundary_guard=False, normalize_morph_readings=False):
    if generation_mode not in (None, 'autoregressive'):
        raise ValueError('unsupported generation mode')
    model, ck = load_checkpoint(checkpoint)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    save_file({k: v.contiguous() for k, v in model.state_dict().items()}, out / "model.safetensors")
    (out / "vocab.json").write_bytes(ck["vocab_bytes"])
    (out / "model-config.json").write_text(json.dumps(ck["model_config"], indent=2), encoding="utf-8")
    manifest = {"version": "0.1.0", "dtype": "F32", "parameter_count": model.parameter_count(),
                "vocab_sha256": hashlib.sha256(ck["vocab_bytes"]).hexdigest(),
                "weights_sha256": hashlib.sha256((out / "model.safetensors").read_bytes()).hexdigest(),
                "config_sha256": hashlib.sha256((out / "model-config.json").read_bytes()).hexdigest(),
                "data_manifest": ck["data_manifest"], "training_config": ck["config"],
                "torch_version": ck["torch_version"], "seed": ck["config"]["seed"]}
    if generation_mode is not None:
        manifest.update(generation_mode='autoregressive', allow_static_fallback=False)
    if polite_boundary_guard:
        manifest['polite_boundary_guard'] = True
    if normalize_morph_readings:
        manifest['reading_normalization'] = 'japanese_morph_v1'
    if 'training_device' in ck:
        manifest['training_hardware'] = {key: ck.get(key) for key in ('training_device', 'device_name', 'cuda_version', 'peak_cuda_memory_bytes')}
    if 'source' in ck['data_manifest']:
        origin = ck['data_manifest']['source']
        manifest['data_sources'] = [{k:origin[k] for k in ('dataset','revision','license','authors','dataset_url','paper','sha256') if k in origin}]
        if origin.get('dataset') == 'SimpleStories/SimpleStories-JA':
            manifest['lexical_boundary_guard'] = True
            manifest['data_sources'].append(dict(source='Original paired Japanese corpus',license='CC0-1.0',
                rows=ck['data_manifest'].get('initial_counts',{}).get('train',0)))
    if 'averaging_sources' in ck:
        manifest['averaging_sources'] = ck['averaging_sources']
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    fixtures = []
    with torch.no_grad():
        for ids in ([0, 3], [0, 4, 7], [0, 5, *[min(i, model.config.vocab_size - 1) for i in (8, 9, 10)]]):
            fixtures.append({"ids": ids, "logits": model(torch.tensor([ids]))[0].tolist()})
    (out / "parity.json").write_text(json.dumps(fixtures), encoding="utf-8")
    print(json.dumps({"parameter_count": model.parameter_count(), "files_bytes": {p.name: p.stat().st_size for p in out.iterdir() if p.is_file()}}))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument('--generation-mode', choices=['autoregressive'])
    p.add_argument('--polite-boundary-guard', action='store_true')
    p.add_argument('--normalize-morph-readings', action='store_true')
    a = p.parse_args()
    export(a.checkpoint, a.output, a.generation_mode, a.polite_boundary_guard, a.normalize_morph_readings)
