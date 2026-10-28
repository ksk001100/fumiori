"""Symmetric per-row INT8 storage; runtime explicitly dequantizes to FP32."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def quantize(source, output):
    source, output = Path(source), Path(output)
    if source.resolve() == output.resolve():
        raise ValueError("output must differ from source")
    output.mkdir(parents=True, exist_ok=True)
    tensors, restored = {}, {}
    for name, weight in load_file(source / "model.safetensors").items():
        if weight.ndim == 2:
            scale = weight.abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / 127
            q = (weight / scale).round().clamp(-127, 127).to(torch.int8)
            tensors[name] = (q.to(torch.int16) + 128).to(torch.uint8)
            tensors[name + ".scale"] = scale
            restored[name] = q.float() * scale
        else:
            tensors[name] = restored[name] = weight
    save_file(tensors, output / "model.safetensors")
    for name in ("vocab.json", "model-config.json"):
        shutil.copyfile(source / name, output / name)
    if (source / 'THIRD_PARTY_NOTICES.txt').exists():
        shutil.copyfile(source / 'THIRD_PARTY_NOTICES.txt', output / 'THIRD_PARTY_NOTICES.txt')
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    manifest.update(dtype="I8_ROW_F32", quantization={
        "scheme": "symmetric_per_row", "storage": "U8_offset_128", "range": [-127, 127],
        "compute_dtype": "F32", "dequantization": "once_at_load",
        "source_weights_sha256": manifest["weights_sha256"]})
    manifest["weights_sha256"] = hashlib.sha256((output / "model.safetensors").read_bytes()).hexdigest()
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    # Golden fixtures describe the dequantized model, not the original FP32 model.
    from .model import ModelConfig, Transformer
    model = Transformer(ModelConfig(**json.loads((output / "model-config.json").read_text())))
    model.load_state_dict(restored)
    model.eval()
    cases = json.loads((source / "parity.json").read_text())
    with torch.no_grad():
        for case in cases:
            case["logits"] = model(torch.tensor([case["ids"]]))[0].tolist()
    (output / "parity.json").write_text(json.dumps(cases), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    quantize(args.model, args.output)
