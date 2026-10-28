"""Average aligned training runs into one model, with explicit source provenance."""
import hashlib
import json
from pathlib import Path
import time
import torch
from .model import loss_for
from .train import load_checkpoint, read_json, read_rows, tensor_batch


def average(config_path):
    c = read_json(config_path)
    sources = c['average_configs']
    if len(sources) < 2:
        raise ValueError('at least two aligned source configurations required')
    weights = c.get('average_weights', [1 / len(sources)] * len(sources))
    if len(weights) != len(sources) or any(w <= 0 for w in weights) or abs(sum(weights) - 1) > 1e-8:
        raise ValueError('averaging weights must be positive and sum to one')
    torch.set_num_threads(c.get('threads', 2))
    source_checkpoints, provenance = [], []
    for path in sources:
        config = read_json(path)
        if 'average_configs' in config:
            raise ValueError('averaging sources must be direct training configurations')
        checkpoint = Path(config['checkpoint'])
        if checkpoint.resolve() == Path(c['checkpoint']).resolve():
            raise ValueError('average output must differ from each source')
        reusable = False
        if checkpoint.exists():
            _, ck = load_checkpoint(checkpoint)
            reusable = ck['config'] == config and all(
                hashlib.sha256((Path(config['data_dir']) / file).read_bytes()).hexdigest() == expected
                for file, expected in ck['data_manifest']['split_sha256'].items())
            reusable = reusable and hashlib.sha256((Path(config['data_dir']) / 'vocab.json').read_bytes()).hexdigest() == ck['data_manifest']['vocab_sha256']
            reusable = reusable and read_json(Path(config['data_dir']) / 'manifest.json').get('source_generator_sha256') == ck['data_manifest'].get('source_generator_sha256')
        if not reusable:
            from .train import train
            train(path)
        _, ck = load_checkpoint(checkpoint)
        source_checkpoints.append(ck)
        provenance.append(dict(config_path=str(path), config=config,
            checkpoint=str(checkpoint), checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            selected_step=ck['selected_step']))
    first = source_checkpoints[0]
    if (c['seed'] != first['config']['seed'] or c['model'] != first['config']['model']
            or Path(c['data_dir']).resolve() != Path(first['config']['data_dir']).resolve()):
        raise ValueError('average configuration must match the source model and data')
    for other in source_checkpoints[1:]:
        if (other['model_config'] != first['model_config'] or other['vocab_bytes'] != first['vocab_bytes']
                or other['data_manifest']['split_sha256'] != first['data_manifest']['split_sha256']
                or other['config']['seed'] != first['config']['seed']):
            raise ValueError('averaging requires the same architecture, vocabulary, splits and initialization seed')
    model, _ = load_checkpoint(provenance[0]['checkpoint'])
    state = {name: sum(ck['state_dict'][name] * weight for ck, weight in zip(source_checkpoints, weights))
             for name in first['state_dict']}
    model.load_state_dict(state, strict=True)
    start = time.perf_counter()
    with torch.no_grad():
        validation = float(loss_for(model, tensor_batch(read_rows(Path(c['data_dir']) / 'validation.jsonl'), 'cpu')))
    ck = {key: value for key, value in first.items() if key not in ('state_dict', 'config', 'logs')}
    ck.update(state_dict=state, config=c, selected_step=None, best_validation_loss=validation,
              stopped_early=False, logs=[], averaging_sources=provenance,
              training_seconds=sum(s['training_seconds'] for s in source_checkpoints) + time.perf_counter() - start)
    out = Path(c['checkpoint']); out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ck, out)
    summary = {k: v for k, v in ck.items() if k not in ('state_dict', 'vocab', 'vocab_bytes')}
    out.with_suffix('.metrics.json').write_bytes(json.dumps(summary, ensure_ascii=False, indent=2).encode())
    print(json.dumps(dict(averaged_sources=len(sources), validation_loss=validation)), flush=True)
    return summary
