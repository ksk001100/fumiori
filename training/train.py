"""Train only from explicitly prepared local files. Never fetch data."""
import argparse
import hashlib
import json
import math
import os
import platform
from pathlib import Path
import time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
from .model import ModelConfig, Transformer, greedy, loss_for


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_rows(path):
    return [json.loads(s) for s in Path(path).read_text(encoding="utf-8").splitlines() if s]


def mixture_loss(losses, weights):
    if (not losses or set(losses) != set(weights)
            or any(not math.isfinite(v) or v <= 0 for v in weights.values())
            or any(not math.isfinite(v) or v < 0 for v in losses.values())):
        raise ValueError('validation mixture needs positive masses and losses for every observed group')
    return sum(weights[k]*losses[k] for k in losses)/sum(weights.values())


def tensor_batch(rows, device):
    if not rows:
        raise ValueError("empty dataset split")
    length = max(len(r["ids"]) for r in rows)
    if length > 64:
        raise ValueError("sequence exceeds 64")
    return torch.tensor([r["ids"] + [2] * (length - len(r["ids"])) for r in rows], device=device)


def load_checkpoint(path):
    ck = torch.load(path, map_location="cpu", weights_only=True)
    model = Transformer(ModelConfig(**ck["model_config"]))
    model.load_state_dict(ck["state_dict"], strict=True)
    return model.eval(), ck


def sampling_weights(rows, lexical_balance=None, topic_balance=None, corpus_mixture=None):
    """Equal expected mass for each observed difficulty; no held-out rows used."""
    counts = {d: sum(r['ids'][1] == d for r in rows) for d in (3, 4, 5)}
    if lexical_balance is None and topic_balance is None and corpus_mixture is None:
        return torch.tensor([1.0 / counts[r['ids'][1]] for r in rows], dtype=torch.float64)
    from scripts.diversity import lexical_factors, topic_factors
    factors = lexical_factors(rows, **lexical_balance) if lexical_balance is not None else [1.0] * len(rows)
    if topic_balance is not None:
        factors = [a * b for a, b in zip(factors, topic_factors(rows, **topic_balance))]
    if corpus_mixture is not None:
        if (not corpus_mixture or any(not isinstance(k, str) or not math.isfinite(v) or v <= 0
                                     for k, v in corpus_mixture.items())
                or any(r.get('training_group') not in corpus_mixture for r in rows)):
            raise ValueError('corpus mixture requires positive finite masses and explicit training_group labels')
        groups = {(r['ids'][1], r['training_group']) for r in rows}
        group_totals = {key: sum(f for r, f in zip(rows, factors)
                                if (r['ids'][1], r['training_group']) == key) for key in groups}
        factors = [f * corpus_mixture[r['training_group']] / group_totals[(r['ids'][1], r['training_group'])]
                   for r, f in zip(rows, factors)]
    totals = {d: sum(f for r, f in zip(rows, factors) if r['ids'][1] == d) for d in counts}
    return torch.tensor([f / totals[r['ids'][1]] for r, f in zip(rows, factors)], dtype=torch.float64)


def noun_context(batch, noun_ids, probability):
    """Mask input nouns only; original next-token targets stay untouched."""
    inputs = batch[:, :-1]
    mask = noun_ids[inputs] & (torch.rand(inputs.shape, device=inputs.device) < probability)
    return inputs.masked_fill(mask, 6)


def learning_rate_at(config, step):
    base = config['learning_rate']
    schedule = config.get('lr_schedule', 'constant')
    warmup = config.get('warmup_steps', 0)
    minimum = config.get('min_lr_ratio', .1)
    if schedule not in ('constant', 'cosine') or not 0 <= warmup < max(1, config['steps']) or not 0 <= minimum <= 1:
        raise ValueError('invalid learning rate schedule')
    if warmup and step < warmup:
        return base * (step + 1) / warmup
    if schedule == 'constant':
        return base
    progress = min(1.0, max(0.0, (step - warmup) / max(1, config['steps'] - warmup)))
    return base * (minimum + (1 - minimum) * .5 * (1 + math.cos(math.pi * progress)))


def train(config_path):
    c = read_json(config_path)
    if 'average_configs' in c:
        from .average import average
        return average(config_path)
    torch.set_num_threads(c.get("threads", 2))
    torch.manual_seed(c["seed"])
    torch.use_deterministic_algorithms(True)
    device = torch.device(c.get("device", "cpu"))
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; install CUDA-enabled PyTorch. CPU fallback is not automatic.')
    precision = c.get('precision', 'fp32')
    if precision not in ('fp32', 'bf16'):
        raise ValueError('precision must be fp32 or bf16')
    use_bf16 = precision == 'bf16'
    if use_bf16 and (device.type != 'cuda' or not torch.cuda.is_bf16_supported()):
        raise ValueError('bf16 training requires a supported CUDA device')
    device_name = torch.cuda.get_device_name(device) if device.type == 'cuda' else platform.processor()
    print(json.dumps(dict(training_device=str(device), device_name=device_name, torch_version=str(torch.__version__)), ensure_ascii=True), flush=True)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    data = Path(c["data_dir"])
    vocab = read_json(data / "vocab.json")
    manifest = read_json(data / "manifest.json")
    if hashlib.sha256((data / "vocab.json").read_bytes()).hexdigest() != manifest["vocab_sha256"]:
        raise ValueError("vocabulary hash mismatch")
    for file, expected in manifest["split_sha256"].items():
        if hashlib.sha256((data / file).read_bytes()).hexdigest() != expected:
            raise ValueError(f"prepared split hash mismatch: {file}")
    rows = {s: read_rows(data / f"{s}.jsonl") for s in ("train", "validation", "test")}
    if c.get('require_generation_eligible', False):
        from scripts.diversity import require_generation_eligible
        for split in rows.values():
            require_generation_eligible(split)
    for split in rows.values():
        for r in split:
            if r["trust"] != "verified" and not c.get("allow_auto_aligned", False):
                raise ValueError("unverified training data requires explicit allow_auto_aligned")
            ids = r["ids"]
            if not (3 <= len(ids) <= 64 and ids[0] == 0 and ids[1] in (3, 4, 5)
                    and ids[-1] == 1 and all(7 <= i < len(vocab["tokens"]) for i in ids[2:-1])):
                raise ValueError("invalid prepared token sequence")
            if vocab.get('encoding') == 'sentencepiece_lexicon_v1':
                text = ''.join(vocab['tokens'][i]['surface'] for i in ids[2:-1]).replace('▁','')
                kana = ''.join(s['kana'] for s in r['segments'])
                if ''.join(s['surface'] for s in r['segments']) != r['text']:
                    raise ValueError('subword lexical segments differ from text')
            else:
                text = ''.join(vocab['tokens'][i]['surface'] for i in ids[2:-1])
                kana = ''.join(vocab['tokens'][i]['kana'] for i in ids[2:-1])
            if text != r['text'] or kana != r['kana']:
                raise ValueError("prepared data roundtrip mismatch")
    model = Transformer(ModelConfig(vocab_size=len(vocab["tokens"]), **c["model"])).to(device)
    initialization = None
    if c.get('initialize_from'):
        initial_path = Path(c['initialize_from'])
        initial = torch.load(initial_path, map_location='cpu', weights_only=True)
        if initial.get('vocab_sha256') != manifest['vocab_sha256']:
            raise ValueError('pretrained initialization vocabulary differs')
        model.load_state_dict(initial['state_dict'], strict=True)
        initialization = dict(checkpoint_sha256=hashlib.sha256(initial_path.read_bytes()).hexdigest(),
                              source=initial['source'])
    smoothing = c.get('label_smoothing', 0.0)
    if not 0 <= smoothing < 1:
        raise ValueError('label_smoothing must be in [0, 1)')
    context_dropout = c.get('noun_context_dropout', 0.0)
    if not 0 <= context_dropout < 1:
        raise ValueError('noun_context_dropout must be in [0, 1)')
    noun_ids = torch.tensor([t['pos'] == '名詞' for t in vocab['tokens']], device=device)
    if c.get('body_lr_ratio') is not None:
        ratio = c['body_lr_ratio']
        if not 0 < ratio <= 1:
            raise ValueError('body_lr_ratio must be in (0, 1]')
        embedding = list(model.token_emb.parameters()) + list(model.pos_emb.parameters())
        body = [p for name,p in model.named_parameters() if not name.startswith(('token_emb.', 'pos_emb.'))]
        parameters = [dict(params=embedding, lr_scale=1.0), dict(params=body, lr_scale=ratio)]
    else:
        parameters = model.parameters()
    optimizer = torch.optim.AdamW(parameters, lr=c["learning_rate"], weight_decay=c.get('weight_decay', 0.01))
    train_batch = tensor_batch(rows["train"], device)
    if any(c.get(k) is not None for k in ('lexical_balance', 'topic_balance', 'corpus_mixture')) and not c.get('balance_difficulties', False):
        raise ValueError('lexical/topic balance requires balance_difficulties')
    weights = sampling_weights(rows['train'], c.get('lexical_balance'), c.get('topic_balance'), c.get('corpus_mixture')).to(device) if c.get('balance_difficulties', False) else None
    val_batch = tensor_batch(rows["validation"], device)
    validation_weights = c.get('validation_corpus_mixture')
    validation_groups = None
    if validation_weights is not None:
        names = {r.get('training_group') for r in rows['validation']}
        mixture_loss({name:0. for name in names},validation_weights)
        validation_groups = {name:val_batch[torch.tensor(
            [i for i,r in enumerate(rows['validation']) if r['training_group']==name],device=device)] for name in names}
    train_eval = train_batch
    if c.get('train_loss_samples'):
        import random
        indices = random.Random(c['seed'] + 901).sample(range(len(train_batch)), min(c['train_loss_samples'],len(train_batch)))
        train_eval = train_batch[torch.tensor(indices,device=device)]
    out = Path(c["checkpoint"])
    out.parent.mkdir(parents=True, exist_ok=True)
    logs = []
    start = time.perf_counter()
    batch_size = c.get("batch_size", 32)
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    def measured_loss(batch):
        weighted, count = 0.0, 0
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
            for part in batch.split(batch_size):
                n = int((part[:, 1:] != 2).sum())
                weighted += float(loss_for(model, part)) * n
                count += n
        return weighted / count

    best_state, best_validation_loss, best_step, bad_checks = None, math.inf, 0, 0
    patience = c.get("early_stopping_patience", 0)
    select_best = c.get("select_best_validation", False)
    min_delta = c.get("validation_min_delta", 0.0)
    if patience < 0 or min_delta < 0:
        raise ValueError("early stopping settings must be nonnegative")
    stopped_early = False
    for step in range(c["steps"] + 1):
        rate = learning_rate_at(c, step)
        for group in optimizer.param_groups:
            group['lr'] = rate * group.get('lr_scale', 1.0)
        model.train()
        if weights is not None:
            indices = torch.multinomial(weights, batch_size, replacement=True)
        else:
            indices = torch.randperm(len(train_batch), device=device)[:batch_size] if len(train_batch) > batch_size else None
        sampled = train_batch if indices is None else train_batch[indices]
        context = noun_context(sampled, noun_ids, context_dropout) if context_dropout else None
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
            loss = loss_for(model, sampled, smoothing, context_ids=context)
        if step % c.get("log_every", 50) == 0 or step == c["steps"]:
            model.eval()
            group_losses = {name:measured_loss(batch) for name,batch in validation_groups.items()} if validation_groups is not None else None
            val = mixture_loss(group_losses,validation_weights) if group_losses is not None else measured_loss(val_batch)
            entry = dict(step=step, train_loss=measured_loss(train_eval), train_loss_rows=len(train_eval), validation_loss=val,
                         learning_rate=rate,
                         validation_perplexity=math.exp(min(val, 80)),
                         samples={str(d): greedy(model, d, vocab) for d in (3, 4, 5)})
            if group_losses is not None:
                entry['validation_group_losses'] = group_losses
                entry['validation_selection'] = 'configured corpus mixture; per-group losses retained'
            if val < best_validation_loss - min_delta:
                best_validation_loss, best_step, bad_checks = val, step, 0
                if select_best:
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                bad_checks += 1
            stopped_early = patience > 0 and bad_checks >= patience
            entry["best_step"] = best_step
            entry["early_stopping_bad_checks"] = bad_checks
            logs.append(entry)
            print(json.dumps({k: v for k, v in entry.items() if k != "samples"}), flush=True)
        if step == c["steps"] or stopped_early:
            break
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    if select_best and best_state is not None:
        model.load_state_dict(best_state)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    metadata = dict(training_device=str(device), device_name=device_name, cuda_version=torch.version.cuda, initialization=initialization,
                    peak_cuda_memory_bytes=torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None, selected_step=best_step if select_best else step,
                    best_validation_loss=best_validation_loss, stopped_early=stopped_early,
                    config=c, model_config=vars(model.config), vocab=vocab,
                    vocab_bytes=(data / "vocab.json").read_bytes(), data_manifest=manifest,
                    torch_version=str(torch.__version__), python=platform.python_version(),
                    platform=platform.platform(), parameter_count=model.parameter_count(),
                    training_seconds=time.perf_counter() - start, logs=logs)
    torch.save(dict(state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()}, **metadata), out)
    # Byte payload retained in checkpoint to preserve the exact cross-language hash.
    summary = {k: v for k, v in metadata.items() if k not in ("vocab_bytes", "vocab")}
    out.with_suffix(".metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    train(parser.parse_args().config)
