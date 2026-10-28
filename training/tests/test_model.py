import json
from pathlib import Path
import subprocess
import torch
import pytest
from training.model import ModelConfig, Transformer, loss_for
from training.train import load_checkpoint
from training.export import export


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.set_num_threads(2)
    torch.manual_seed(17)


def small_config():
    return ModelConfig(vocab_size=12, d_model=16, n_heads=4, n_layers=2, d_ff=32, max_seq_len=16)


def test_forward_causal_and_tied_weights():
    m = Transformer(small_config()).eval()
    a = torch.tensor([[0, 3, 7, 8, 9]])
    b = torch.tensor([[0, 3, 7, 10, 11]])
    with torch.no_grad():
        x, y = m(a), m(b)
    assert x.shape == (1, 5, 12)
    torch.testing.assert_close(x[:, :3], y[:, :3], rtol=0, atol=0)
    # No separately registered output projection: same parameter used by F.linear.
    assert not any("out_proj" in name or "lm_head" in name for name, _ in m.named_parameters())
    with pytest.raises(ValueError):
        m(torch.zeros((1, 17), dtype=torch.long))


def test_pad_mask_and_training_loss_descent():
    m = Transformer(small_config())
    batch = torch.tensor([[0, 3, 7, 8, 9, 10, 1]])
    padded = torch.cat([batch, torch.tensor([[2, 2, 2]])], dim=1)
    torch.testing.assert_close(loss_for(m, batch), loss_for(m, padded))
    initial = float(loss_for(m, batch).detach())
    optimizer = torch.optim.AdamW(m.parameters(), lr=0.01)
    for _ in range(40):
        optimizer.zero_grad()
        loss = loss_for(m, batch)
        loss.backward()
        optimizer.step()
    assert float(loss_for(m, batch).detach()) < initial * 0.2


def test_regularization_is_training_only_and_smoothing_ignores_pad():
    config = small_config()
    config.dropout = 0.2
    m = Transformer(config)
    batch = torch.tensor([[0, 3, 7, 8, 9, 10, 1]])
    assert not torch.equal(m(batch), m(batch))
    m.eval()
    torch.testing.assert_close(m(batch), m(batch), atol=0, rtol=0)
    padded = torch.cat([batch, torch.tensor([[2, 2]])], dim=1)
    torch.testing.assert_close(loss_for(m, batch, 0.1), loss_for(m, padded, 0.1))
    with pytest.raises(ValueError):
        ModelConfig(vocab_size=12, dropout=1)


def test_noun_corruption_changes_only_inputs_not_next_token_targets():
    from training.train import noun_context
    m = Transformer(small_config()).eval()
    batch = torch.tensor([[0, 3, 7, 8, 9, 10, 1, 2]])
    original = batch.clone()
    lookup = torch.zeros(12, dtype=torch.bool)
    lookup[7] = True
    context = noun_context(batch, lookup, 1.0)
    assert context.tolist() == [[0, 3, 6, 8, 9, 10, 1]]
    assert torch.equal(batch, original)
    expected = torch.nn.functional.cross_entropy(m(context).reshape(-1, 12), batch[:, 1:].reshape(-1), ignore_index=2)
    torch.testing.assert_close(loss_for(m, batch, context_ids=context), expected)
    torch.testing.assert_close(loss_for(m, batch, context_ids=noun_context(batch, lookup, 0.0)), loss_for(m, batch))


@pytest.fixture
def exported(tmp_path):
    specials = ["BOS", "EOS", "PAD", "EASY", "NORMAL", "HARD", "UNK"]
    tokens = [dict(id=i, surface=s, kana="", pos="special", metadata={}) for i, s in enumerate(specials)]
    for s, k, p in [("猫", "ねこ", "名詞"), ("が", "が", "助詞"), ("眠る", "ねむる", "動詞"), ("。", "", "記号"), ("犬", "いぬ", "名詞")]:
        tokens.append(dict(id=len(tokens), surface=s, kana=k, pos=p, metadata={}))
    vocab = dict(version="0.1.0", encoding="UTF-8", tokens=tokens)
    config = small_config()
    config.dropout = 0.2  # Export must disable training noise in both runtimes.
    m = Transformer(config).eval()
    ck = dict(state_dict=m.state_dict(), model_config=vars(m.config), vocab=vocab,
              vocab_bytes=json.dumps(vocab, ensure_ascii=False).encode(), data_manifest={},
              config={"seed": 17}, torch_version=str(torch.__version__))
    checkpoint = tmp_path / "model.pt"
    torch.save(ck, checkpoint)
    restored, _ = load_checkpoint(checkpoint)
    ids = torch.tensor([[0, 3, 7]])
    torch.testing.assert_close(m(ids), restored(ids), atol=0, rtol=0)
    out = tmp_path / "exported"
    export(checkpoint, out)
    return out


def cli(*args):
    return subprocess.run(["cargo", "run", "--quiet", "-p", "fumiori-cli", "--", *map(str, args)],
                          capture_output=True, encoding="utf-8", timeout=180)


def test_python_rust_parity(exported):
    result = cli("parity", "--model", exported)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["max_absolute_error"] < 2e-4


def test_authored_frame_constraints_are_explicit_optional_and_checked(exported, tmp_path):
    from training.constraints import attach_constraints
    from training.package import package
    row = dict(text='猫が眠る。', kana='ねこがねむる', segments=[
        dict(surface=s, kana=k, pos=p) for s,k,p in [('猫','ねこ','名詞'),('が','が','助詞'),('眠る','ねむる','動詞'),('。','','記号')]])
    report = attach_constraints(exported, [row])
    assert report['frames'] == 1
    runtime = tmp_path / 'frames-runtime'
    package(exported, runtime)
    result = cli('generate', '--model', runtime, '--difficulty', 'easy', '--seed', 42)
    assert result.returncode == 0, result.stderr
    prompt = json.loads(result.stdout)
    assert prompt['source'] == 'model_constrained' and prompt['text'] == '猫が眠る'
    raw = cli('generate', '--model', runtime, '--difficulty', 'easy', '--unconstrained', '--max-attempts', 1, '--seed', 42)
    assert raw.returncode == 0, raw.stderr
    assert json.loads(raw.stdout)['source'] != 'model_constrained'
    manifest_path = runtime / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    manifest['generation_constraints']['sequences'][0][1] = 50000
    manifest_path.write_text(json.dumps(manifest), encoding='utf-8')
    assert cli('generate', '--model', runtime, '--difficulty', 'easy').returncode != 0


def test_current_model_activation_replaces_old_bundle_and_installs_matching_exports(exported, tmp_path):
    import shutil
    from training.activate import activate
    from training.quantize import quantize
    build = tmp_path / 'build'
    shutil.copytree(exported, build / 'stories-v1-3mb')
    quantize(build / 'stories-v1-3mb', build / 'stories-v1-3mb-int8')
    old = build / 'model'
    old.mkdir()
    (old / 'manifest.json').write_bytes(b'previous-model')
    (old / 'notes.txt').write_bytes(b'keep this review')
    result = activate(build)
    assert result['backups'] == []
    assert not (build / 'model-history').exists()
    assert result['fp32_source'] == 'stories-v1-3mb'
    assert (old / 'notes.txt').read_bytes() == b'keep this review'
    assert (old / 'model.safetensors').read_bytes() == (exported / 'model.safetensors').read_bytes()
    assert cli('parity', '--model', old).returncode == 0
    assert cli('parity', '--model', build / 'model-int8').returncode == 0
    assert activate(build)['backups'] == []
    before = (old / 'manifest.json').read_bytes()
    (build / 'stories-v1-3mb-int8' / 'model.safetensors').write_bytes(b'broken')
    with pytest.raises(ValueError, match='checksum mismatch'):
        activate(build)
    assert (old / 'manifest.json').read_bytes() == before


@pytest.mark.parametrize("file", ["vocab.json", "model-config.json", "model.safetensors"])
def test_corrupt_model_rejected(exported, file):
    path = exported / file
    path.write_bytes(path.read_bytes() + b" ")
    result = cli("generate", "--model", exported)
    assert result.returncode != 0
    assert "checksum mismatch" in result.stderr


def test_model_missing_files_and_unknown_ids(exported):
    result = cli("logits", "--model", exported, "--ids", "0,3,50000")
    assert result.returncode != 0 and "unknown input ID" in result.stderr
    (exported / "model.safetensors").unlink()
    assert cli("generate", "--model", exported).returncode != 0


def test_fallback_and_bounded_retry_are_explicit(exported):
    result = cli("generate", "--model", exported, "--max-attempts", 0, "--count", 3, "--seed", 42)
    assert result.returncode == 0, result.stderr
    prompts = [json.loads(line) for line in result.stdout.splitlines()]
    assert len(prompts) == 3
    assert len({p["text"] for p in prompts}) == 3
    for p in prompts:
        assert p["source"] == "fallback"
        assert p["text"] == "".join(s["surface"] for s in p["segments"])
        assert p["kana"] == "".join(s["kana"] for s in p["segments"])
        assert 13 <= len(p["kana"]) <= 22
    assert cli("generate", "--model", exported, "--temperature", "NaN").returncode != 0


def test_no_implicit_fallback():
    assert cli("generate").returncode != 0
    assert cli("generate", "--model", "does-not-exist").returncode != 0


def test_skipping_near_diagnostic_does_not_change_generated_samples(exported, tmp_path):
    reference=tmp_path/'training.jsonl'
    reference.write_text(json.dumps(dict(text='猫が眠る。'),ensure_ascii=False)+'\n',encoding='utf-8')
    args=['evaluate','--model',exported,'--samples',3,'--seed',42,'--training-data',reference]
    regular=cli(*args,'--report',tmp_path/'regular.json')
    skipped=cli(*args,'--report',tmp_path/'skipped.json','--skip-near-statistics')
    assert regular.returncode==skipped.returncode==0,regular.stderr+skipped.stderr
    a,b=json.loads(regular.stdout),json.loads(skipped.stdout)
    assert a['samples']==b['samples']
    assert a['model_training_exact_rate']==b['model_training_exact_rate']
    assert b['model_training_near_rate'] is None


def test_autoregressive_bundle_disallows_static_fallback_and_preserves_policy(exported, tmp_path):
    from training.package import package
    path = exported / 'manifest.json'
    manifest = json.loads(path.read_text(encoding='utf-8'))
    manifest.update(generation_mode='autoregressive', allow_static_fallback=False,
                    generation_defaults=dict(temperature=1.0, top_p=.95))
    path.write_text(json.dumps(manifest), encoding='utf-8')
    runtime = tmp_path / 'open-runtime'
    package(exported, runtime)
    result = cli('generate', '--model', runtime, '--max-attempts', 0)
    assert result.returncode != 0 and 'static fallback is disabled' in result.stderr
    minimal = json.loads((runtime / 'manifest.json').read_text(encoding='utf-8'))
    assert minimal['generation_defaults'] == manifest['generation_defaults']
    minimal['generation_defaults']['temperature'] = -1
    (runtime / 'manifest.json').write_text(json.dumps(minimal), encoding='utf-8')
    assert cli('generate', '--model', runtime).returncode != 0


def test_mixed_difficulty_cycles_feasible_lengths_with_shared_history():
    args = ['generate', '--fallback-only', '--difficulty', 'mixed', '--count', '6', '--seed', '3432']
    first, repeated = cli(*args), cli(*args)
    assert first.returncode == repeated.returncode == 0, first.stderr
    assert first.stdout == repeated.stdout
    rows = [json.loads(line) for line in first.stdout.splitlines()]
    for i, row in enumerate(rows):
        lo, hi = [(5, 12), (13, 22), (23, 40)][i % 3]
        assert lo <= len(row['kana']) <= hi
        assert not row['text'].endswith('。')
        assert row['text'] == ''.join(s['surface'] for s in row['segments'])
    assert len({row['text'] for row in rows}) == 6
    restricted = cli(*args, '--min-kana', '13', '--max-kana', '22')
    assert restricted.returncode == 0, restricted.stderr
    assert all(13 <= len(json.loads(line)['kana']) <= 22 for line in restricted.stdout.splitlines())
    assert cli(*args, '--min-kana', '41', '--max-kana', '50').returncode != 0


def test_reference_parameter_budget():
    m = Transformer(ModelConfig(vocab_size=16000))
    assert m.parameter_count() == 2849536


def test_quantized_bundle_parity_and_missing_scale(exported, tmp_path):
    from training.quantize import quantize
    from safetensors.torch import load_file, save_file
    import hashlib
    out = tmp_path / "quantized"
    quantize(exported, out)
    result = cli("parity", "--model", out)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["max_absolute_error"] < 2e-4
    result = cli("generate", "--model", out, "--max-attempts", 0)
    assert result.returncode == 0, result.stderr
    weights = out / "model.safetensors"
    tensors = {name: tensor.clone() for name, tensor in load_file(weights).items()}
    tensors.pop("token_emb.weight.scale")
    save_file(tensors, weights)
    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["weights_sha256"] = hashlib.sha256(weights.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    result = cli("generate", "--model", out)
    assert result.returncode != 0 and "missing row scale" in result.stderr


def test_minimal_runtime_package_preserves_generation(exported, tmp_path):
    from training.package import package
    from training.quantize import quantize
    import zipfile
    quantized = tmp_path / 'quantized'
    quantize(exported, quantized)
    runtime = tmp_path / 'runtime'
    report = package(quantized, runtime)
    assert report['runtime_bytes'] == sum(p.stat().st_size for p in runtime.iterdir())
    first = runtime.with_suffix('.zip').read_bytes()
    assert package(quantized, runtime) == report
    assert runtime.with_suffix('.zip').read_bytes() == first
    with zipfile.ZipFile(runtime.with_suffix('.zip')) as archive:
        assert set(archive.namelist()) == {'manifest.json', 'model-config.json', 'model.safetensors', 'vocab.json', 'LICENSE'}
    # Load/logits verifies the compacted JSON hashes and unchanged IDs/weights.
    a = cli('logits', '--model', quantized, '--ids', '0,3,7')
    b = cli('logits', '--model', runtime, '--ids', '0,3,7')
    assert a.returncode == b.returncode == 0, b.stderr
    assert json.loads(a.stdout) == json.loads(b.stdout)
    (quantized / 'vocab.json').write_bytes(b'{}')
    with pytest.raises(ValueError, match='checksum mismatch'):
        package(quantized, runtime)


def test_fixed_idiom_completes_without_fallback(exported):
    import hashlib
    from training.constraints import attach_constraints
    path=exported/'vocab.json'
    vocab=json.loads(path.read_text(encoding='utf-8'))
    vocab['tokens'][11].update(surface='一期一会',kana='いちごいちえ',pos='成句')
    payload=json.dumps(vocab,ensure_ascii=False).encode()
    path.write_bytes(payload)
    manifest_path=exported/'manifest.json'
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    manifest['vocab_sha256']=hashlib.sha256(payload).hexdigest()
    manifest_path.write_text(json.dumps(manifest),encoding='utf-8')
    attach_constraints(exported,[dict(text='一期一会。',kana='いちごいちえ',segments=[
        dict(surface='一期一会',kana='いちごいちえ',pos='成句'),dict(surface='。',kana='',pos='記号')],
        generation=dict(event_family='fixed-idiom',family='fixed-idiom',topic='idiom'))])
    result=cli('generate','--model',exported,'--difficulty','easy','--seed',42)
    assert result.returncode==0,result.stderr
    prompt=json.loads(result.stdout)
    assert prompt['text']=='一期一会' and prompt['source']=='model_constrained'
