import argparse
import hashlib
import json
from pathlib import Path
from safetensors.torch import save_file
import torch
from .train import load_checkpoint


def export(checkpoint, output, generation_mode=None, polite_boundary_guard=False, normalize_morph_readings=False, lexical_boundary_guard=False, subword_start_temperature=None):
    if generation_mode not in (None, 'autoregressive'):
        raise ValueError('unsupported generation mode')
    model, ck = load_checkpoint(checkpoint)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    save_file({k: v.contiguous() for k, v in model.state_dict().items()}, out / "model.safetensors")
    vocab_bytes = ck['vocab_bytes']
    vocabulary = json.loads(vocab_bytes)
    if vocabulary.get('encoding') == 'sentencepiece_lexicon_v1':
        from collections import Counter
        training = Path(ck['config']['data_dir'])/'train.jsonl'
        if hashlib.sha256(training.read_bytes()).hexdigest() != ck['data_manifest']['split_sha256']['train.jsonl']:
            raise ValueError('reading-context source changed')
        lexicon = vocabulary['lexicon']
        ids = {(w['surface'],w['kana'],w['pos']):i for i,w in enumerate(lexicon)}
        variants = Counter((w['surface'],w['pos']) for w in lexicon)
        contexts = Counter()
        for line in training.read_text(encoding='utf-8').splitlines():
            segments = json.loads(line)['segments']
            for i,s in enumerate(segments):
                if variants[(s['surface'],s['pos'])] > 1:
                    key = (segments[i-2]['surface'] if i>=2 else '',segments[i-1]['surface'] if i>=1 else '',
                           ids[(s['surface'],s['kana'],s['pos'])])
                    contexts[key] += 1
        vocabulary['reading_contexts'] = [dict(previous=[a,b],word=i,count=n) for (a,b,i),n in sorted(contexts.items())]
        vocab_bytes = json.dumps(vocabulary,ensure_ascii=False,separators=(',',':')).encode()
    (out / "vocab.json").write_bytes(vocab_bytes)
    (out / "model-config.json").write_text(json.dumps(ck["model_config"], indent=2), encoding="utf-8")
    manifest = {"version": "0.1.0", "dtype": "F32", "parameter_count": model.parameter_count(),
                "vocab_sha256": hashlib.sha256(vocab_bytes).hexdigest(),
                "weights_sha256": hashlib.sha256((out / "model.safetensors").read_bytes()).hexdigest(),
                "config_sha256": hashlib.sha256((out / "model-config.json").read_bytes()).hexdigest(),
                "data_manifest": ck["data_manifest"], "training_config": ck["config"],
                "torch_version": ck["torch_version"], "seed": ck["config"]["seed"]}
    if generation_mode is not None:
        manifest.update(generation_mode='autoregressive', allow_static_fallback=False)
    if polite_boundary_guard:
        manifest['polite_boundary_guard'] = True
    if lexical_boundary_guard:
        manifest['lexical_boundary_guard'] = True
    if normalize_morph_readings:
        manifest['reading_normalization'] = 'japanese_morph_v1'
    if subword_start_temperature is not None:
        if vocabulary.get('encoding') != 'sentencepiece_lexicon_v1' or not .05 <= subword_start_temperature <= 2:
            raise ValueError('invalid subword start temperature')
        manifest['subword_start_temperature'] = subword_start_temperature
    if 'training_device' in ck:
        manifest['training_hardware'] = {key: ck.get(key) for key in ('training_device', 'device_name', 'cuda_version', 'peak_cuda_memory_bytes')}
    if 'source' in ck['data_manifest']:
        origin = ck['data_manifest']['source']
        manifest['data_sources'] = [{k:origin[k] for k in ('dataset','revision','license','authors','dataset_url','paper','sha256','attribution','license_url') if k in origin}]
        if origin.get('dataset') == 'SimpleStories/SimpleStories-JA':
            manifest['lexical_boundary_guard'] = True
            manifest['data_sources'].append(dict(source='Original paired Japanese corpus',license='CC0-1.0',
                rows=ck['data_manifest'].get('initial_counts',{}).get('train',0)))
        if origin.get('dataset') == 'Tatoeba Japanese':
            manifest['data_sources'].append(dict(source='Original paired/everyday Japanese prose', license='CC0-1.0'))
    if 'averaging_sources' in ck:
        manifest['averaging_sources'] = ck['averaging_sources']
    if ck['data_manifest'].get('license') == 'CC0-1.0':
        manifest['data_sources'] = [dict(source='Original paired Japanese prose', license='CC0-1.0',
                                       rows=ck['data_manifest']['splits']['train'],
                                       independent_human_review='pending')]
    if ck.get('initialization'):
        origin = ck['initialization']['source']
        manifest['initialization'] = ck['initialization']
        manifest.setdefault('data_sources', []).append({k:origin[k] for k in ('model','revision','license','authors','url')})
    if 'dictionary' in ck['data_manifest']:
        import importlib.metadata
        dictionary_info = ck['data_manifest']['dictionary']
        manifest.setdefault('data_sources', []).append(dictionary_info)
        distribution = importlib.metadata.distribution('SudachiDict-core')
        license_file = next(distribution.locate_file(p) for p in distribution.files
                            if str(p).replace('\\', '/').endswith('licenses/LICENSE-2.0.txt'))
        notices = ('Training text: Tatoeba.org and its contributors\n'
                   'https://tatoeba.org/en/downloads\n'
                   'CC BY 2.0 FR: https://creativecommons.org/licenses/by/2.0/fr/\n'
                   'Original paired/everyday prose: CC0-1.0; see data/paired-v1/LICENSE.\n\n'
                   'Dictionary-derived lexemes/readings: SudachiDict by Works Applications Co., Ltd.\n'
                   'https://github.com/WorksApplications/SudachiDict\n'
                   'Includes UniDic and a part of NEologd; see the upstream repository.\n\n'
                   + license_file.read_text(encoding='utf-8'))
        if ck.get('initialization'):
            origin = ck['initialization']['source']
            notices += ('\n\nPretrained initialization: ' + origin['model'] + '\n'
                        + origin['authors'] + '\n' + origin['url'] + '\n'
                        'Released by the authors under the MIT license.\n'
                        + (Path(__file__).resolve().parents[1] / 'LICENSE').read_text(encoding='utf-8').replace(
                            'Copyright (c) 2026 Fumiori contributors', 'Copyright (c) rinna Co., Ltd.') + '\n'
                        +
                        'Word embeddings were remapped and the model was fine-tuned for this application.\n')
        notices = notices.encode()
        (out/'THIRD_PARTY_NOTICES.txt').write_bytes(notices)
        manifest['third_party_notices_sha256'] = hashlib.sha256(notices).hexdigest()
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
    p.add_argument('--lexical-boundary-guard', action='store_true')
    p.add_argument('--subword-start-temperature', type=float)
    a = p.parse_args()
    export(a.checkpoint, a.output, a.generation_mode, a.polite_boundary_guard, a.normalize_morph_readings, a.lexical_boundary_guard, a.subword_start_temperature)
