"""Merge offline teacher prose without changing subword IDs or leaking held-out bodies."""
from collections import Counter
import argparse
import copy
import hashlib
import json
from pathlib import Path

from scripts.generalization import evaluation_text
from scripts.lexical_rules import lexical_collision
from scripts.prepare_tatoeba import analyze
from scripts.semantic_keys import core_key


def split_for(text):
    bucket = int(hashlib.sha256(core_key(evaluation_text(text)).encode()).hexdigest()[:8],16) % 100
    return 'validation' if bucket < 5 else 'test' if bucket < 10 else 'train'


def orthographic_readings(segments):
    # A curated spelling correction, not a guessed reading for unknown kanji.
    # Sudachi can return spoken ユウ for 言う; typing uses orthographic いう.
    return [dict(s, kana='いう'+s['kana'][2:]) if s['pos'].startswith('動詞')
            and s['surface'].startswith('言う') and s['kana'].startswith('ゆう') else s
            for s in segments]


def prepare(raw_paths, output, base=Path('build/everyday-v6-data'), max_new_words=3500):
    from sudachipy import Dictionary, SplitMode
    import sentencepiece as spm
    import torch
    manifest = json.loads((base/'manifest.json').read_text(encoding='utf-8'))
    vocab_bytes = (base/'vocab.json').read_bytes()
    if hashlib.sha256(vocab_bytes).hexdigest() != manifest['vocab_sha256']:
        raise ValueError('base vocabulary changed')
    vocabulary = json.loads(vocab_bytes)
    parts, protected = {}, set()
    for split, expected in manifest['split_sha256'].items():
        payload = (base/split).read_bytes()
        if hashlib.sha256(payload).hexdigest() != expected:
            raise ValueError('base split changed')
        rows = [json.loads(line) for line in payload.decode('utf-8').splitlines()]
        parts[split.removesuffix('.jsonl')] = rows
        protected.update(core_key(evaluation_text(r['text'])) for r in rows)
    tokenizer = Dictionary().tokenizer()
    source = Path('build/pretrained/rinna-xsmall')
    origin = json.loads((source/'source.json').read_text())
    if hashlib.sha256((source/'spiece.model').read_bytes()).hexdigest() != origin['files']['spiece.model']['sha256']:
        raise ValueError('pretrained tokenizer changed')
    sp = spm.SentencePieceProcessor(model_file=str(source/'spiece.model'))
    piece_ids = {token['surface']: token['id'] for token in vocabulary['tokens'][7:]}
    lexicon = vocabulary['lexicon']
    word_keys = {(w['surface'],w['kana'],w['pos']) for w in lexicon}
    candidates, seen, stats, sources = [], set(protected), Counter(), []
    for path in raw_paths:
        info = json.loads(path.with_suffix('.manifest.json').read_text(encoding='utf-8'))
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != info['raw_sha256']:
            raise ValueError('raw teacher output changed')
        sources.append(info)
        for line in payload.decode('utf-8').splitlines():
            record = json.loads(line)
            for text in record['sentences']:
                stats['raw_sentences'] += 1
                core = core_key(evaluation_text(text))
                if core in seen:
                    stats['duplicate_or_existing_body'] += 1
                    continue
                seen.add(core)
                segments = analyze(text, tokenizer, SplitMode.C)
                if segments is None or any(lexical_collision(a,b) for a,b in zip(segments,segments[1:])):
                    stats['dictionary_or_boundary'] += 1
                    continue
                segments = orthographic_readings(segments)
                kana = ''.join(s['kana'] for s in segments)
                if not 5 <= len(kana) <= 40:
                    stats['kana_length'] += 1
                    continue
                raw_ids = sp.encode(text,out_type=int)
                if 0 in raw_ids or len(raw_ids)+3 > 64 or any(sp.id_to_piece(i) not in piece_ids for i in raw_ids):
                    stats['unrepresented_subword'] += 1
                    continue
                difficulty = 3 if len(kana) <= 12 else 4 if len(kana) <= 22 else 5
                row = dict(text=text,kana=kana,segments=segments,trust='auto_aligned',
                    reading_validation='Sudachi dictionary annotation; not independently verified',
                    ids=[0,difficulty,*[piece_ids[sp.id_to_piece(i)] for i in raw_ids],1],
                    training_group='teacher',generation=dict(requested_topic=record['topic'],topic_review='pending'),
                    source=dict(teacher_model=info['teacher']['model'],revision=info['teacher']['revision'],
                                raw_file_sha256=info['raw_sha256'],request_index=record['index']))
                candidates.append((split_for(text),row))
    counts = Counter((s['surface'],s['kana'],s['pos']) for split,r in candidates if split=='train' for s in r['segments'])
    new_keys = [key for key,_ in counts.most_common() if key not in word_keys][:max_new_words]
    for surface,kana,pos in new_keys:
        lexicon.append(dict(id=len(lexicon)+7,surface=surface,kana=kana,pos=pos,
                            metadata=dict(count=counts[(surface,kana,pos)])))
    word_keys.update(new_keys)
    for split,row in candidates:
        if any((s['surface'],s['kana'],s['pos']) not in word_keys for s in row['segments']):
            stats['word_not_in_training_lexicon_'+split] += 1
            continue
        parts[split].append(row)
        stats['accepted_'+split] += 1
        stats['accepted_'+str(row['ids'][1])] += 1
    if stats['accepted_train'] < 50 or not stats['accepted_validation']:
        raise ValueError(f'insufficient distinct teacher prose: {dict(stats)}')
    output.mkdir(parents=True,exist_ok=True)
    vocabulary_payload = json.dumps(vocabulary,ensure_ascii=False,separators=(',',':')).encode()
    (output/'vocab.json').write_bytes(vocabulary_payload)
    result = copy.deepcopy(manifest)
    result.update(description='Japanese teacher sequence distillation mixed with the original subword corpus',
                  parent_manifest_sha256=hashlib.sha256((base/'manifest.json').read_bytes()).hexdigest(),
                  vocab_sha256=hashlib.sha256(vocabulary_payload).hexdigest(),
                  synthetic_sources=sources,teacher_preparation_stats=dict(stats),
                  new_dictionary_words=len(new_keys),splits={},split_sha256={},
                  teacher_split_strategy='SHA256 normalized body: 90% train/5% validation/5% test; all prior bodies excluded',
                  source_generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    for split,rows in parts.items():
        payload = ''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows).encode()
        (output/(split+'.jsonl')).write_bytes(payload)
        result['splits'][split] = len(rows)
        result['split_sha256'][split+'.jsonl'] = hashlib.sha256(payload).hexdigest()
    (output/'manifest.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    checkpoint = Path('build/checkpoints/everyday-v8-32mb.pt')
    parent = torch.load(checkpoint,map_location='cpu',weights_only=True)
    # Only the word annotation lexicon grows. Neural subword IDs/weights are unchanged.
    if json.loads(parent['vocab_bytes'])['tokens'] != vocabulary['tokens']:
        raise ValueError('warm start subword IDs changed')
    warm_start = dict(state_dict=parent['state_dict'],model_config=parent['model_config'],
                     vocab_sha256=result['vocab_sha256'],source=dict(parent['initialization']['source'],
                     warm_start_model='everyday-v8-32mb',warm_start_checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest()))
    torch.save(warm_start,output/'warm-start.pt')
    print(json.dumps(dict(stats=stats,new_words=len(new_keys),splits=result['splits'])),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw',type=Path,nargs='+',required=True)
    parser.add_argument('--output',type=Path,default=Path('build/everyday-v9-data'))
    args = parser.parse_args()
    prepare(args.raw,args.output)
