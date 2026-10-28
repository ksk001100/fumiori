"""Prepare dictionary-backed Tatoeba prose, retaining source and reading provenance.

Only readings that agree with the supplied transcription are admitted. OOV and
proper nouns are rejected. No character spelling fallback or guessed kanji
reading is used. Human transcription review is recorded separately from this
automatic agreement check and from sentence-quality review.
"""
from collections import Counter
import argparse
import bz2
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import unicodedata

from scripts.semantic_keys import core_key


def hiragana(text):
    return ''.join(chr(ord(c)-0x60) if '\u30a1' <= c <= '\u30f6' else c for c in text)


def transcription(text):
    surface = re.sub(r'\[([^|\[\]]+)\|([^\[\]]*)\]', lambda m: m[1], text)
    reading = re.sub(r'\[([^|\[\]]+)\|([^\[\]]*)\]', lambda m: m[2].replace('|', ''), text)
    if any(c in surface + reading for c in '[]|'):
        return None
    return unicodedata.normalize('NFC', surface), hiragana(unicodedata.normalize('NFC', reading))


def normalize_sentence(text):
    text = unicodedata.normalize('NFC', text.strip())
    if text.endswith(('。', '！', '？', '!', '?')):
        text = text[:-1]
    if not text or not re.fullmatch(r'[ぁ-ゖァ-ヺ一-龯々ー、]+', text):
        return None
    return text + '。'


def analyze(text, tokenizer, mode):
    morphs = list(tokenizer.tokenize(text, mode))
    if any(m.is_oov() or '固有名詞' in m.part_of_speech() or '人名' in m.part_of_speech() for m in morphs):
        return None
    segments = []
    i = 0
    while i < len(morphs):
        m = morphs[i]
        details = m.part_of_speech()
        pos = details[0]
        if pos == '補助記号':
            if m.surface() not in ('。', '、'):
                return None
            segments.append(dict(surface=m.surface(), kana='、' if m.surface() == '、' else '', pos='記号'))
            i += 1
            continue
        pos = '名詞' if pos == '代名詞' else pos
        surface = m.surface()
        kana = surface if pos == '助詞' and surface in ('は', 'へ', 'を') else hiragana(m.reading_form())
        last = details
        if pos in ('動詞', '形容詞', '助動詞'):
            j = i + 1
            while j < len(morphs):
                n = morphs[j]
                nd = n.part_of_speech()
                attach = (nd[0] == '助動詞'
                          or (nd[0] == '助詞' and (n.surface() in ('て', 'で') or nd[1] == '終助詞'))
                          or (nd[0] == '動詞' and nd[1] == '非自立可能'))
                if not attach:
                    break
                surface += n.surface()
                kana += hiragana(n.reading_form())
                if nd[0] != '助詞':
                    last = nd
                j += 1
            if not last[5].startswith(('終止形', '連体形', '命令形')):
                pos += '-非終止'
            i = j
        else:
            i += 1
        if not kana or not re.fullmatch(r'[ぁ-ゖー]+', kana):
            return None
        segments.append(dict(surface=surface, kana=kana, pos=pos))
    if len(segments) < 2 or segments[-2]['pos'] not in ('動詞', '形容詞', '助動詞', '名詞'):
        return None
    # Existing runtime grammar rejects adjacent particle tokens.
    if any(a['pos'] == b['pos'] == '助詞' for a, b in zip(segments, segments[1:])):
        return None
    return segments


def prepare(source, out, max_vocab=8000):
    from sudachipy import Dictionary, SplitMode
    source_info = json.loads((source/'source.json').read_text(encoding='utf-8'))
    for name, info in source_info['files'].items():
        if hashlib.sha256((source/name).read_bytes()).hexdigest() != info['sha256']:
            raise ValueError(f'upstream source changed: {name}')
    readings = {}
    with bz2.open(source/'jpn_transcriptions.tsv.bz2', 'rt', encoding='utf-8') as file:
        for line in file:
            fields = line.rstrip('\n').split('\t')
            if len(fields) != 5 or fields[2] != 'Hrkt':
                continue
            parsed = transcription(fields[4])
            if parsed:
                reviewed = fields[3] not in ('', '\\N')
                if fields[0] not in readings or reviewed:
                    readings[fields[0]] = (*parsed, fields[3], reviewed)
    tokenizer = Dictionary().tokenizer()
    rows, stats, seen = [], Counter(), set()
    # Reserve all older held-out bodies; none may become Tatoeba training rows.
    protected = set()
    for split in ('validation', 'test', 'family-test'):
        for line in Path('data/paired-v1', split+'.jsonl').read_text(encoding='utf-8').splitlines():
            protected.add(core_key(json.loads(line)['text']))
    with bz2.open(source/'jpn_sentences_detailed.tsv.bz2', 'rt', encoding='utf-8') as file:
        for line in file:
            fields = line.rstrip('\n').split('\t')
            if len(fields) != 6 or fields[1] != 'jpn':
                continue
            sid, _, original, author, _, modified = fields
            text = normalize_sentence(original)
            supplied = readings.get(sid)
            if text is None or supplied is None or normalize_sentence(supplied[0]) != text:
                stats['shape_or_missing_transcription'] += 1
                continue
            expected = normalize_sentence(supplied[1])
            if expected is None or not 5 <= len(expected) - 1 <= 40:
                stats['length_or_reading_shape'] += 1
                continue
            key = core_key(text)
            if key in seen or key in protected:
                stats['duplicate_or_protected'] += 1
                continue
            segments = analyze(text, tokenizer, SplitMode.C)
            if segments is None:
                stats['unknown_proper_or_grammar'] += 1
                continue
            kana = ''.join(s['kana'] for s in segments)
            if kana + '。' != expected:
                stats['reading_disagreement'] += 1
                continue
            if len(segments) + 3 > 64:
                stats['token_length'] += 1
                continue
            seen.add(key)
            bucket = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % 100
            split = 'validation' if bucket < 3 else 'test' if bucket < 6 else 'train'
            rows.append(dict(id='tatoeba-'+sid, text=text, kana=kana, segments=segments, split=split,
                             source='Tatoeba.org and its contributors', license='CC-BY-2.0-FR',
                             source_url='https://tatoeba.org/en/sentences/show/'+sid,
                             author=author, source_modified=modified, training_group='tatoeba', trust='auto_aligned',
                             reading_provenance=dict(method='Sudachi dictionary reading agrees with supplied transcription',
                                                     transcription_reviewer=supplied[2], transcription_marked_reviewed=supplied[3])))
    # Curated everyday prose provides short, grounded tasks alongside free prose.
    for split in ('train', 'validation', 'test', 'family-test'):
        for line in Path('build/everyday-v3-data', split+'.jsonl').read_text(encoding='utf-8').splitlines():
            row = json.loads(line)
            if core_key(row['text']) in seen:
                continue
            seen.add(core_key(row['text']))
            row['split'] = split
            rows.append(row)
    counts = Counter((s['surface'], s['kana'], s['pos']) for row in rows if row['split'] == 'train' for s in row['segments'])
    keys = [key for key, _ in sorted(counts.items(), key=lambda x: (-x[1], x[0]))[:max_vocab-7]]
    specials = ['BOS', 'EOS', 'PAD', 'EASY', 'NORMAL', 'HARD', 'UNK']
    tokens = [dict(id=i, surface=s, kana='', pos='special') for i, s in enumerate(specials)]
    tokens += [dict(id=i+7, surface=s, kana=k, pos=p) for i, (s,k,p) in enumerate(keys)]
    ids = {key:i+7 for i,key in enumerate(keys)}
    parts = {s:[] for s in ('train','validation','test','family-test')}
    for row in rows:
        seq = [(s['surface'], s['kana'], s['pos']) for s in row['segments']]
        if any(t not in ids for t in seq):
            stats['vocabulary_filtered_'+row['split']] += 1
            continue
        d = 3 if len(row['kana']) <= 12 else 4 if len(row['kana']) <= 22 else 5
        row['ids'] = [0, d, *[ids[t] for t in seq], 1]
        split = row.pop('split')
        parts[split].append(row)
    out.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(dict(version='0.1.0',encoding='UTF-8',tokens=tokens),ensure_ascii=False,separators=(',',':')).encode()
    (out/'vocab.json').write_bytes(payload)
    manifest = dict(description='Tatoeba free Japanese prose plus original everyday prose',
                    source=source_info, dictionary=dict(name='SudachiDict-core', version=importlib.metadata.version('SudachiDict-core'),
                    analyzer=importlib.metadata.version('SudachiPy'), license='Apache-2.0',
                    url='https://github.com/WorksApplications/SudachiDict'),
                    splits={}, split_sha256={}, vocab_sha256=hashlib.sha256(payload).hexdigest(),
                    stats=dict(stats), source_generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    split_strategy='Normalized bodies partitioned by SHA256 before train-only vocabulary selection; older held-out bodies protected',
                    independent_human_review='pending')
    for split, data in parts.items():
        blob = ''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in data).encode()
        (out/(split+'.jsonl')).write_bytes(blob)
        manifest['splits'][split] = len(data)
        manifest['split_sha256'][split+'.jsonl'] = hashlib.sha256(blob).hexdigest()
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(splits=manifest['splits'], vocab=len(tokens), stats=stats)),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('build/tatoeba-source'))
    parser.add_argument('--out-dir', type=Path, default=Path('build/everyday-v4-data'))
    parser.add_argument('--max-vocab', type=int, default=8000)
    args = parser.parse_args()
    prepare(args.source,args.out_dir,args.max_vocab)
