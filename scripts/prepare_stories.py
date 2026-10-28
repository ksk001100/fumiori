"""Align short upstream prose with explicit reading lexemes; reject unknown kanji.

Kana character tokens carry a literal reading and POS='文字', not a guessed
word-level POS. No dictionary/reading is invented for an unknown kanji.
"""
from collections import Counter, defaultdict
import argparse
import hashlib
import json
from pathlib import Path
import re
import unicodedata
import pyarrow.parquet as pq

from scripts.generalization import evaluation_text
from scripts.semantic_keys import core_key


ICHIDAN = ('食べる', '見る', '寝る', '起きる', '借りる', '開ける', '閉める', '入れる',
    '出る', '見える', '聞こえる', '冷める', '混ぜる', '比べる', '届ける', '忘れる',
    '始める', '乗せる', '載せる', '並べる', '続ける', '変える', '集める', '建てる',
    '広げる', '付ける', '片付ける', '見つける', '考える', '加える', '終える', '調べる',
    '伝える', '確かめる', '決める', '覚える', '勧める', '教える', '眺める', '迎える', '見せる', 'かける',
    'いる', 'できる', '出来る', '生きる', '浴びる', '降りる', '足りる', '落ちる', '感じる',
    '信じる', '閉じる', '消える', '増える', '燃える', '覚める')
GODAN = {'う': ('い','わ','って','った'), 'く': ('き','か','いて','いた'),
    'ぐ': ('ぎ','が','いで','いだ'), 'す': ('し','さ','して','した'),
    'つ': ('ち','た','って','った'), 'ぬ': ('に','な','んで','んだ'),
    'ぶ': ('び','ば','んで','んだ'), 'む': ('み','ま','んで','んだ'),
    'る': ('り','ら','って','った')}


def lexicon(base):
    entries = {(t['surface'], t['kana'], t['pos']): dict(t) for t in base['tokens'][7:]}
    def add(surface, kana, pos, origin):
        entries.setdefault((surface, kana, pos), dict(surface=surface, kana=kana, pos=pos,
            metadata=dict(origin=origin, contextual_reading='not independently verified')))
    for token in base['tokens'][7:]:
        s, k, p = (token[x] for x in ('surface','kana','pos'))
        if p != '動詞':
            continue
        if s.endswith('する') and k.endswith('する'):
            stem_s, stem_k = s[:-2], k[:-2]
            forms = [('した','した'),('して','して'),('します','します'),('しました','しました'),
                ('しない','しない'),('している','している'),('していた','していた'),('していました','していました')]
        elif s.endswith('くる') and k.endswith('くる'):
            stem_s, stem_k = s[:-2], k[:-2]
            forms = [(x,x) for x in ('きた','きて','きます','きました','こない','きている','きていた','きていました')]
        elif s.endswith('来る') and k.endswith('くる'):
            stem_s, stem_k = s[:-2], k[:-2]
            forms = [('来た','きた'),('来て','きて'),('来ます','きます'),('来ました','きました')]
        elif s.endswith(ICHIDAN) and k.endswith('る'):
            stem_s, stem_k = s[:-1], k[:-1]
            forms = [(x,x) for x in ('た','て','ます','ました','ない','ている','ていた','ていました')]
        elif s[-1:] in GODAN and k.endswith(s[-1]):
            stem_s, stem_k = s[:-1], k[:-1]
            ren, neg, te, past = GODAN[s[-1]]
            if s.endswith('行く'):
                te, past = 'って','った'
            forms = [(x,x) for x in (ren+'ます',ren+'ました',neg+'ない',te,past,te+'いる',te+'いた',te+'いました')]
        else:
            continue
        for suffix_s, suffix_k in forms:
            add(stem_s + suffix_s, stem_k + suffix_k, '動詞', 'explicit-conjugation')
    common = {
        '彼':'かれ','彼女':'かのじょ','私':'わたし','少年':'しょうねん','少女':'しょうじょ',
        '男の子':'おとこのこ','女の子':'おんなのこ','子供':'こども','子ども':'こども',
        '村人':'むらびと','人々':'ひとびと','自分':'じぶん','みんな':'みんな','皆':'みな',
        '毎日':'まいにち','一日':'いちにち','ある日':'あるひ','今日':'きょう','明日':'あした',
        '気持ち':'きもち','気分':'きぶん','心':'こころ','夢':'ゆめ','声':'こえ','名前':'なまえ',
        '一緒':'いっしょ','幸せ':'しあわせ','嬉しい':'うれしい','嬉しかった':'うれしかった',
        '楽しい':'たのしい','楽しかった':'たのしかった','小さな':'ちいさな','大きな':'おおきな',
        '風船':'ふうせん','太陽':'たいよう','夕焼け':'ゆうやけ','朝焼け':'あさやけ',
        '笑顔':'えがお','貝殻':'かいがら','光る':'ひかる','光った':'ひかった',
        '光って':'ひかって','光ります':'ひかります','光りました':'ひかりました',
    }
    for s, k in common.items():
        pos = '動詞' if s.startswith('光') else '形容詞' if s.endswith(('しい','かった')) else '名詞'
        add(s,k,pos,'explicit-reading')
    for s in ('だ','です','でした','だった','ます','ました','ない','なかった','でしたね'):
        add(s,s,'助動詞','explicit-function-word')
    for s in ('には','では','とは','からは','へと','のに','ので','けれど','けれども','ても','でも','までに','だけで','たり'):
        add(s,s,'助詞','explicit-function-word')
    add('、','','記号','punctuation')
    for code in list(range(0x3041,0x3097)) + list(range(0x30a1,0x30f7)) + [0x30fc]:
        s = chr(code); k = chr(code-0x60) if 0x30a1 <= code <= 0x30f6 else s
        add(s,k,'文字','literal-kana-character; not a morphological word')
    tokens = [dict(t) for t in base['tokens'][:7]]
    for key in sorted(entries):
        token = dict(entries[key]); token['id'] = len(tokens); tokens.append(token)
    return dict(version=base['version'], encoding=base['encoding'], tokens=tokens)


class Aligner:
    def __init__(self, vocab):
        self.by_first = defaultdict(list)
        for token in vocab['tokens'][7:]:
            self.by_first[token['surface'][0]].append(token)
        for group in self.by_first.values():
            group.sort(key=lambda t: (-len(t['surface']), t['pos']=='文字', t['kana'], t['id']))

    def align(self, text):
        tokens, position = [], 0
        while position < len(text):
            token = next((t for t in self.by_first[text[position]] if text.startswith(t['surface'],position)), None)
            if token is None:
                return None
            # Do not teach prompts to spell names or unknown words using kana
            # characters. Keep literal tokens in the vocab for ID compatibility.
            if token['pos'] == '文字' or any(p in ('人名', '固有名詞') for p in token['pos'].split('-')):
                return None
            if tokens and lexical_collision(tokens[-1], token):
                return None
            tokens.append(token); position += len(token['surface'])
        if len(tokens) < 2 or tokens[-1]['surface'] != '。' or tokens[-2]['pos'] not in ('動詞','形容詞','助動詞'):
            return None
        if tokens[-2]['pos'] == '動詞' and tokens[-2]['surface'].endswith(('て','で','に','ず')):
            return None
        if any(a['pos']==b['pos']=='助詞' for a,b in zip(tokens,tokens[1:])):
            return None
        kana = ''.join(t['kana'] for t in tokens)
        if not 5 <= len(kana) <= 40 or len(tokens)+3 > 64:
            return None
        segments = [{k:t[k] for k in ('surface','kana','pos')} for t in tokens]
        difficulty = 3 if len(kana)<=12 else 4 if len(kana)<=22 else 5
        return dict(text=text,kana=kana,segments=segments,ids=[0,difficulty,*[t['id'] for t in tokens],1],
            trust='auto_aligned',source='SimpleStories/SimpleStories-JA',license='MIT')


def lexical_collision(previous, token):
    kanji = lambda c: '\u4e00'<=c<='\u9fff' or c=='々'
    return (kanji(previous['surface'][-1]) and previous['pos']=='名詞' and
            ((kanji(token['surface'][0]) and token['pos']=='名詞') or token['pos']=='文字'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir',type=Path,default=Path('build/stories-v1-data'))
    parser.add_argument('--train-limit',type=int,default=100000)
    args = parser.parse_args()
    source = Path('build/simple-stories')
    out = args.out_dir; out.mkdir(parents=True,exist_ok=True)
    base_dir = Path('data/paired-v1')
    base_manifest = json.loads((base_dir/'manifest.json').read_text(encoding='utf-8'))
    for name, expected in {**base_manifest['split_sha256'], 'vocab.json': base_manifest['vocab_sha256']}.items():
        if hashlib.sha256((base_dir/name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'paired v1 input changed: {name}')
    vocab = lexicon(json.loads((base_dir/'vocab.json').read_text(encoding='utf-8')))
    aligner = Aligner(vocab)
    ids = {(t['surface'],t['kana'],t['pos']):t['id'] for t in vocab['tokens'][7:]}
    parts = {s:[] for s in ('train','validation','test','family-test')}
    blocked = set()
    for split in parts:
        for line in (base_dir/f'{split}.jsonl').read_text(encoding='utf-8').splitlines():
            row = json.loads(line); blocked.add(core_key(row['text']))
            tokens = [ids[(s['surface'],s['kana'],s['pos'])] for s in row['segments']]
            row['ids'] = [0,row['ids'][1],*tokens,1]
            parts[split].append(row)
    initial_counts = {s:len(rows) for s,rows in parts.items()}
    seen, stats, story_count = set(blocked), Counter(), 0
    parquet = pq.ParquetFile(source/'train-00000.parquet')
    for batch in parquet.iter_batches(batch_size=1024,columns=['story']):
        for story in batch.column(0).to_pylist():
            story_count += 1
            n = int(hashlib.sha256(story.encode()).hexdigest()[:8],16)%20
            split = 'validation' if n==0 else 'test' if n==1 else 'train'
            for sentence in re.split(r'[。！？!?\n]',story):
                text = unicodedata.normalize('NFC',sentence.strip())+'。'
                if len(text)>80 or any(c in text for c in '「」『』()（）【】'):
                    stats['unsupported_shape'] += 1; continue
                if text.startswith(('すると','そこで','その後','それから','それは','その中','そこに')) or re.match(r'^.{1,8}に、ある(?:ところ|日)',text):
                    stats['context_or_seed_artifact'] += 1; continue
                if re.match(r'^[ぁ-ゖー]{2,12}(?:に|と)、.*(?:が|に)(?:ありました|いました|あった|いた)。$',text):
                    stats['context_or_seed_artifact'] += 1; continue
                key = core_key(text)
                if key in seen:
                    stats['duplicate_or_protected'] += 1; continue
                row = aligner.align(text)
                if row is None:
                    stats['unalignable'] += 1; continue
                if split!='train' and len(parts[split])>=initial_counts[split]+500:
                    continue
                seen.add(key)
                row['id'] = 'simplestories-'+hashlib.sha256(text.encode()).hexdigest()[:20]
                row['upstream_story_sha256'] = hashlib.sha256(story.encode()).hexdigest()
                parts[split].append(row)
            if len(parts['train'])>=args.train_limit:
                break
        if len(parts['train'])>=args.train_limit:
            break
    payload = json.dumps(vocab,ensure_ascii=False,separators=(',',':')).encode()
    (out/'vocab.json').write_bytes(payload)
    manifest = dict(split_sha256={},splits={},vocab_sha256=hashlib.sha256(payload).hexdigest(),
        source=json.loads((source/'source.json').read_text()),initial_counts=initial_counts,
        alignment='longest explicit lexeme; unknown kanji rejected; literal kana chars tagged as characters, not guessed POS',
        split_strategy='protected paired prose splits plus upstream train stories partitioned by story SHA256; duplicate normalized texts removed',
        stats=dict(stats),stories_scanned=story_count,independent_human_review='pending',
        requested_train_limit=args.train_limit,
        source_exhausted_before_limit=len(parts['train']) < args.train_limit,
        source_generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    for name,rows in parts.items():
        data=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows).encode()
        (out/f'{name}.jsonl').write_bytes(data)
        manifest['split_sha256'][f'{name}.jsonl']=hashlib.sha256(data).hexdigest(); manifest['splits'][name]=len(rows)
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(splits=manifest['splits'],tokens=len(vocab['tokens']),stories=story_count,stats=stats)),flush=True)


if __name__=='__main__':
    main()
