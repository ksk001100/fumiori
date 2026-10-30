"""Expand a frozen word corpus from recorded local prose, with exact near filtering."""
import argparse
from collections import Counter, defaultdict
import copy
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import time

from scripts.audit_combinations import load_pairs
from scripts.audit_data_diversity import lexical_sequence
from scripts.curate_training_data import style_key
from scripts.novelty_evaluate import sha, write_json
from scripts.novelty_metrics import normalized
from scripts.prepare_combination_holdout import prepare as prepare_holdout
from scripts.prepare_tatoeba import normalize_sentence
from scripts.prepare_word_teacher import canonical, encode_candidate, vocabulary_index, report_texts


class NearFilter:
    """Exact Dice threshold join using globally ordered Jaccard prefixes.

    Dice >= d iff Jaccard >= d/(2-d). Qualifying sets must share their
    |set|-ceil(t*|set|)+1 prefixes. Order is frozen from reference frequencies.
    This is a surface filter, not a semantic similarity model.
    """
    def __init__(self, texts, threshold=.85):
        if not 0 < threshold <= 1:
            raise ValueError('threshold outside (0,1]')
        self.threshold = threshold
        self.ratio = threshold/(2-threshold)
        self.frequency = Counter(g for text in texts for g in self.grams(text))
        self.sets, self.postings = [], defaultdict(list)
        self.exact = set()
        for text in texts:
            self.add(text)

    @staticmethod
    def grams(text):
        text = normalized(text)
        return frozenset(zip(text,text[1:]))

    def prefix(self, grams):
        size = len(grams)-math.ceil(self.ratio*len(grams)-1e-12)+1
        return sorted(grams,key=lambda g:(self.frequency[g],g))[:size]

    def contains_near(self, text):
        text = normalized(text)
        if text in self.exact:
            return True
        grams = self.grams(text)
        candidates = set()
        for gram in self.prefix(grams):
            candidates.update(self.postings.get(gram,()))
        for i in candidates:
            other = self.sets[i]
            if not self.ratio*len(grams)-1e-12 <= len(other) <= len(grams)/self.ratio+1e-12:
                continue
            if 2*len(grams & other) >= self.threshold*(len(grams)+len(other))-1e-12:
                return True
        return False

    def add(self, text):
        text = normalized(text)
        if text in self.exact:
            return
        grams = self.grams(text)
        i = len(self.sets)
        self.sets.append(grams)
        self.exact.add(text)
        for gram in self.prefix(grams):
            self.postings[gram].append(i)


def standalone(text):
    if len(text)>60 or any(c in text for c in '「」『』()（）【】'):
        return False
    if text.startswith(('すると','そこで','その後','それから','それは','その中','そこに',
        'しかし','だから','そして','でも、','そのため','こうして','それを','それに','それが',
        'それ以来','そうして','そのとき','その時','さらに','途中で','いきなり','昔々','むかしむかし','やがて')):
        return False
    if re.match(r'^[ぁ-ゖー]{2,12}(?:に|と)、.*(?:が|に)(?:ありました|いました|あった|いた)。$',text):
        return False
    if text.startswith('ほぼ毎日') and text.endswith(('いました。','ありました。')):
        return False
    if re.search(r'(?:森|海|山|町|村|部屋|中|前|そば|近く|学校|公園)で、?[^、。]{1,24}が(?:いました|いた|います|いる|ありました|あった|あります|ある)。$',text):
        return False
    return 'ぎりぎりの場所' not in text


def reading_issue(row):
    # 宇宙+人 may be TWO segments; a whole-surface token check misses it.
    if '宇宙人' in row['text'] and 'うちゅうにん' in row['kana']:
        return True
    if '何より' in row['text'] and 'なんより' in row['kana']:
        return True
    for i,s in enumerate(row['segments']):
        if s['surface']=='実' and s['kana']=='じつ':
            tail=''.join(t['surface'] for t in row['segments'][i:])
            if not tail.startswith(('実は','実に','実を言','実のところ')):
                return True  # Uncertain standalone fruit/truth reading: quarantine.
    return False


def prepare(config_path):
    import pyarrow.parquet as pq
    from sudachipy import Dictionary, SplitMode
    config=json.loads(Path(config_path).read_text(encoding='utf-8'))
    base,source,merged,output=(Path(config[k]) for k in ('base','source','merged','output'))
    if merged.exists() or output.exists():
        raise ValueError('new directories required')
    manifest=json.loads((base/'manifest.json').read_text(encoding='utf-8'))
    for name,digest in manifest['split_sha256'].items():
        if sha(base/name)!=digest:raise ValueError('base split changed')
    if sha(base/'vocab.json')!=manifest['vocab_sha256']:raise ValueError('base vocab changed')
    info=json.loads((source/'source.json').read_text(encoding='utf-8'))
    parquet=source/'train-00000.parquet'
    if sha(parquet)!=info['sha256'] or info['downloaded_split']!='train':
        raise ValueError('source identity changed or not upstream TRAIN')
    parts={name:[json.loads(s) for s in (base/name).read_text(encoding='utf-8').splitlines()] for name in manifest['split_sha256']}
    parent_count=len(parts['train.jsonl'])
    needed=config['target_train']-parent_count
    if needed <= 0:raise ValueError('target must expand train')
    tokenizer,mode=Dictionary().tokenizer(),SplitMode.C
    vocab=vocabulary_index(json.loads((base/'vocab.json').read_text(encoding='utf-8')))
    cases=load_pairs(config['pairs'],tokenizer,mode)
    targets={tuple(c[k]) for c in cases for k in ('positive_key','negative_key')}
    reference=[r['text'] for rows in parts.values() for r in rows]
    protection=[]
    for filename in config['protect_jsonl']:
        p=Path(filename);reference.extend(json.loads(s)['text'] for s in p.read_text(encoding='utf-8').splitlines())
        protection.append(dict(path=filename,sha256=sha(p)))
    for filename in config['protect_reports']:
        p=Path(filename);reference.extend(report_texts(json.loads(p.read_text(encoding='utf-8'))))
        protection.append(dict(path=filename,sha256=sha(p)))
    reference=list(dict.fromkeys(reference))
    protected={canonical(t) for t in reference}
    styles={style_key(t,tokenizer,mode) for t in reference}
    near=NearFilter(reference,config['near_threshold'])
    stats,topics,patterns,starts,levels=Counter(),Counter(),Counter(),Counter(),Counter()
    train,test=[],[]
    seen=set()
    start=time.perf_counter()
    table=pq.ParquetFile(parquet)
    columns=['story','topic','theme','style']
    for batch in table.iter_batches(batch_size=1024,columns=columns):
        for record in batch.to_pylist():
            stats['stories_scanned']+=1
            story=record['story'];story_sha=hashlib.sha256(story.encode()).hexdigest()
            split='stories-test' if int(story_sha[:8],16)%10==0 else 'train'
            if (split=='train' and len(train)>=needed) or (split=='stories-test' and len(test)>=config['new_test']):continue
            for sentence in re.split(r'[。！？!?\n]',story):
                stats['raw_fragments']+=1
                text=normalize_sentence(sentence)
                if text is None or not standalone(text):stats['shape_or_context']+=1;continue
                if text in seen:stats['repeated_candidate']+=1;continue
                seen.add(text)
                if any(term in text for term in config['quarantine_terms']):stats['topic_quarantine']+=1;continue
                row,reason=encode_candidate(text,vocab,tokenizer,mode,protected,targets)
                if reason:stats[reason]+=1;continue
                if reading_issue(row):stats['reviewed_context_reading']+=1;continue
                if any(s['surface'] in config.get('exclude_surfaces',[]) for s in row['segments']):
                    stats['reviewed_reading_surface']+=1;continue
                pattern=(row['ids'][1],lexical_sequence(row,True))
                beginning=''.join(s['surface'] for s in row['segments'][:2])
                if split=='train' and (patterns[pattern]>=config['pattern_limit'] or topics[record['topic']]>=config['topic_limit'] or starts[beginning]>=config['start_limit']):
                    stats['diversity_cap']+=1;continue
                key=style_key(text,tokenizer,mode)
                if key in styles:stats['style_duplicate']+=1;continue
                if near.contains_near(text):stats['near_duplicate']+=1;continue
                near.add(text);styles.add(key);protected.add(canonical(text))
                row.update(training_group='stories',id='v20-'+hashlib.sha256(text.encode()).hexdigest()[:20],
                    source=dict(dataset=info['dataset'],revision=info['revision'],parquet_sha256=info['sha256'],story_sha256=story_sha),
                    license=info['license'],generation=dict(requested_topic=record['topic'],theme=record['theme'],style=record['style'],topic_review='metadata only; actual topic/tone not independently verified'))
                (train if split=='train' else test).append(row)
                if split=='train':topics[record['topic']]+=1;patterns[pattern]+=1;starts[beginning]+=1;levels[row['ids'][1]]+=1
                if (split=='train' and len(train)>=needed) or (split=='stories-test' and len(test)>=config['new_test']):break
            if stats['stories_scanned']%10000==0:
                print(json.dumps(dict(stories=stats['stories_scanned'],train=len(train),test=len(test),seconds=time.perf_counter()-start)),flush=True)
            if len(train)>=needed and len(test)>=config['new_test']:break
            if config.get('max_stories') and stats['stories_scanned']>=config['max_stories']:break
        if (len(train)>=needed and len(test)>=config['new_test']) or (config.get('max_stories') and stats['stories_scanned']>=config['max_stories']):break
    report=dict(config_sha256=sha(config_path),script_sha256=sha(__file__),source=info,stats=dict(stats),added_train=len(train),added_test=len(test),topics=dict(topics),difficulty=dict(levels),
        largest_pattern=max(patterns.values(),default=0),largest_start=max(starts.values(),default=0),protection=protection,seconds=time.perf_counter()-start,
        upstream_stories=table.metadata.num_rows,source_exhausted=stats['stories_scanned']>=table.metadata.num_rows,
        target_unfilled=len(train)<needed or len(test)<config['new_test'])
    merged.mkdir(parents=True)
    write_json(merged/'expansion.json',report)
    # Preserve a useful candidate snapshot if the strict gate cannot fill 100k.
    for name,rows in [('candidate-train.jsonl',train),('candidate-test.jsonl',test)]:
        (merged/name).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8',newline='\n')
    if report['target_unfilled']:return report
    shutil.copyfile(base/'vocab.json',merged/'vocab.json')
    m=copy.deepcopy(manifest);m.pop('combination_holdout');m.pop('data_curation',None)
    for name in parts:
        if name=='train.jsonl':
            (merged/name).write_bytes((base/name).read_bytes()+''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in train).encode())
            m['splits']['train']=config['target_train'];m['split_sha256'][name]=sha(merged/name)
        else:shutil.copyfile(base/name,merged/name)
    shutil.copyfile(merged/'candidate-test.jsonl',merged/'stories-test.jsonl')
    m['splits']['stories-test']=len(test);m['split_sha256']['stories-test.jsonl']=sha(merged/'stories-test.jsonl')
    notice=(Path('LICENSE').read_text(encoding='utf-8').replace('Copyright (c) 2026 Fumiori contributors','Copyright (c) The SimpleStories Project'))
    m['additional_sources']=[dict(info,license_text=notice,license_text_sha256=hashlib.sha256(notice.encode()).hexdigest())]
    m['diverse_expansion']=dict(parent_manifest_sha256=sha(base/'manifest.json'),report_sha256=sha(merged/'expansion.json'),
        policy='Original rows plus near/style/pattern-capped prose; fixed word IDs; whole-story heldout; metadata topic quotas do not certify actual semantics')
    write_json(merged/'manifest.json',m)
    final=prepare_holdout(merged,config['pairs'],output)
    if final['splits']['train']!=config['target_train']:raise ValueError('unexpected extra holdout removal')
    shutil.copyfile(merged/'expansion.json',output/'expansion.json')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    args=parser.parse_args()
    print(json.dumps(prepare(args.config),ensure_ascii=False),flush=True)
