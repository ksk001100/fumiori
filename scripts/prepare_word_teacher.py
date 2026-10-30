"""Expand a frozen paired-word corpus using local teacher outputs; offline only."""
import argparse
from collections import Counter, defaultdict
import copy
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import unicodedata

from scripts.audit_combinations import lexical_triples, load_pairs, sha
from scripts.prepare_combination_holdout import prepare as prepare_holdout
from scripts.prepare_tatoeba import analyze
from scripts.prepare_teacher_prose import orthographic_readings
from scripts.lexical_rules import lexical_collision
from scripts.generalization import evaluation_text
from scripts.semantic_keys import core_key


def canonical(text):
    return core_key(evaluation_text(unicodedata.normalize('NFKC',text)))


def report_texts(value):
    if isinstance(value,dict):
        if isinstance(value.get('text'),str):
            yield value['text']
        for item in value.values():
            yield from report_texts(item)
    elif isinstance(value,list):
        for item in value:
            yield from report_texts(item)


def encode_candidate(text,vocab,tokenizer,mode,protected,targets):
    if canonical(text) in protected:
        return None,'duplicate_or_protected'
    if text.startswith(('以下の','次の文','こちらが','以上が')) or '例文' in text or '文字以内' in text:
        return None,'instruction_artifact'
    segments=analyze(text,tokenizer,mode)
    if segments is None or any(lexical_collision(a,b) for a,b in zip(segments,segments[1:])):
        return None,'dictionary_or_boundary'
    segments=orthographic_readings(segments)
    mapping,ambiguous=vocab
    if any((s['surface'],s['pos']) in ambiguous for s in segments):
        return None,'ambiguous_lexeme_reading'
    if any((s['surface'],s['kana'],s['pos']) not in mapping for s in segments):
        return None,'out_of_vocabulary'
    kana=''.join(s['kana'] for s in segments)
    if not 5<=len(kana)<=40:
        return None,'kana_length'
    if lexical_triples(text,tokenizer,mode)&targets:
        return None,'protected_combination'
    level=3 if len(kana)<=12 else 4 if len(kana)<=22 else 5
    ids=[0,level,*[mapping[(s['surface'],s['kana'],s['pos'])] for s in segments],1]
    if len(ids)>64:
        return None,'sequence_length'
    return dict(text=text,kana=kana,segments=segments,ids=ids,trust='auto_aligned',training_group='teacher',
        reading_validation='Dictionary agreement with frozen word IDs; ambiguous lexemes rejected; contextual reading not independently certified'),None


def vocabulary_index(vocab):
    readings=defaultdict(set)
    mapping={}
    for t in vocab['tokens'][7:]:
        mapping[t['surface'],t['kana'],t['pos']]=t['id']
        readings[t['surface'],t['pos']].add(t['kana'])
    return mapping,{key for key,values in readings.items() if len(values)>1}


def prepare(config_path):
    from sudachipy import Dictionary,SplitMode
    config=json.loads(Path(config_path).read_text(encoding='utf-8'))
    base=Path(config['base']);manifest=json.loads((base/'manifest.json').read_text(encoding='utf-8'))
    if sha(base/'vocab.json')!=manifest['vocab_sha256']:
        raise ValueError('base vocabulary changed')
    for filename,expected in manifest['split_sha256'].items():
        if sha(base/filename)!=expected:raise ValueError('base split changed')
    if 'combination_holdout' not in manifest or sha(base/'holdout.json')!=manifest['combination_holdout']['report_sha256']:
        raise ValueError('verified combination holdout required')
    if importlib.metadata.version('SudachiDict-core')!=manifest['dictionary']['version']:
        raise ValueError('dictionary version differs from the frozen corpus')
    for variant in config['variants']:
        if Path(variant['merged']).exists() or Path(variant['output']).exists():
            raise ValueError('all variant directories must be new')
    parts={name:[json.loads(line) for line in (base/name).read_text(encoding='utf-8').splitlines()] for name in manifest['split_sha256']}
    protected={canonical(r['text']) for rows in parts.values() for r in rows}
    protection=[]
    for name in config['protect_reports']:
        path=Path(name);report=json.loads(path.read_text(encoding='utf-8'))
        protected.update(canonical(t) for t in report_texts(report))
        protection.append(dict(path=name,sha256=sha(path)))
    parent_lines=set((base/'train.jsonl').read_text(encoding='utf-8').splitlines())
    tokenizer,mode=Dictionary().tokenizer(),SplitMode.C
    cases=load_pairs(config['pairs'],tokenizer,mode)
    targets={tuple(c[field]) for c in cases for field in ('positive_key','negative_key')}
    vocab=vocabulary_index(json.loads((base/'vocab.json').read_text(encoding='utf-8')))
    additions=[];stats=Counter();sources=[]
    for name in config['raw']:
        path=Path(name);info=json.loads(path.with_suffix('.manifest.json').read_text(encoding='utf-8'))
        if sha(path)!=info['raw_sha256']:raise ValueError('raw teacher output changed')
        examples=Path(info['fewshot_training_path'])
        if sha(examples)!=info['fewshot_training_sha256'] or not set(examples.read_text(encoding='utf-8').splitlines())<=parent_lines:
            raise ValueError('teacher examples must be unchanged rows from the filtered train')
        sources.append(info)
        for line in path.read_text(encoding='utf-8').splitlines():
            record=json.loads(line)
            for text in record['sentences']:
                stats['raw_sentences']+=1
                row,reason=encode_candidate(text,vocab,tokenizer,mode,protected,targets)
                if reason:
                    stats[reason]+=1;continue
                protected.add(canonical(text))
                row.update(generation=dict(requested_topic=record['topic'],topic_review='pending',expansion='v17'),
                    source=dict(teacher_model=info['teacher']['model'],revision=info['teacher']['revision'],raw_file_sha256=info['raw_sha256'],request_index=record['index']))
                # A deterministic unseen teacher subset stays outside all student training.
                bucket=int(hashlib.sha256(canonical(text).encode()).hexdigest()[:8],16)%10
                additions.append(('teacher-test' if bucket==0 else 'train',row))
    training=sorted((r for s,r in additions if s=='train'),key=lambda r:hashlib.sha256(r['text'].encode()).hexdigest())
    heldout=[r for s,r in additions if s=='teacher-test']
    if len(training)<config['minimum_new_train']:
        raise ValueError(f'insufficient new rows: {len(training)}; {dict(stats)}')
    outputs=[]
    for variant in config['variants']:
        selected=training[:variant['train_limit']] if variant.get('train_limit') is not None else training
        merged=Path(variant['merged']);merged.mkdir(parents=True)
        shutil.copyfile(base/'vocab.json',merged/'vocab.json')
        merged_manifest=copy.deepcopy(manifest)
        merged_manifest.pop('combination_holdout')
        merged_manifest['description']='Intermediate teacher expansion; use the separately audited final holdout directory for training'
        merged_manifest['augmentation_parent_manifest_sha256']=sha(base/'manifest.json')
        merged_manifest['synthetic_sources']=[*manifest.get('synthetic_sources',[]),*sources]
        merged_manifest['teacher_expansion']=dict(script_sha256=sha(__file__),config_sha256=sha(config_path),
            new_train=len(selected),new_teacher_test=len(heldout),candidate_new_train=len(training),stats=dict(stats),
            protected_reports=protection,reading_policy='Reject existing surface/POS with multiple readings, OOV and lexical boundary collisions; dictionary agreement is not contextual certification')
        for filename,rows in parts.items():
            # Preserve the original validation/test bytes for a common selection criterion.
            if filename!='train.jsonl':
                shutil.copyfile(base/filename,merged/filename);continue
            payload=(base/filename).read_bytes()+''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in selected).encode()
            (merged/filename).write_bytes(payload)
            merged_manifest['splits']['train']=len(rows)+len(selected)
            merged_manifest['split_sha256'][filename]=sha(merged/filename)
        teacher_path=merged/'teacher-test.jsonl'
        teacher_path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in heldout),encoding='utf-8')
        merged_manifest['splits']['teacher-test']=len(heldout)
        merged_manifest['split_sha256']['teacher-test.jsonl']=sha(teacher_path)
        (merged/'manifest.json').write_text(json.dumps(merged_manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        final=prepare_holdout(merged,config['pairs'],variant['output'])
        if final['splits']['train']!=merged_manifest['splits']['train']:
            raise ValueError('unexpected combination leak after initial filtering')
        outputs.append(dict(output=variant['output'],new_train=len(selected),manifest_sha256=sha(Path(variant['output'])/'manifest.json')))
    return dict(stats=dict(stats),candidate_train=len(training),teacher_test=len(heldout),outputs=outputs)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--report',required=True)
    args=parser.parse_args()
    if Path(args.report).exists():raise ValueError('report must be new')
    result=prepare(args.config)
    Path(args.report).write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))
