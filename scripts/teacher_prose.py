"""Offline sequence distillation data, with free wording and recorded topic prompts."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import re
import time
from functools import lru_cache

TOPICS = {
    'home': '片付け、洗濯、収納、部屋の手入れ',
    'cooking': '下ごしらえ、味見、献立、調理道具',
    'food': '食感、香り、食事、飲み物',
    'work': '連絡、打ち合わせ、予定、仕事の工夫',
    'learning': '勉強、復習、質問、調べもの',
    'shopping': '買い物、品物選び、会計、包装',
    'travel': '道案内、移動、乗り物、旅先',
    'town': '街並み、駅、店、図書館',
    'nature': '海、山、木々、川辺',
    'weather': '雨、日差し、雲、風',
    'animals': '鳥、猫、犬、昆虫の観察',
    'garden': '園芸、草花、手入れ、芽',
    'music': '楽器、合唱、練習、音楽鑑賞',
    'art': '絵、写真、展示、色彩',
    'craft': '裁縫、工作、材料、道具',
    'exercise': '散歩、運動、体操、休憩',
    'social': '挨拶、相談、約束、お礼',
    'books': '読書、本選び、感想、図鑑',
    'science': '観察、実験、測定、記録',
    'senses': '触り心地、音、香り、明るさ',
    'season': '季節の変化、衣替え、行事、旬',
    'game': '将棋、囲碁、遊び、パズル',
    'care': '家族の手伝い、世話、気配り、支度',
    'making': '修理、組み立て、制作、手作り',
}
@lru_cache(maxsize=1)
def length_examples():
    rows = [json.loads(line) for line in Path('data/paired-v1/train.jsonl').read_text(encoding='utf-8').splitlines()]
    return [[r['text'] for r in rows if low <= len(r['kana']) <= high]
            for low,high in ((5,8),(13,18),(25,35))]


def prompt_for(index, seed, count=8):
    topic = list(TOPICS)[index % len(TOPICS)]
    rng = random.Random(seed + index)
    subject = rng.choice(TOPICS[topic].split('、'))
    level = index // len(TOPICS) % 3
    examples = '\n'.join(rng.sample(length_examples()[level],3))
    prompt = (f'次の見本と同じ程度の長さの日本語の短文を{count}文書いてください。\n見本：\n{examples}\n'
              f'今回の話題は「{subject}」。見本とは異なる内容にし、文の型は自由に変えてください。'
              'それぞれ独立した自然な日常文にしてください。'
              '一行に一文、句点で終え、本文だけを出力してください。番号や引用符は不要です。')
    if level == 0:
        prompt += '見本のように数語だけの、とても短い文にしてください。'
    return topic, prompt


def extract_sentences(response):
    from scripts.prepare_tatoeba import normalize_sentence
    result = []
    for line in response.splitlines():
        line = re.sub(r'^\s*(?:[-・*]|[0-9０-９]+[.．、)）])\s*', '', line.strip())
        if len(line) >= 2 and (line[0],line[-1]) in (('「','」'),('"','"')):
            line = line[1:-1]
        if line.count('。') != 1 or not line.endswith('。'):
            continue
        text = normalize_sentence(line)
        if text is not None:
            result.append(text)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--teacher', type=Path, default=Path('build/teacher/llm-jp-3.1-1.8b-instruct4'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--requests', type=int, default=288)
    parser.add_argument('--offset', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--max-new-tokens', type=int, default=256)
    parser.add_argument('--seed', type=int, default=671003)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('output exists; use a new path rather than mixing generation runs')
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if not torch.cuda.is_available():
        raise RuntimeError('offline teacher requires CUDA; no implicit CPU fallback')
    source = json.loads((args.teacher/'source.json').read_text())
    for name, info in source['files'].items():
        with (args.teacher/name).open('rb') as file:
            if hashlib.file_digest(file,'sha256').hexdigest() != info['sha256']:
                raise ValueError('teacher source changed')
    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(args.teacher, local_files_only=True, trust_remote_code=False, padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.teacher, local_files_only=True, trust_remote_code=False,
        use_safetensors=True, dtype=torch.bfloat16, attn_implementation='sdpa').to('cuda').eval()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    with args.output.open('w', encoding='utf-8') as output:
        for batch_start in range(0, args.requests, args.batch_size):
            indices = list(range(args.offset+batch_start, args.offset+min(batch_start+args.batch_size,args.requests)))
            prompts = [prompt_for(i,args.seed) for i in indices]
            rendered = [tokenizer.apply_chat_template([
                dict(role='system',content='あなたは日本語の編集者です。自然で簡潔な文を書きます。'),
                dict(role='user',content=p)],tokenize=False,add_generation_prompt=True) for _,p in prompts]
            inputs = tokenizer(rendered,padding=True,return_tensors='pt',return_token_type_ids=False).to('cuda')
            torch.manual_seed(args.seed+batch_start+args.offset)
            with torch.inference_mode():
                tokens = model.generate(**inputs,max_new_tokens=args.max_new_tokens,do_sample=True,
                    temperature=.85,top_p=.95,repetition_penalty=1.05,pad_token_id=tokenizer.pad_token_id)
            responses = tokenizer.batch_decode(tokens[:,inputs['input_ids'].shape[1]:],skip_special_tokens=True)
            for i,(topic,prompt),response in zip(indices,prompts,responses):
                output.write(json.dumps(dict(index=i,topic=topic,prompt=prompt,response=response,
                    sentences=extract_sentences(response)),ensure_ascii=False)+'\n')
            output.flush()
            print(json.dumps(dict(completed=batch_start+len(indices),seconds=time.perf_counter()-start,
                peak_cuda_bytes=torch.cuda.max_memory_allocated(),sentences=sum(len(extract_sentences(r)) for r in responses))),flush=True)
    payload = args.output.read_bytes()
    args.output.with_suffix('.manifest.json').write_text(json.dumps(dict(teacher=source,
        generation=dict(seed=args.seed,offset=args.offset,requests=args.requests,batch_size=args.batch_size,
        temperature=.85,top_p=.95,repetition_penalty=1.05,max_new_tokens=args.max_new_tokens),
        source_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        fewshot_training_sha256=hashlib.sha256(Path('data/paired-v1/train.jsonl').read_bytes()).hexdigest(),
        raw_sha256=hashlib.sha256(payload).hexdigest(),quality='synthetic; not independently reviewed',
        inference_use='none; prompts and teacher never used by deployed student'),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


if __name__ == '__main__':
    main()
