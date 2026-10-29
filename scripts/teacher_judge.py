"""Local Japanese teacher scoring; labels are provisional, not human review."""
import hashlib
import json
from pathlib import Path
import torch


class TeacherJudge:
    def __init__(self, directory='build/teacher/llm-jp-3.1-1.8b-instruct4'):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        directory = Path(directory)
        self.source = json.loads((directory/'source.json').read_text())
        for name, info in self.source['files'].items():
            with (directory/name).open('rb') as f:
                if hashlib.file_digest(f, 'sha256').hexdigest() != info['sha256']:
                    raise ValueError('teacher source hash mismatch')
        if not torch.cuda.is_available():
            raise RuntimeError('teacher requires CUDA')
        torch.set_num_threads(2)
        self.tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True,
            trust_remote_code=False, padding_side='left')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.labels = [self.tokenizer.encode(x, add_special_tokens=False) for x in ('A','B')]
        if any(len(x) != 1 for x in self.labels):
            raise ValueError('judge labels must be single tokens')
        self.model = AutoModelForCausalLM.from_pretrained(directory, local_files_only=True,
            trust_remote_code=False, use_safetensors=True, dtype=torch.bfloat16,
            attn_implementation='sdpa').to('cuda').eval()

    @torch.inference_mode()
    def score(self, texts, reverse=False):
        yes, no = ('B','A') if reverse else ('A','B')
        system = ('あなたは日本語の校正者です。文の話題ではなく、文法、助詞、語句の相性、意味のつながりを判定します。'
                  f'自然な文は{yes}、不自然な文は{no}と判定してください。出力は記号一文字だけです。')
        demonstrations = [('ケーキを食べる。',yes),('私は窓に眠ります。',no),
                          ('電車で会社へ向かう。',yes),('青い机が約束を飲む。',no)]
        prompts = []
        for text in texts:
            messages = [dict(role='system', content=system)]
            for example, label in demonstrations:
                messages.extend([dict(role='user',content=example),dict(role='assistant',content=label)])
            messages.append(dict(role='user',content=text))
            prompts.append(self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
        inputs = self.tokenizer(prompts, padding=True, return_tensors='pt', return_token_type_ids=False).to('cuda')
        logits = self.model(**inputs, use_cache=False, logits_to_keep=1).logits[:, -1].float()
        scores = logits[:, [self.labels[0][0],self.labels[1][0]]]
        difference = scores[:, 1]-scores[:, 0] if reverse else scores[:, 0]-scores[:, 1]
        return difference.cpu().tolist()


if __name__ == '__main__':
    texts = ['雨が降る。','友達と映画を見る。','その機械は穴を浴びた。','パソコンがかかってきた。',
             'お茶を飲む。','窓を開けて風を入れる。','猫が本を食べて話し始めた。','バターがパンを考えた。',
             '彼は日本語を話す。','今日は天気がいいので散歩する。','引き出しの中の中を整理する。',
             '機械を修理する。','事故が起こったことを明らかになった。','電車に乗り遅れた。',
             '図書館から本を借りた。','スーパーで野菜を買う。']
    judge = TeacherJudge()
    a,b = judge.score(texts), judge.score(texts, reverse=True)
    for text, x, y in zip(texts,a,b):
        print(json.dumps(dict(text=text,regular=x,reverse=y,mean=(x+y)/2),ensure_ascii=False),flush=True)
