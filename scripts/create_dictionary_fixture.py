import json
from pathlib import Path
p=Path('data/examples/lindera-src');p.mkdir(parents=True,exist_ok=True)
rows=[json.loads(s) for s in Path('data/examples/sentences.jsonl').read_text(encoding='utf-8').splitlines()]
tokens=sorted({(s['surface'],s['kana'],s['pos']) for r in rows for s in r['segments']})
def katakana(s): return ''.join(chr(ord(c)+0x60) if '\u3041'<=c<='\u3096' else c for c in s)
(p/'lexicon.csv').write_text(''.join(','.join([a,'0','0','0',pos,'*','*','*','*','*',a,katakana(k) or a,katakana(k) or a])+'\n' for a,k,pos in tokens),encoding='utf-8')
(p/'char.def').write_text('DEFAULT 0 1 0\nSPACE 0 1 0\n0x0020 SPACE\n',encoding='utf-8')
(p/'unk.def').write_text('DEFAULT,0,0,10000,unknown,*,*,*,*,*,*\nSPACE,0,0,10000,unknown,*,*,*,*,*,*\n',encoding='utf-8')
(p/'matrix.def').write_text('1 1\n0 0 0\n',encoding='utf-8')
# No segments: exercise real Lindera matching plus rejection of wrong known kana.
auto=[]
for i in [0,6,7,14]:
 r=dict(rows[i]);r.pop('segments');r['id']='analyze-'+str(i);auto.append(r)
r=dict(auto[0]);r['id']='intentional-reading-mismatch';r['kana']='いぬがねむる';auto.append(r)
Path('data/examples/analysis-cases.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in auto),encoding='utf-8')
