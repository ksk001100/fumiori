"""Conservative novelty proxies, not a semantic parser or a reading validator."""
from scripts.generalization import content_key

LEMMA_GROUPS = {
    '読む': ['読んだ', '読みます'], '書く': ['書いた', '書きます'],
    '描く': ['描いた', '描きます'], '飲む': ['飲んだ', '飲みます'],
    '食べる': ['食べた', '食べます'], '買う': ['買った', '買います'],
    '洗う': ['洗った', '洗います'], '畳む': ['畳んだ'],
    '開ける': ['開けた'], '閉める': ['閉めた'], '咲く': ['咲いた'],
    '鳴く': ['鳴いた'], '歌う': ['歌った'], '撮る': ['撮った', '撮ります'],
    '練習する': ['練習した'], '掃除する': ['掃除した', '掃除します'],
    '休む': ['休んだ'], '走る': ['走った'], '泳ぐ': ['泳いだ'],
    '付ける': ['付けた', '付けます'], '着る': ['着た'], '履く': ['履いた'],
    '焼く': ['焼いた'], '切る': ['切った'], '整理する': ['整理した'],
    '来る': ['来た'], '帰る': ['帰った'], '笑う': ['笑った'],
}
MANNER = ('丁寧に', 'ゆっくり', '静かに', 'きちんと', '手早く', 'じっくり',
          '元気に', '笑顔で', '安全に', '気持ちよく', '少しずつ', 'よく', '軽く', '一緒に')


def normalize_core(text):
    replacements = [(variant, lemma) for lemma, variants in LEMMA_GROUPS.items() for variant in variants]
    for variant, lemma in sorted(replacements, key=lambda pair: len(pair[0]), reverse=True):
        text = text.replace(variant, lemma)
    for phrase in MANNER:
        text = text.replace(phrase, '')
    return text


def core_key(text):
    return normalize_core(content_key(text))


def pair_key(row):
    slots = row['generation']['slots']
    names = list(slots)[:2]
    values = [normalize_core(''.join(token.split('/')[0] for token in slots[name].split())) for name in names]
    return row['generation']['family'] + ':' + '|'.join(values)
