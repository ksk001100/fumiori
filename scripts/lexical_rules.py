"""Boundary rule shared by data preparation; no dictionary download required."""


def lexical_collision(previous, token):
    kanji = lambda c: '\u4e00' <= c <= '\u9fff' or c == '々'
    return (kanji(previous['surface'][-1]) and previous['pos'] == '名詞' and
            ((kanji(token['surface'][0]) and token['pos'] == '名詞') or token['pos'] == '文字'))
