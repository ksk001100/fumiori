"""Text normalization shared by v1 data preparation and evaluation."""
from functools import lru_cache


@lru_cache(maxsize=1)
def temporal_prefixes():
    return ['よく晴れた日の朝に', '休みの日の午後に', '休日の午後に', '休みの日に', '晴れた夜に', 'ようやく', '寝る前に', '昼休みに', '午前中に', '午後に', '休日に', '夕方に', '夕方', '夜に', '昨日', '今朝', '朝に', '今日', '朝']


def content_key(text):
    for prefix in temporal_prefixes():
        if text.startswith(prefix):
            return text[len(prefix):]
    return text


def evaluation_text(text):
    return text if text.endswith('。') else text + '。'
