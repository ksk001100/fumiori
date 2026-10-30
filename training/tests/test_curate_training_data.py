import pytest

from scripts.curate_training_data import cap_group, coverage, style_key


@pytest.fixture
def key():
    from sudachipy import Dictionary, SplitMode
    tokenizer=Dictionary().tokenizer()
    return lambda text:style_key(text,tokenizer,SplitMode.C)


def test_style_dedup_keeps_tense_negation_voice_and_mood(key):
    assert key('本を読んだ。')==key('本を、読みました。')
    assert key('これは本だった。')==key('これは本でした。')
    assert key('本を読む。')!=key('本を読んだ。')
    assert key('本を読む。')!=key('本を読まない。')
    assert key('彼は行く。')!=key('彼は行け。')
    assert key('行く。')!=key('行こう。')
    assert key('行こう。')==key('行きましょう。')
    assert key('食べる。')!=key('食べられる。')
    assert key('これは本です。')!=key('これは本です？')
    assert key('新たに税を課した。')!=key('新たな税を課した。')


def test_pattern_cap_has_explicit_coverage_exceptions():
    features={0:{'a','b'},1:{'a'},2:{'c'},3:{'c'}}
    selected,extra=cap_group([0,1,2,3],features,{i:str(i) for i in features},1)
    assert selected==[0,2] and extra==1
    assert set().union(*(features[i] for i in selected))=={'a','b','c'}
    assert cap_group([3,2,1,0],features,{i:str(i) for i in features},1)==(selected,extra)


def test_role_coverage_is_separate_from_simple_noun_presence():
    def row(noun,particle):
        return dict(segments=[dict(surface=noun,pos='名詞'),dict(surface=particle,pos='助詞'),dict(surface='読む',pos='動詞')])
    assert ('role','本','を','読む') in coverage(row('本','を'))
    assert coverage(row('本','を'))!=coverage(row('本','に'))
