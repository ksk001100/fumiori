from scripts.audit_data_diversity import NeighborIndex, lexical_sequence, replacement_groups


def row(text, words):
    return dict(text=text,segments=[dict(surface=s,pos=p) for s,p in words])


def test_nearest_excludes_self_but_counts_another_duplicate():
    rows=[dict(text='白い靴を選ぶ。'),dict(text='白い靴を選ぶ'),dict(text='赤い靴を選ぶ。')]
    index=NeighborIndex(rows)
    assert index.nearest(rows[0]['text'],exclude=0)==(1.,1)
    score,other=index.nearest(rows[2]['text'],exclude=2)
    assert other==0 and 0 < score < 1
    assert NeighborIndex([rows[0]]).nearest(rows[0]['text'],exclude=0)==(0.,None)


def test_replacement_requires_same_position_and_distinct_texts():
    rows=[row('白い靴を選ぶ',[('白い','形容詞'),('靴','名詞'),('を','助詞'),('選ぶ','動詞')]),
          row('赤い靴を選ぶ',[('赤い','形容詞'),('靴','名詞'),('を','助詞'),('選ぶ','動詞')]),
          row('靴白いを選ぶ',[('靴','名詞'),('白い','形容詞'),('を','助詞'),('選ぶ','動詞')])]
    assert replacement_groups(rows)==[[0,1]]
    assert replacement_groups([rows[0],rows[0]])==[]
    assert replacement_groups([rows[0],dict(rows[0],text='白い靴を、選ぶ')])==[]
    assert replacement_groups([row('愛してる',[('愛してる','動詞')]),row('捕まえた',[('捕まえた','動詞')])])==[]


def test_noun_mask_keeps_predicate_and_drops_only_punctuation():
    r=row('母と読む。',[('母','名詞'),('と','助詞'),('読む','動詞'),('。','記号')])
    assert lexical_sequence(r,True)==('<名詞>','と','読む')
