from scripts.novelty_evaluate import review_selection
from scripts.review_novelty import summarize


def fixture():
    selected=review_selection([dict(status='success',text=t,kana=k) for t,k in [('猫','ねこ'),('犬','いぬ'),('空読む','そらよむ')]],3,0)
    report=dict(results=[dict(name='test',review_selection=selected,metrics=dict(details=[
        dict(text='猫',exact=True,near=True),dict(text='犬',exact=False,near=True),dict(text='空読む',exact=False,near=False)]))])
    mapping=[dict(blind_id=i,key=f'test:{i}') for i in range(3)]
    ratings=[dict(blind_id=str(i),status='success',text=s['text'],kana=s['kana'],natural=str(int(i<2)),reading_ok='1',
                  reviewer='Codex',reviewer_kind='agent',notes='fixture judgement') for i,s in enumerate(selected)]
    return report,mapping,ratings


def test_joint_metric_does_not_reward_copies_or_naturalness_failures():
    report,mapping,ratings=fixture()
    result=summarize(report,mapping,ratings)['by_condition']['test']['by_reviewer_kind']
    agent=result['agent']
    assert agent['rated']==3
    assert agent['joint_nonexact_count']==1 and agent['joint_nonexact_rate_among_rated']==1/3
    assert agent['joint_not_near_count']==0
    assert agent['train_nonexact']['joint']==1 and agent['train_nonexact']['reviewed']==2
    assert result['independent_human']['joint_nonexact_rate_among_rated'] is None


def test_unrated_labels_are_not_imputed():
    report,mapping,ratings=fixture()
    for row in ratings:row.update(natural='',reading_ok='')
    result=summarize(report,mapping,ratings)['by_condition']['test']['by_reviewer_kind']['agent']
    assert result['rated']==0 and result['joint_nonexact_rate_among_rated'] is None
