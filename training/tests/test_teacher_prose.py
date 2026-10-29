from scripts.teacher_prose import extract_sentences, prompt_for, TOPICS
from scripts.prepare_teacher_prose import split_for, orthographic_readings
import io
import pytest


def test_teacher_output_parser_rejects_numbers_and_partial_lines():
    rows = extract_sentences('以下の通りです。\n1. 窓を開ける。\n- 本を読む。\n読んだ本は３冊。\n途中で切れた\n雨だ。晴れた。')
    assert '窓を開ける。' in rows and '本を読む。' in rows
    assert all('３' not in row and row.endswith('。') for row in rows)
    assert '途中で切れた' not in rows and '雨だ。晴れた。' not in rows


def test_prompts_cover_topics_and_lengths_without_sentence_skeletons():
    assert {prompt_for(i,42)[0] for i in range(len(TOPICS))} == set(TOPICS)
    assert prompt_for(0,42) == prompt_for(0,42)
    assert len({prompt_for(i*len(TOPICS),42)[1] for i in range(3)}) == 3
    assert all('{' not in prompt_for(i,42)[1] for i in range(100))


def test_teacher_splits_share_canonical_body_normalization():
    assert split_for('本を読む') == split_for('本を読む。')
    assert split_for('本を読んだ。') == split_for('本を読む。')
    assert {split_for(f'短文{i}。') for i in range(1000)} == {'train','validation','test'}


def test_teacher_download_resumes_silent_early_eof(monkeypatch,tmp_path):
    from scripts.fetch_teacher import download
    class Response(io.BytesIO):
        def __init__(self,payload,status,headers):
            super().__init__(payload)
            self.status,self.headers=status,headers
    requests=[]
    def open_url(request,timeout):
        requests.append(request)
        return (Response(b'ab',200,{'Content-Length':'5'}) if len(requests)==1 else
                Response(b'cde',206,{'Content-Range':'bytes 2-4/5'}))
    monkeypatch.setattr('scripts.fetch_teacher.urllib.request.urlopen',open_url)
    target=tmp_path/'model.safetensors'
    download('https://example.invalid/model',target)
    assert target.read_bytes()==b'abcde'
    assert requests[1].get_header('Range')=='bytes=2-'
    assert not target.with_suffix('.part').exists()


def test_validation_mixture_tracks_new_corpus_without_erasing_old_group_losses():
    from training.train import mixture_loss
    assert mixture_loss({'original':2.,'teacher':1.},{'original':.5,'teacher':.5}) == 1.5
    assert mixture_loss({'original':2.,'teacher':1.},{'original':2.,'teacher':2.}) == 1.5
    with pytest.raises(ValueError):
        mixture_loss({'teacher':1.},{'teacher':.5,'missing':.5})
    with pytest.raises(ValueError):
        mixture_loss({'teacher':1.},{'teacher':float('nan')})


def test_curated_orthographic_correction_does_not_rewrite_unrelated_readings():
    words=[dict(surface='言う',kana='ゆう',pos='動詞'),
           dict(surface='夕方',kana='ゆうがた',pos='名詞')]
    fixed=orthographic_readings(words)
    assert fixed[0]['kana']=='いう' and fixed[1]['kana']=='ゆうがた'
    assert words[0]['kana']=='ゆう'  # Original annotations remain available to audit.
