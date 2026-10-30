# v20：10万文への拡充と2・4・8層の比較

2026-10-30。v19の整理済み57,354文へ、既存語彙で符号化できる新しい散文42,646文を加え、**学習用100,000文**を構築した。近いパラメータ数の2・4・8層を学習し、全4モデルをデモへ追加した。**今回の標本では4層が品質面で先行、2層が速度面で有利。8層は品質を上げず、CPU90件中2件が100msを超えた。** 4層と2層の品質差を確定したものでも、全構成からの最適解を発見したものでもない。

## デモ

[ローカルデモ](http://127.0.0.1:8000/)を再読み込みし、「v20 10万文・4層」「v20 10万文・2層」「v20 10万文・8層」を選択→「読み込む」→難易度→「お題を生成」。「v20 対照・約5.7万文・4層」でデータ追加前とも比較できる。ローカルサーバー稼働中に利用できる。

全19モデルを再ビルドし、WASMと実Edgeで選択・読み込み・3難易度生成・標準への再切替を確認した。標準はstories-v2-diverse-3mbを維持する。公開GitHub Pagesへのpush・デプロイは未実施。

## 文数と容量を分ける

| 条件 | 学習文数 | 層 / 幅 / FFN幅 | パラメータ | 配布5ファイル合計 |
|---|---:|---|---:|---:|
| control4 | 57,354 | 4 / 128 / 384 | 1,796,608 | 2,528,193 bytes |
| layers4 | 100,000 | 4 / 128 / 384 | 1,796,608 | 2,529,769 bytes |
| layers2 | 100,000 | 2 / 152 / 456 | 1,813,664 | 2,530,129 bytes |
| layers8 | 100,000 | 8 / 104 / 312 | 1,797,536 | 2,557,713 bytes |

語彙は全条件8,802 ID、4head、最大64token、入出力埋め込み共有。データ追加で語彙と構造を変えない4層同士の容量差は追加出典のmanifest・LICENSEによる。学習文そのものは配布しない。全候補3,000,000 bytes以内。INT8は保存形式、CPU/WASM演算はFP32で、アプリにPython・CUDA・教師は不要。

2層のパラメータ差は4層比約0.95%、8層は約0.05%。深さを増やす際に幅を減らして容量をそろえた比較であり、幅を固定した純粋な層数効果ではない。

## 水増しを抑えた追加データ

新しい外部取得・API教師生成は行わず、取得済みの `SimpleStories/SimpleStories-JA` の公開train原本を使った。revision `d25370316dd77ec0ee4fd9db57bfe57075cf4b90`、原本SHA-256を照合した。データカードのライセンスはMIT、著者表記はThe SimpleStories Project。ローカルsource.jsonではgpt-4o-mini由来の合成物語と記録されている。独立した人間による全件点検済みの文章ではない。

元の57,354行は本文・かな・区切り・出典・順序を保持したまま追加した。旧validation2,692、test2,642、family-test145、teacher-test282件はバイト列とhashを変えない。新しいstories-test1,000件は**作品全体のSHA-256**でtrainから分け、同じ作品の別文が両側に入らないことを検査した。

追加の42,646文は28,028作品から選んだ。全語が既存の表記・かな・品詞IDに一致し、かな5〜40文字・64token以内の文だけを採用する。新しい語彙・かなの文字fallbackは追加しない。

主な選別条件は以下。

- 既存全split、旧コーパス、旧モデルの既存生成出力との完全一致・正規化一致を除外する。
- v19の活用を区別するキーで、読点・丁寧語等による近似を除外する。
- 追加する文が、保護文と既に採用した追加文に対して文字bigram Dice **0.85以上**なら採用しない。検索は固定順序のprefixで候補を絞る閾値join。総当たりDiceとの一致を単体テストした。意味類似モデルではない。
- 追加分の名詞伏せパターンは難易度ごと32件、先頭2区切りが同じ書き出しは500件、元のtopicラベルは1,500件を上限とする。旧コーパス全体がこの上限以下になるという意味ではない。
- 辞書外・語境界衝突、既存語彙内の複数読み、保護する名詞・格・述語キー、台詞の切れ端や一部の文脈依存表現・生成指示由来の不自然な冒頭を保留する。
- 新規候補では「宇宙人／うちゅうにん」「何より／なんより」等の確認した読み誤りと、読みが不確かな単独「実／じつ」を保留する。既存の語彙IDを書き換えず、推論にも禁止語表を入れない。

新規候補の初期点検で、複合語「宇宙人」が複数区切りに分かれるため表記トークンの一致だけでは読み誤りを捕捉できないことや、「静かな森の中で女の子がいました」等の存在の格の不自然さが見つかった。点検と回帰テストを加えてから10万文モデルの学習へ進んだ。[事前条件と修正経緯](everyday-v20-protocol.json)に条件・hash・時点を保存した。対照モデルの学習は独立した元データで先に開始している。

最終選別では62,949作品、678,222断片を走査し、42,646文と評価1,000文を採用した。shape/context327,677、語彙外166,011、辞書・境界88,350、複数読み15,605、再出現候補16,107、集中上限12,493、対象共起2,255、近似2,083、完全一致等1,470、かな長1,349、確認済み文脈読み605、文体近似571件を最初に該当する理由で除いた。保守的な規則であり、除外文が全て誤った日本語というラベルではない。

追加分のかな長区分・話題ラベル等の実数、原本・全split・処理コードのhashは[データ記録](everyday-v20-data.json)に保存した。話題はmetadata上53種類で、実際の意味・文体が均等である保証ではない。

### 多様性と品質の限界

別実装の監査で追加文250件を無作為抽出し、自身を除く全trainと照合した。Dice0.85以上は0/250、0.8以上は20/250、0.9以上は0/250。全追加文のうち「4単語以上で1単語だけ違う別文がある行」は4,336/42,646、名詞伏せパターンを共有する行は7,559件。表層の近似を抑えた結果で、意味的な新規性・有効な文数の認証ではない。[全監査](everyday-v20-data-diversity-audit.json)を参照。

最終選別直前の追加文60件のCodex点検は、物語・比喩を許容する文法評価59/60、読み59/60だった。読み誤りには除外規則を追加した。「ふと、朝日が昇ると…」のような不自然な副詞のつながりは限界として記録し、全件の自然さを保証しない。この標本は最終版の全件審査でも、独立人手評価でもない。[点検票](everyday-v20-training-data-agent-review.json)、[途中の点検](everyday-v20-data-refinement-review.json)、[初期点検](everyday-v20-initial-data-agent-review.json)を参照。

新データには物語調・擬人化・前後を参照する文が残る。文数・近似率の条件だけで「日常文の質が全て向上した」と扱わない。普通の風景描写を排除する方針でもない。

## 学習条件と損失

全モデルをseed42、batch128、固定最終10,000step、CUDA bf16でランダム初期値から学習した。従来v19の5,000step・corpus mixtureとは異なるため、対照control4を新たに作った。今回の全条件は**難易度を等質量にし、その中では行を一様抽選**する。同じ抽選方針でもデータ追加により出典の期待質量が変わる。元の65/15/15/5比率を固定した比較ではない。

4層の対照と10万文モデルの初期state hashは一致する。異なる構成は同じseedでも重みの形状が違い、同じ初期重みとは言わない。validationの基準は従来の65/15/15/5を全条件で固定し、最良stepへ戻さず最終stepを使用した。

| 条件 | 最終共通validation loss | 新作品1,000文のtoken NLL | 自然側を選好 / 91 | 学習時間 |
|---|---:|---:|---:|---:|
| 対照4層・57,354文 | 2.6081 | 4.1383 | 75 | 176.7秒 |
| 4層・10万文 | 2.5929 | 2.1342 | 78 | 160.2秒 |
| 2層・10万文 | 2.5976 | 2.1397 | 75 | 100.8秒 |
| 8層・10万文 | 2.6101 | 2.1473 | 72 | 284.8秒 |

学習時間はRTX 4070 Ti SUPER・PyTorch2.9.0+cu128、既存の更新ループ測定範囲。CPU処理等との重なりが異なるため厳密な速度ベンチマークではない。全学習曲線、peak CUDA memory、抽選trace、checkpoint・コードhashは[学習記録](everyday-v20-training.json)。新作品NLLはFP32、difficulty・句点・EOSを含みPADを除外した有効token加重平均。自然さ・意味の正答率ではない。[heldout診断](everyday-v20-heldout-loss.json)を参照。

組合せは既存100対中符号化可能な91対、最初に異なるtokenのFP32確率による診断。全splitの対象キー一致0件を再検査した。係り受け・同義語・教師の事前学習を含む意味漏洩の保証ではない。[監査](everyday-v20-combination-audit.json)を参照。

## 自由生成：同じ新しいseedで比較

各条件600要求、easy/normal/hard各200、履歴0、候補1、temperature0.8/top-p0.95。request seedは `920000 + request_id*4`。点検100件はseed9101で出力を見る前に抽出した。全条件600/600成功。

コピー参照は全条件の実trainの和集合、すなわち10万文train。完全一致はNFKC＋末尾句点、近似は文字bigram集合Dice0.8以上で完全一致を含む。各自のtrainだけを使う値も別に保存した。これらの索引は生成後の評価専用。

| 条件 | 固有本文 / 600 | 完全一致 | 近似一致 | 自然さ / 100 | 読み / 100 | 両方合格かつ非一致 / 100 | 両方合格かつ近似も除外 / 100 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 対照4層 | 598 | 18.2% | 29.5% | 75 | 100 | 61 | 49 |
| 4層・10万文 | 597 | 10.0% | 23.3% | 86 | 100 | **78** | **68** |
| 2層・10万文 | 594 | 14.0% | 28.0% | 85 | 100 | **74** | **61** |
| 8層・10万文 | 598 | 7.3% | 18.3% | 74 | 100 | **71** | **63** |

データ追加による4層同士の比較では自然さと自然な非一致文が増えた。8層はコピーが最も少ないが、不自然な文も増えた。「彼のクラスは社長です」「その計画は実行することに大きな学問をある」等の意味・構文の破綻が残る。コピー率の低さだけで8層を最良としない。

### 追加seedでも方向を確認した

最初の採点後、4層・2層と対照を別の開始seed930000で各600要求比較し、seed9102で事前抽出した60件を点検した。8層は最初の品質・速度が劣ったため追加比較から外した。これは候補を絞った後の探索的追試で、厳密な事前登録済みの独立確認や予約した830000台の採用確認ではない。再学習や設定変更はしていない。

| 条件 | 完全一致 | 近似一致 | 自然さ・読みの両方合格 / 60 | 両方合格かつ非一致 / 60 | 両方合格かつ近似も除外 / 60 |
|---|---:|---:|---:|---:|---:|
| 対照4層 | 12.7% | 28.2% | 39 | **33** | 26 |
| 4層・10万文 | 9.8% | 25.3% | 50 | **46** | 32 |
| 2層・10万文 | 11.3% | 26.0% | 48 | **42** | 33 |

今回は追加データの効果が両seed群で同じ方向に出た。4層は品質の主指標で先行するが、2層との差は100件で4件、60件で4件にとどまる。近似まで除いた追加seedの値は2層が1件上回った。学習seedは一つで、統計的に確立した優劣・全構成に対する最適解とは扱わない。

## 文体と採点の限界

自然な物語・比喩・擬人化、普通の省略を許容し、構文や意味関係が崩れる文を不合格にした。例えば「雪だるまと遊ぶ約束」「木々は笑って答えた」は物語として許容する一方、「彼のクラスは社長」は不合格。文体だけで合格とせず、境界例の理由も保存する。日常の事実だけを対象とする厳格な採点とは異なる。

「少年・少女・男の子・女の子・魔法・王国・冒険」のいずれかを含む出力は、最初の600件で対照2件に対し、4層53、2層51、8層49件。追加seedでは対照10、4層36、2層42件。これは文字列のproxyで童話調の人手分類率ではないが、追加ソースによる文体変化を示す。[語彙診断](everyday-v20-narrative-keywords.json)を参照。

全580行はCodex点検。既存票と同じ本文・かなで判定が矛盾しない3行を再利用し、重複2行を同一判定として、新たな575組を点検した。独立した人間の採点ではなく、条件列を隠しても文体から条件を推測できる。今回の読み合格率100%も全出力の保証ではない。複数学習seed、独立人手の自然さ・読み・題材適性、意味的に完全な未見の保証は未実施。

## 通常生成の速度

| 条件 | CPU p50 / p95 | Edge p50 / p95 | CPU最大 / 90 |
|---|---:|---:|---:|
| 対照4層 | 30.3 / 52.0ms | 30.1 / 40.8ms | 89.2ms |
| 4層・10万文 | 32.8 / 49.8ms | 33.9 / 43.1ms | 94.6ms |
| 2層・10万文 | **21.0 / 33.8ms** | **26.2 / 37.7ms** | **63.9ms** |
| 8層・10万文 | 52.8 / 93.5ms | 38.4 / 57.1ms | **127.2ms** |

CPUはrelease CLI、各90要求、開始seed925000、履歴0、候補1、temp0.8/top-p0.95。初回・再試行を含みロードとJSON直列化を除く。Edgeは各30件、Web Worker generation_ms、seed42・履歴16でロード・描画を除く。専用測定は学習・最初のCLI評価・テスト終了後に実行し、その後に追加seed評価を行った。8層はCPU2/90件で100ms超。p95が100ms未満という値を全要求の上限達成と混同しない。

## 検証・再実行

Python全143テスト、標準モデル整合性、4モデル×FP32/INT8の数値照合、全19モデル×30件のWASM smoke、実Edge全19モデル×3難易度・通常生成・標準への再切替を通過。Rustの変更はなくworkspace全テストは再実行していない。追加ソースの出典・MIT noticeも配布manifestとLICENSEへ保持し、noticeのhash不一致をexportで拒否する。

設定は `configs/everyday-v20-data.json`、学習は `everyday-v20-{control4,layers4,layers2,layers8}.json`。既存venvと、出典・旧splitを保護するための既存ローカル資産が必要。再実行時はmerged/output/checkpoint/exportを新しいパスにして成果を上書きしない。

```powershell
.venv/Scripts/python.exe -m scripts.diverse_stories --config configs/everyday-v20-data.json
.venv/Scripts/python.exe -m training.train --config configs/everyday-v20-layers4.json
.venv/Scripts/python.exe -m scripts.export_experiment --checkpoint build/checkpoints/everyday-v20-layers4.pt --prefix build/v20-export-recheck --output build/v20-runtime-recheck
.venv/Scripts/python.exe -m scripts.evaluate_candidates --config configs/everyday-v20-evaluation.json --output build/v20-evaluation-recheck
.venv/Scripts/python.exe -m scripts.evaluate_candidates --config configs/everyday-v20-fresh-seeds.json --output build/v20-fresh-recheck
.venv/Scripts/python.exe scripts/build_wasm.py --model-catalog configs/demo-models.json
node scripts/smoke_wasm.mjs --catalog configs/demo-models.json
```

ブラウザー環境はPages workflowに合わせ、`FUMIORI_REQUIRE_CATALOG=configs/demo-models.json` で全登録モデルを確認する。前処理のprototype、原本、最終JSONL、checkpoint、全生成segmentsを `build/` に保持し、hashを記録した。

- [データと出典・split hash](everyday-v20-data.json)、[事前条件と変更経緯](everyday-v20-protocol.json)、[学習曲線・初期値と抽選trace・重みhash](everyday-v20-training.json)
- 最初の比較：[全測定](everyday-v20-measurements.json)、[CSV](everyday-v20-summary.csv)、[対応表](everyday-v20-review-map.json)、[全400行採点](everyday-v20-blind-agent-review.csv)、[品質集計](everyday-v20-agent-review.json)、[結合集計](everyday-v20-joint-review.json)、[新規396組](everyday-v20-new-unique-agent-review.json)、[再利用](everyday-v20-review-reuse.json)、[未記入人手票](everyday-v20-human-review.csv)
- 追加seed：[全測定](everyday-v20-fresh-measurements.json)、[CSV](everyday-v20-fresh-summary.csv)、[対応表](everyday-v20-fresh-review-map.json)、[全180行採点](everyday-v20-fresh-blind-agent-review.csv)、[品質集計](everyday-v20-fresh-agent-review.json)、[結合集計](everyday-v20-fresh-joint-review.json)、[新規179組](everyday-v20-fresh-new-unique-agent-review.json)、[再利用](everyday-v20-fresh-review-reuse.json)、[未記入人手票](everyday-v20-fresh-human-review.csv)
- [CPU測定](everyday-v20-normal-timings.json)、[ブラウザー測定](everyday-v20-browser-timings.json)、[数値照合](everyday-v20-parity.json)、[検証logのhash](everyday-v20-verification.json)

品質・読み・新規率の採用目安にはまだ届いていない。今回の候補内では4層を品質重視、2層を速度重視の比較候補とし、標準への昇格とは区別する。
