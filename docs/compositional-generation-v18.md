# v18：文の追加と抽選比率の効果を分ける

2026-10-30。[v17](compositional-generation-v17.md)の新規教師文2,518件を通常比率でも学び、文追加×抽選比率の2×2比較を完成した。さらに、新しい生成seedで通常比率の対照と追加モデルを比較した。**コピー率の低下は両seed群で確認できたが、自然で読みが合う非一致文の増加は新しいseedで再現せず、標準には採用しない。**

## デモで試す

[ローカルデモ](http://127.0.0.1:8000/)を再読み込みし、「v18 新規文追加・通常比率」を選んで「読み込む」→難易度→「お題を生成」。v15比較基準、v17新規文2518件追加とも比較できる。新モデルは `models/everyday-v18-expanded-base-3mb/`、5ファイル計 **2,528,193 bytes**。

全13モデルをcatalogから再ビルドし、WASM生成、実Edgeでのモデル切替・読み込み・3難易度生成・標準への再切替を確認した。ローカルURLはこのPCのサーバー稼働中に利用できる。公開GitHub Pagesへのpush・デプロイは未実施。標準はstories-v2-diverse-3mbを維持する。

## 仮説と先に固定した条件

v15では少数の教師文を多く抽選すると自然さとコピーが同時に増えた。v17では教師文の種類を増やして改善の兆しがあったが、通常比率での追加効果は未比較だった。そこで今回の変更をcorpus mixtureだけに絞る。

| 条件 | データ | Tatoeba / teacher / everyday / paired |
|---|---|---|
| v15 base | 旧56,593件 | 65 / 15 / 15 / 5% |
| v15 quality | 旧56,593件 | 25 / 45 / 25 / 5% |
| v17 expanded | 新59,111件 | 25 / 45 / 25 / 5% |
| **v18 expanded-base（今回の新モデル）** | 新59,111件 | 65 / 15 / 15 / 5% |

新しい教師生成・外部取得・修正文・語彙拡張は行わない。追加2,518件、学習から外したteacher-test282件、既存validation/test/family-testのバイト列とhashはv17のまま。教師のモデルとデータの出典・ライセンスも継承する。[データ対応](everyday-v18-data.json)、[v17の選別条件と出典](everyday-v17-data.json)を参照。

全条件は同じ語彙8,802 ID、4層・幅128・1,796,608パラメータ、seed42、batch128、最終5,000stepの重みを使う。今回もランダム初期値から学習し、入力マスク・分岐補正・修正学習は使わない。新モデルは最終stepを固定。旧v15の対照はvalidationで選ばれたものだが、そのstepも5,000である。validationの比率は全条件65/15/15/5。

新モデルと対照の初期state hashは一致した。データが増えるため、学習行の集合と抽選traceのhashは異なる。同じ行を引いた比較とは扱わない。RTX 4070 Ti SUPER、PyTorch2.9.0+cu128、CUDA bf16。学習時間69.5秒、peak CUDA allocated601,620,992 bytes。既存と同じく前処理等を除く更新ループの測定範囲。[学習記録](everyday-v18-training.json)に設定・trace・checkpoint hashを残す。

既存seedでの結果にかかわらず、新しい開始seed887000でも通常比率の2モデルを各600要求・100件点検する条件を先に固定した。[事前条件と設定hash](everyday-v18-protocol.json)を参照。これは追加の生成seed比較で、独立した人手評価や複数の学習seedによる確認ではない。予約した採用確認用830000台は使っていない。

## 既存の開発seedによる4条件比較

各600要求、easy/normal/hard各200、履歴0、候補1、temperature0.8/top-p0.95。request seedは `880000 + request_id*4`、点検100件はseed9041で事前抽出。全条件600/600成功した。

コピー診断の共通参照は4条件の実trainの和集合、すなわちv17-expandedのtrainのみ。heldoutを参照へ混ぜず、各モデルの学習元・語彙・配布重みとの対応をhashで照合した。近似一致は文字bigram集合Dice0.8以上で完全一致を含む。照合は生成後の診断であり、推論に文一覧や棄却表を配布しない。

| 条件 | 固有本文 / 600 | 完全一致 | 近似一致 | 自然さ / 100 | 読み / 100 | 両方合格かつ非一致 / 100 | 両方合格かつ近似一致も除外 / 100 |
|---|---:|---:|---:|---:|---:|---:|---:|
| v15 base | 593 | 23.5% | 38.7% | 67 | 98 | 44 | 32 |
| v15 quality | 549 | 67.5% | 84.8% | 88 | 98 | 25 | 8 |
| v17 expanded | 575 | 51.7% | 76.2% | 92 | 100 | 43 | 14 |
| v18 expanded-base | 597 | 15.5% | 33.2% | 72 | 99 | **57** | **41** |

共通参照を拡大したため、v15当時の近似一致率とはわずかに異なる。v18の単語Self-BLEU4は0.485、v15 baseは0.477で、コピーが減っても語彙系列の多様性指標全てが改善したわけではない。

この開発用seedでは、通常比率でも教師文の種類を増やすことでコピーと自然な非一致文が改善したように見えた。教師文を多く抽選する条件は引き続き自然さが高い一方、コピーが多い。この折り合い自体は文追加だけでは消えていない。

## 新しい生成seedでは改善が安定しなかった

同じ2モデルを開始seed887000、requestごとに+4、点検抽出seed9051で比較した。難易度・件数・生成条件・共通train参照は上と同じ。各600/600成功。結果を見て学習・設定を選び直したり、生成を取り除いたりしていない。

| 条件 | 固有本文 / 600 | 完全一致 | 近似一致 | 自然さ / 100 | 読み / 100 | 両方合格かつ非一致 / 100 | 両方合格かつ近似一致も除外 / 100 |
|---|---:|---:|---:|---:|---:|---:|---:|
| v15 base | 594 | 23.7% | 41.0% | 73 | 100 | **53** | 35 |
| v18 expanded-base | 594 | 18.0% | 35.5% | 68 | 99 | **52** | 39 |

コピー率の低下は両seed群で同じ方向だが、自然な非一致文は開発側44→57/100に対して新seed側53→52/100。自然さも開発側67→72/100に対して新seed側73→68/100で方向が変わった。近似一致まで除いた件数は新seedでも35→39/100だが、小標本で4件差にとどまる。開発側の良い結果だけを根拠に採用しない。

「フランス語にかなり上手に速く走った」「車の調子を持っている」等の格・意味関係の誤りが残る。「ベランダになると風も心地よい」等は構文に加え、風の読みが「ふう」になっていた。新規性だけでは、これらを良いお題として数えられない。

追加比較の旧baseには自殺を望む・自殺に言及する出力も点検標本中にあった。文法上の自然さとタイピングのお題としての題材適性を混同せず、該当票に注記した。題材適性の網羅的な評価や実行時のフィルターは今回追加していない。

## 点検の再利用と限界

開発側の既存3モデル300件は、配布5ファイルのhash・生成条件・抽出した全文とsample hashの一致を検査して元の判定を再利用した。加えて開発側2件、新seed側4件は、hash検証済みのv15〜v17票と本文・かなが同一で判定が矛盾しないため再利用した。

残る294出力の異なる本文・かな292組をCodexが点検し、同一出力には同一判定を反映した。600行は600件の新規独立採点ではない。条件名の列を除いた一覧だが、開発側の新規採点が新モデルに集中し、既知の文体からも条件を推測できる。厳密な独立盲検評価とは扱わない。

基準はv15〜v17を継承し、日常の省略・口語・成立する解釈は許容、明確な格・述語・修飾・意味関係の破綻と解消できない読みは不合格とした。境界例を含め全判断理由を保存する。今回も独立した人間の採点はなく、一つの学習seedと主観を含む100件標本の比較である。学習seedを増やした再現性、意味漏洩の精査、統計的有意な改善は確認していない。

## 損失・組合せ診断は改善しても、品質の代用にはならない

| 診断 | v15 base | v18 expanded-base |
|---|---:|---:|
| 最終の共通validation loss | 2.6296 | 2.5763 |
| 未学習の教師文282件のtoken NLL | 2.4941 | 2.2045 |
| 組合せで自然側を選好 / 91 | 74 | 80 |

教師文NLLはFP32 PyTorch、difficulty・句点・EOSを含みPADを除外した3,496tokenの加重平均。組合せ診断は既存100対中、符号化できる91対の最初に異なるtokenのFP32確率比較で、全文の自然さではない。全splitの対象共起キー一致0件を再検査したが、同義語や係り受けを含む完全な意味漏洩検査ではない。

これらの診断は良い方向に変わった一方、新しい生成seedの自然さは改善しなかった。損失・対照問題の得点と、自由生成の文章品質を別々に測る必要を示す結果である。[教師文損失](everyday-v18-teacher-test-loss.json)、[組合せ監査](everyday-v18-combination-audit.json)を参照。

## 容量・速度・検証

新モデルの5ファイル計2,528,193 bytes、INT8保存・FP32 CPU/WASM演算、アプリにPython・CUDAは不要。CPU normal90要求ではp50 **23.3ms**、p95 **33.0ms**。release CLI、開始seed890000、履歴0、候補1、temp0.8/top-p0.95、ロード・JSON直列化を除き初回・再試行を含む。

実Edgeのnormal30件はp50 **15.1ms**、p95 **20.4ms**。Web Workerのgeneration_ms、seed42、履歴16保持でロード・描画を除く。CPU・ブラウザーの専用測定は自由生成評価・学習・Pythonテスト完了後に実施した。環境・履歴・標本が異なり、全要求の上限保証ではない。

Python129テスト、標準モデル整合性、FP32/INT8数値照合、全13モデル×30件のWASM smoke、実Edge全13モデル×3難易度を通過。Rustソースは変更しておらず、workspace全テストは再実行していない。公開Pagesへのデプロイと独立した人間の採点は未実施。

## 再実行と証跡

既存venvとv17準備データを使う。既存checkpointを保護するため、再学習時は設定のcheckpointとexport先を新しいパスへ変更する。

```powershell
.venv/Scripts/python.exe -m training.train --config configs/everyday-v18-expanded-base.json
.venv/Scripts/python.exe -m scripts.export_experiment --checkpoint build/checkpoints/everyday-v18-expanded-base.pt --prefix build/v18-export-recheck --output build/v18-runtime-recheck
.venv/Scripts/python.exe -m scripts.evaluate_candidates --config configs/everyday-v18-evaluation.json --output build/v18-development-recheck
.venv/Scripts/python.exe -m scripts.evaluate_candidates --config configs/everyday-v18-fresh-seeds.json --output build/v18-fresh-recheck
.venv/Scripts/python.exe scripts/build_wasm.py --model-catalog configs/demo-models.json
node scripts/smoke_wasm.mjs --catalog configs/demo-models.json
$env:FUMIORI_REQUIRE_EXPERIMENTAL='1'
$env:FUMIORI_REQUIRE_CATALOG='configs/demo-models.json'
node scripts/test_pages.cjs
```

ブラウザー環境はPages workflowと既存手順に合わせる。`scripts.review_candidates` / `scripts.review_novelty` へ元report・mappingと実際に採点した票を渡してhash付き集計を作る。空欄の人手用票を採点済みとして扱わない。

- 開発側：[全生成診断とモデルhash](everyday-v18-development-measurements.json)、[比較CSV](everyday-v18-development-summary.csv)、[対応表](everyday-v18-development-review-map.json)、[全400行採点](everyday-v18-development-blind-agent-review.csv)、[品質集計](everyday-v18-development-agent-review.json)、[結合集計](everyday-v18-development-joint-review.json)、[再利用対応](everyday-v18-development-review-reuse.json)、[人手用票](everyday-v18-development-human-review.csv)
- 新seed側：[全生成診断とモデルhash](everyday-v18-fresh-measurements.json)、[比較CSV](everyday-v18-fresh-summary.csv)、[対応表](everyday-v18-fresh-review-map.json)、[全200行採点](everyday-v18-fresh-blind-agent-review.csv)、[品質集計](everyday-v18-fresh-agent-review.json)、[結合集計](everyday-v18-fresh-joint-review.json)、[再利用対応](everyday-v18-fresh-review-reuse.json)、[人手用票](everyday-v18-fresh-human-review.csv)
- [新規292組の点検](everyday-v18-new-unique-agent-review.json)、[CPU測定](everyday-v18-normal-timings.json)、[ブラウザー測定](everyday-v18-browser-timings.json)、[数値照合](everyday-v18-parity.json)、[検証logのhash](everyday-v18-verification.json)

checkpoint・全出力segments・生の測定結果は `build/` に保持し、対応するhashを記録した。今回わかったのは「通常比率でも文の種類を増やすとコピーは減ったが、自然さの改善は生成seedを変えると安定しない」ということ。次の学習信号を検討する際も、既存seedでの最良値だけを追わず、自由生成での自然さと意味関係の転移を評価する。
