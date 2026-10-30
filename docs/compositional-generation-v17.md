# v17：ローカル教師の新規文を増やす比較

2026-10-30。[v15](compositional-generation-v15.md)で少量の教師文を多く抽選するとコピーが増え、[v16](compositional-generation-v16.md)の分岐補正でも十分な改善は得られなかった。今回はモデル容量と学習予算を保ち、教師文の種類を500件／2,518件増やした。**同じ混合比の対照に対し、Codex点検で自然さ・読みが合格するtrain非一致文は25→32→43/100に増えた。ただし近似文が多く、目標達成や標準への採用とは扱わない。**

## デモ

[ローカルデモ](http://127.0.0.1:8000/)を再読み込みし、「v17 新規文500件追加」または「v17 新規文2518件追加」を選んで「読み込む」→難易度→「お題を生成」。新モデルはそれぞれ **2,528,193 bytes**。従来モデルも残した全12モデルを再ビルドし、WASM生成と実Edgeでの選択・読み込み・3難易度生成を確認した。

標準は引き続きstories-v2-diverse-3mb。今回の変更を公開GitHub Pagesへpush・デプロイする操作は行っていない。ローカルURLはこのPCのサーバー稼働中に利用できる。

## データ追加の条件

既存のローカル教師 `llm-jp/llm-jp-3.1-1.8b-instruct4` を再利用した。revisionは `f19510db409090bb1737f24f868d17c4bdc86c8e`。既存ファイルのhashを照合し、新しい外部取得・API呼び出しは行っていない。教師はオフラインの文収集専用で、アプリへ配布しない。

few-shot例はv13除去済みtrainのteacher/everyday/paired群7,279行から取り、元行との完全一致とhashを照合した。heldoutの行を例として与える入力は拒否する。教師生成はseed840100、request offset10000、1,440request、batch16、temperature0.85、top-p0.95、最大192token。24話題×3長さを依頼するが、実際の出力の話題分布を人手で分類して均衡を保証したものではない。生成352.2秒、peak CUDA allocated 4,948,144,128 bytes。pilot24requestの文は本学習へ入れない。

生成された10,256文を、既存の語彙ID・辞書と照合して選別した。全ての旧split、v14〜v16の既存生成出力との正規化一致を除外し、既存100対の名詞・格・述語キーも除いた。既存評価出力を保護に使うため、この比較は継続中の開発評価であり、独立した確認評価ではない。語彙に同じ表記・品詞で複数の読みがある場合は追加対象から外す。この検査だけで文脈読みの正しさを保証しない。

| 結果（最初に該当した理由） | 文数 |
|---|---:|
| 語彙外 | 2,107 |
| 辞書／語境界の不整合 | 2,227 |
| 保護する組合せキー | 1,461 |
| かな長の範囲外 | 166 |
| 既存語彙で複数読み | 433 |
| 重複・保護対象との一致 | 1,062 |
| 採用候補 | 2,800 |

候補を本文hashで固定分割し、282件を共通の `teacher-test.jsonl` として学習から外した。残る2,518件を本文SHA-256順に並べ、先頭500件だけを加えるsmallと全件を加えるexpandedを比較する。新しい教師文で語彙を作り直さない。旧validation2,692件・test2,642件・family-test145件はバイト列とhashを維持した。

| 条件 | train | 新規教師文 | 新規teacher-test |
|---|---:|---:|---:|
| v15 quality対照 | 56,593 | 0 | 学習なし・診断のみ282 |
| v17 small | 57,093 | 500 | 282 |
| v17 expanded | 59,111 | 2,518 | 282 |

追加文は `auto_aligned` の自動対応付け。2,518件から出力を見る前にseed9047で抽出した50件のCodex点検は自然さ49/50、読み50/50だった。「趣味で始めたジョギングコースを新しく設定しました」は修飾が不自然として不合格。点検後に文を削除・修正したり、評価ラベルをlossへ使ったりしていない。独立した人間による採点ではなく、全件の品質保証でもない。[抽出文](everyday-v17-data-review-sample.json)、[点検票](everyday-v17-data-agent-review.csv)、[集計](everyday-v17-data-agent-review.json)に残す。

全splitを既存の除去パイプラインと別の共起監査で再検査し、対象キーの一致は0件。構成語の被覆は96/100、符号化可能な対は91/100のまま。語形・辞書表記の正規化を含む文字列ベースのキーであり、同義語・係り受け・教師の事前学習まで含む意味的な未知性の保証ではない。

## 同じ容量・更新回数で再学習

v15 qualityと同じcorpus mixture（Tatoeba25%・teacher45%・everyday25%・paired5%）、4層・幅128・1,796,608パラメータ、batch128、seed42、CUDA bf16でランダム初期値から学習した。名詞マスク・分岐補正・修正学習は使わない。新2モデルは `select_best_validation: false` で最終5,000stepを固定し、検証損失が最良の4,500stepへ戻さない。旧対照も選択stepは5,000だった。

| 条件 | 使用step | 最終共通validation loss | 学習時間 | 組合せで自然側を選好 / 91 |
|---|---:|---:|---:|---:|
| v15 quality対照 | 5,000 | 2.8599 | 既存記録 | 67 |
| v17 small | 5,000 | 2.8325 | 75.4秒 | 65 |
| v17 expanded | 5,000 | 2.7569 | 93.1秒 | 71 |

validationの群比率は全条件65/15/15/5。初期state hashは3条件で一致する。データ追加により行の集合と抽選が変わるため、学習行抽選hashは一致しない。モデルdropoutと行抽選の乱数を分離するv15の設定を維持した。学習時間は前処理等を除く既存の更新ループ測定範囲。RTX 4070 Ti SUPER、PyTorch2.9.0+cu128。新モデルのpeak CUDA allocatedは約602MB。

組合せの確率診断はFP32 checkpointの最初の異なるtokenを比較する開発指標。91対中67→65→71で、構成的一般化の安定した改善を示す結果ではない。

共通の未学習教師文282件のFP32 token NLLは **2.8596→2.6377→2.2228**。difficulty・句点・EOSを含めPADを除外し、3,496個の有効tokenで加重平均した。これは同じ教師分布における予測損失で、独立ドメインの自然さや意味関係の正答率ではない。[記録](everyday-v17-teacher-test-loss.json)を参照。

## 生成・コピー・品質

各条件600要求、3難易度各200、temperature0.8、top-p0.95、候補1、履歴0。request seedは既存の開発用 `880000 + request_id*4`、点検100件の抽出seed9041。全条件600/600成功した。

trainの集合が違うため、完全一致・近似一致の共通参照は3条件の**実際のtrainの和集合**とする。参照集合が正確に和集合と一致し、旧・新のheldout本文が混入しないことを検査する。正規化はNFKC＋末尾句点の除去、近似一致は文字bigram集合Dice0.8以上で完全一致を含む。参照照合は生成後の診断専用で、推論での棄却に使わない。

| 条件 | 固有本文 / 600 | 完全一致 | 近似一致 | 単語Self-BLEU4 | 自然さ / 100 | 読み / 100 | 両方合格かつ非一致 / 100 |
|---|---:|---:|---:|---:|---:|---:|---:|
| v15 quality対照 | 549 | 67.5% | 84.8% | 0.619 | 88 | 98 | 25 |
| v17 small | 563 | 60.5% | 79.2% | 0.601 | 90 | 100 | 32 |
| v17 expanded | 575 | 51.7% | 76.2% | 0.629 | 92 | 100 | 43 |

自分自身のtrainだけを参照した完全一致は順に67.5%、60.33%、51.67%。smallにはexpandedでだけ学習した文と一致する出力が1件あり、共通参照ではコピーに数える。対照の近似一致がv15記録とわずかに違うのも参照拡大による。

自然・読み合格かつ**近似一致も除いた**文は8→8→14/100。expandedでも大半が学習文に近く、単語Self-BLEU4も改善していない。「この靴は、みんな医者になると思う」「家族と一緒に夕食のメニューを考えながら、の準備をする」等の意味・構文の失敗が残る。非一致の増加を意味的な新しさと同一視しない。

過去の通常比率v15 baseは自然で読みが合う非一致44/100だった。今回のexpanded43/100はそれを超えておらず、自然さとコピーの折り合いを探る段階である。この過去値は混合比・参照が異なる参考で、今回の厳密な対照ではない。

### 点検票の由来と制限

対照100件はv15のモデル全5ファイルhash・生成条件・抽出文のsample hashが完全一致したため、元の判定を再利用した。新モデルの12出力も、hash検証済みのv15/v16票と本文・かなが一致し、判定が矛盾しないものだけを再利用した。残る188出力の異なる本文・かな187組をCodexが条件名を隠した一覧で点検し、重複へ同じ判定を適用した。

したがって300行は300件の新規・独立採点ではない。既知の文体から条件を推測できる限界もある。通常の省略・口語・成立する解釈はv15と同様に許容し、明確な格・意味・構文の破綻は不合格。境界例と理由も残した。独立した人間の点検、複数学習seed、新しい確認生成seed、統計的有意差の検証は未実施。

[全300行](everyday-v17-blind-agent-review.csv)、[新規187組](everyday-v17-new-unique-agent-review.json)、[再利用の対応とhash](everyday-v17-review-reuse.json)、[品質集計](everyday-v17-agent-review.json)、[非一致との結合集計](everyday-v17-joint-review.json)、[未記入の人手用票](everyday-v17-human-review.csv)を参照。

## 容量と生成時間

各bundleは重み1,873,528、語彙638,890、設定203、manifest2,125、LICENSE13,447 bytesの5ファイルで計2,528,193 bytes。出典追記のため旧モデルより643 bytes増えるが、構造・語彙・推論演算は増やしていない。INT8は保存形式で、CPU/WASM演算はFP32。

| 条件 | CPU normal p50 / p95 | Edge normal p50 / p95 |
|---|---:|---:|
| v17 small | 25.0 / 39.1ms | 15.1 / 26.3ms |
| v17 expanded | 22.8 / 30.4ms | 16.1 / 35.6ms |

CPUはrelease CLI、各90要求、seed885000、候補1、履歴0、temp0.8/top-p0.95。ほかのプロジェクト学習・テストが終わってから逐次実行した。ロード・JSON直列化を除き、初回・再試行を含む。Edge156.0.4314.56は各30件、Web Workerのgeneration_ms、seed42、履歴16を保持し、ロードと描画を除く。全12モデルのブラウザーテスト開始時には自由生成比較の終盤が重なったが、最後に訪問したv17の測定時には比較は終了している。CPUとブラウザーは履歴等が異なるので同じ条件の速度比較ではない。いずれも今回の標本でp95が100ms未満という結果で、全環境・全要求の上限保証ではない。

## 再実行と保存先

既存venv・ローカル教師・v13準備データが必要。出力先は既存成果を上書きしない新しいパスを使う。few-shotファイルの作成元と群は[データ記録](everyday-v17-data.json)の `fewshot_source` に保持する。教師は `--examples` で指定したtrain部分集合以外を参照しない。

```powershell
.venv/Scripts/python.exe -m scripts.teacher_prose --output build/teacher-prose/v17-recheck.jsonl --requests 1440 --offset 10000 --batch-size 16 --max-new-tokens 192 --seed 840100 --examples build/v17-examples/train.jsonl
# everyday-v17-data.jsonのraw・merged・outputを新しいパスにした設定を使う
.venv/Scripts/python.exe -m scripts.prepare_word_teacher --config build/v17-recheck-data.json --report build/v17-recheck-data-report.json
```

学習は `configs/everyday-v17-{small,expanded}.json` のdata_dirとcheckpointを新しいパスへ設定して `python -m training.train --config ...`。exportは次の形式で行う。

```powershell
.venv/Scripts/python.exe -m scripts.export_experiment --checkpoint build/checkpoints/everyday-v17-small.pt --prefix build/v17-small-recheck --output build/v17-small-recheck-3mb
.venv/Scripts/python.exe -m scripts.evaluate_candidates --config configs/everyday-v17-evaluation.json --output build/v17-evaluation-recheck
.venv/Scripts/python.exe -m scripts.audit_combinations --data build/everyday-v17-expanded-data --checkpoint build/checkpoints/everyday-v17-small.pt --checkpoint build/checkpoints/everyday-v17-expanded.pt --output build/v17-combination-recheck.json
.venv/Scripts/python.exe scripts/build_wasm.py --model-catalog configs/demo-models.json
node scripts/smoke_wasm.mjs --catalog configs/demo-models.json
$env:FUMIORI_REQUIRE_EXPERIMENTAL='1'
$env:FUMIORI_REQUIRE_CATALOG='configs/demo-models.json'
node scripts/test_pages.cjs
```

ブラウザー依存の準備とEdge選択はPages workflowおよび既存手順に合わせる。未採点の人手用票を `scripts.review_candidates` / `scripts.review_novelty` へ渡し、元report・mappingのhashに結び付けて集計する。

- [データ出典・原本hash・split・除去報告](everyday-v17-data.json)、[学習設定・trace・checkpoint hash](everyday-v17-training.json)
- [全生成の診断・モデルhash・抽出標本](everyday-v17-measurements.json)、[比較CSV](everyday-v17-summary.csv)、[対応表](everyday-v17-review-map.json)
- [組合せ監査](everyday-v17-combination-audit.json)、[CPU測定](everyday-v17-normal-timings.json)、[ブラウザー測定](everyday-v17-browser-timings.json)、[FP32/INT8照合](everyday-v17-parity.json)、[検証記録](everyday-v17-verification.json)

元教師出力・学習データ・checkpoint・全生成segmentsは `build/` に保持し、hashを記録した。教師のモデル重みのApache-2.0表記と、合成文・混合コーパスの権利を同一視しない。既存の各出典を継承し、新教師ソースも配布manifestとLICENSEへ追記した。

全Python129件、標準モデル整合性、新2モデル×FP32/INT8照合、全12モデル×30件のWASM smoke、実Edgeで全12モデル×3難易度と標準への再切替を検証した。評価スクリプトのWindowsパス表記不一致は修正し、新しい出力先で3条件を完走した。Rustソース変更はなく、workspace全テストは再実行していない。次は独立した自然さ・読みの点検と、話題・意味関係の種類を増やす学習信号を検討する。現在の追加文量だけで構成的一般化が解決したとは扱わない。
