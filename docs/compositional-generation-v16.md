# v16：実際の抽選比率を反映した分岐補正

2026-10-30。[v15](compositional-generation-v15.md)では、少量の教師合成文・自作文を多く抽選すると、自然さと同時にコピーが増えた。今回は既存の分岐頻度補正を調べ、行数から頻度を数える方法と、実際の学習文の抽選質量から数える方法を比較した。**新モデル2種類をデモに追加したが、自然で読みが合う非一致文を増やす改善は確認できず、標準には採用しない。**

## デモで比較する

[ローカルデモ](http://127.0.0.1:8000/)を再読み込みし、「v16 分岐補正・行数基準」または「v16 分岐補正・抽選比率基準」を選んで「読み込む」→難易度→「お題を生成」。標準v2、比較用v3、v14の2種類、v15の4種類も残す。新しい2モデルは各2,527,550 bytes。

全10モデルをcatalog経由で再ビルドし、WASMと実Edgeで選択・読み込み・3難易度生成を確認した。標準は引き続きv2で、公開GitHub Pagesへのpush・デプロイは今回も行っていない。

## 比較する補正

既存の `training.branches.branch_weights` は、trainの**全文prefix**ごとに次トークンの頻度pを数え、`q ∝ p^(1−strength)` の方向へ平坦化する係数q/pでtoken CEを重み付けする。語をランダムに交換せず、観測した自然な別の続きを学ぶ比率を増やす仮説である。

従来は各train行の寄与を1として数えていた。しかし今回の学習では、難易度とcorpus mixtureで行の抽選確率が異なる。行数で計算したpと、実際に学習中に見る分岐の比率は一致しない。

そこで `branch_balance.frequency_source: sampling_mass` を追加し、既存の `sampling_weights` が返す行の抽選質量を使ってprefixごとの頻度を数える。省略時は従来の `uniform_rows` のまま。`sampling_mass` は明示的な抽選重みがない設定を拒否する。

例えば2行が同じprefixから別の語へ続き、抽選比率が90:10の場合、行数では50:50と見なされる。抽選質量を反映しstrength0.5、clipなしで補正すると、期待される分岐質量は75:25になる。この例、質量を一様に拡大した場合の不変性、決定的な続き、padding、無効な質量を単体テストで検証した。

実験はstrength0.5、係数を1/4〜4へ制限する。clipとbatchごとのloss正規化があるため、学習全体が厳密にqへ一致するという主張ではない。これは既存trainの再重み付けであり、自然さの教師・意味関係の正解を追加した手法ではない。

全341,951 prefixのうち分岐するprefixは16,924。使うのはtrainだけで、validation/testの頻度や点検結果を学習へ入れない。prefix表・訓練文・係数表は推論へ配布しない。

## 学習の対照条件

新モデルはv15の比率変更モデルと同じTatoeba25%・教師45%・自作日常文25%・paired5%。v13の除去split、語彙、4層・幅128・1,796,608パラメータ、seed42、batch128、5,000stepの計算予算を固定し、ランダム初期値から学習した。名詞入力マスクは使わない。

validationは全条件65/15/15/5の共通基準で選ぶ。初期state hashと5,000batch分の学習行抽選hashは、新しい2モデルと補正なしのv15比率変更モデルで一致した。モデルdropoutと学習文抽選はv15で分離した乱数経路を維持する。

| 条件 | 選択step | 共通validation loss | 学習時間 | 組合せで自然側を選好 / 91 |
|---|---:|---:|---:|---:|
| 通常比率・補正なし（v15 base、参考） | 5,000 | 2.6296 | 既存記録 | 74 |
| 比率変更・補正なし（v15 quality、対照） | 5,000 | 2.8599 | 既存記録 | 67 |
| 比率変更・行数基準の補正 | 5,000 | 2.9325 | 87.7秒 | 69 |
| 比率変更・抽選質量基準の補正 | **4,500** | 2.8693 | 89.6秒 | 67 |

両新モデルとも5,000stepまで実行したが、共通validationで選ばれた抽選質量モデルは4,500stepの重み。最終モデルの更新回数まで同一ではないため、チェックポイント選択の影響を除いて頻度の数え方だけの効果を断定しない。学習時間は既存の測定範囲と同様に、前処理・分岐表構築等を除く更新ループ以降。CUDA bf16、RTX 4070 Ti SUPER、PyTorch 2.9.0。peak CUDA allocated memoryは両新モデル606,705,152 bytesだった。

通常比率のbaseは異なるcorpus mixtureの参考であり、そこからの差を分岐補正だけに帰属させない。過去のv11での分岐補正ともデータ・抽選・条件が異なる。

組合せ診断は既存v13の100対から符号化可能な91対を使う。全splitを再検査し、対象共起キーの一致0件を確認した。最初の異なるトークンのFP32 checkpoint確率を測る開発用診断で、全文の自然さ、意味的に完全に未見な組合せ、最終INT8モデルの正答率とは区別する。係り受け・同義語の漏洩検査は未実施。

## 生成・品質・非一致文の比較

全条件600要求、easy/normal/hard各200、temperature0.8 / top-p0.95、履歴0、候補1。seedと抽出番号はv15と同じ開発用の `880000 + request_id*4`、点検抽出seed9041で固定した。別の独立確認seedへ進んだ評価ではない。

各条件600/600成功。コピー参照は今回のtrainのみ。完全一致はNFKC＋末尾句点の正規化、近似一致は文字bigram集合Dice係数0.8以上で、完全一致を含む。新規性による出力の棄却や訓練文の禁止表は推論に使わない。

| 条件 | 固有本文 / 600 | 完全一致 | 近似一致 | 単語Self-BLEU4 | 自然さ / 100 | 読み / 100 | 両方合格かつ非一致 / 100 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 通常比率・補正なし（参考） | 593 | 23.5% | 38.2% | 0.477 | 67 | 98 | **44** |
| 比率変更・補正なし（対照） | 549 | 67.5% | 84.7% | 0.619 | 88 | 98 | **25** |
| 行数基準の補正 | 536 | 76.8% | 89.0% | 0.597 | 91 | 97 | **15** |
| 抽選質量基準の補正 | 572 | 43.8% | 58.5% | 0.489 | 77 | 100 | **29** |

行数基準はコピーが増え、自然で読みが合う非一致文も15/100に減った。抽選質量を反映するとコピーは減ったが自然さも下がり、同じ比率の対照に対する自然な非一致文は25→29/100という小さな差にとどまった。通常比率の参考44/100を超えていない。

抽選質量モデルの点検では、非一致文50件のうち両方合格が29件。近似一致まで除いた両方合格は、表の順に32、8、4、11/100だった。表面上の非一致は意味的な未知性の証明ではなく、今回は語の相性を自然に一般化する問題を解決していない。

### 評価票の再利用と限界

v15のbase・qualityはモデルの全5ファイルhash、生成条件、抽出100件の本文・かな・segments等を含むsample hashが全て一致することを確認し、既存の200採点をそのまま再利用した。追加モデルの8出力も過去と本文・かなが同一で判定を再利用した。残る192出力のうち異なる本文・かな187組を新たにCodexが点検し、重複には同一判定を反映した。

したがって400行は400件の新しい独立採点ではない。[再利用の対応と入力hash](everyday-v16-review-reuse.json)を保存し、再利用200件を新たな盲検評価として扱わない。新規分も条件名を隠して読んだagent点検で、既知の文体から条件を推測できる限界がある。独立した人間の評価や複数学習seedでの有意な改善ではない。

判定基準はv15の日常文基準を継承する。普通の省略・口語・自然な比喩は許容し、明確な格・意味・文法の破綻と解消できない読みは不合格とする。「読書帳」の用途や因果の省略等の境界判断はCSVの理由に残した。[全400行](everyday-v16-blind-agent-review.csv)、[新規187組](everyday-v16-new-unique-agent-review.json)、[品質集計](everyday-v16-agent-review.json)、[品質と非一致の結合集計](everyday-v16-joint-review.json)、[未記入の人手用票](everyday-v16-human-review.csv)を参照。

### 残った例

| 出力 | 点検 |
|---|---|
| 朝早く起きてジョギングをする | 意味・文法と読みが対応する |
| 祖父に記事の書き方を説明する | 通常の日常文として成立する |
| ハンカチを乾いたので畳んで箱にしまう | 自動詞「乾いた」の格が崩れる |
| 水泳を提出することにしたよ | 動詞と目的語の意味が合わない |
| 炒め物のポケットをスポンジに付ける | 名詞の組合せと行為の意味が崩れる |
| ご彼女はひとりでいつもタクシーを食べたことが好きです | 敬語・動詞の相性・述語が崩れる |
| 夕暮れ時の海辺を散歩している／ゆうぐれときの… | 通常の「ゆうぐれどき」に合わない |

構造検証を通ることと、意味・読みの正しさを改めて区別する。例えば「テーブルを拭くと、春が聞こえてくる」は鳥の声等を比喩で述べた詩的な描写として許容したが、日常文の基準の運用には主観が残る。こうした例を隠して機械的な品質保証とはしない。

## 容量と速度

両新モデルの配布5ファイルは各2,527,550 bytes。頻度表・コーパス・追加の選別モデルをアプリへ入れず、CPU/WASMは既存のFP32演算を使う。

| 新モデル | CPU normal p50 / p95 | Edge normal p50 / p95 |
|---|---:|---:|
| 行数基準 | 27.6 / 40.4ms | 16.2 / 24.6ms |
| 抽選質量基準 | 28.1 / 54.5ms | 15.8 / 27.6ms |

CPUは他の学習・検証が終わった後、seed884000から同方式でnormal各90要求、履歴0。Windows x86_64、Intel Family 6 Model 183、release/Candle FP32。生成部分を測り、モデルload、候補1で使用しない全文logits採点、レポート出力は除く。初回・再試行を含む。

ブラウザーはEdge 156.0.4314.56、実Web Worker、各30件、seed42の継続乱数と既定の履歴16。通信・load・描画は除く。CPUとブラウザーは条件が異なり、数値の差を改善率とはしない。今回のp95は100ms以内だが、少数の探索測定で他端末や長時間運用を保証しない。

## 採否と次の作業

両候補とも標準には採用しない。実際の抽選比率を反映した分岐補正は、行数基準よりも今回のコピーを減らせたが、自然さ・読み・非一致を同時に満たす出力を十分には増やさなかった。小さな差とcheckpoint選択の影響を考え、抽選質量方式そのものの品質改善を断定しない。

既存の少量の自然文に対する頻度補正だけでは、語の相性の知識や自然な続きの種類は新たに増えない。次は学習用の独立した入力から自然な代案の種類を増やし、同じ組合せ除去・語彙・容量で、データ量と意味的な学習信号の効果を分ける必要がある。今回の点検出力を修正学習へ流用し、同じ評価票で改善を主張することはしない。

## 再現と成果物

設定は `configs/everyday-v16-branch-{count,mass}.json`、評価は `configs/everyday-v16-evaluation.json`。既存のvenv・CUDA・v13準備データを用いる。外部データの取得、新しい教師生成、修正文の追加はない。データと配布LICENSEの出典はv15と同じ。

```powershell
.venv/Scripts/python.exe -m pytest -q training/tests/test_branches.py training/tests/test_controlled_rng.py training/tests/test_model.py -k 'branch or controlled_rng or pad_mask_and_training_loss_descent or python_rust_parity'
foreach ($variant in @('branch-count','branch-mass')) {
  if (Test-Path "build/checkpoints/everyday-v16-$variant.pt") { throw '既存checkpointを保護するため新しいパスを設定してください' }
  .venv/Scripts/python.exe -m training.train --config "configs/everyday-v16-$variant.json"
  .venv/Scripts/python.exe -m scripts.export_experiment --checkpoint "build/checkpoints/everyday-v16-$variant.pt" --prefix "build/everyday-v16-$variant" --output "models/everyday-v16-$variant-3mb"
}
.venv/Scripts/python.exe -m scripts.evaluate_candidates --config configs/everyday-v16-evaluation.json --output build/everyday-v16-evaluation-recheck
.venv/Scripts/python.exe -m scripts.audit_combinations --data build/everyday-v13-combination-holdout-final --checkpoint build/checkpoints/everyday-v16-branch-count.pt --checkpoint build/checkpoints/everyday-v16-branch-mass.pt --output build/v16-combination-recheck.json
```

既存結果を保護するため再実行時はcheckpoint・export・モデル配置先を変更し、評価設定とcatalogの対応も揃える。評価票を記入した後の集計はv15と共通で、入力report・mapping・ratingsのhashに紐づける。

```powershell
.venv/Scripts/python.exe -m scripts.review_candidates --report build/everyday-v16-evaluation-recheck/report.json --mapping build/everyday-v16-evaluation-recheck/review-map.json --ratings build/everyday-v16-evaluation-recheck/blind-review.csv --output build/v16-human-review.json
.venv/Scripts/python.exe -m scripts.review_novelty --report build/everyday-v16-evaluation-recheck/report.json --mapping build/everyday-v16-evaluation-recheck/review-map.json --ratings build/everyday-v16-evaluation-recheck/blind-review.csv --output build/v16-joint-human-review.json
python scripts/build_wasm.py --model-catalog configs/demo-models.json
node scripts/smoke_wasm.mjs --catalog configs/demo-models.json
$env:FUMIORI_REQUIRE_CATALOG='configs/demo-models.json'
node scripts/test_pages.cjs
python scripts/serve_demo.py
```

保存した記録:

- [学習条件・抽選trace・source hash](everyday-v16-training.json)、[全生成のコピー診断・モデルhash・抽出標本](everyday-v16-measurements.json)、[比較CSV](everyday-v16-summary.csv)
- [組合せ診断と共起検査](everyday-v16-combination-audit.json)、[CPU測定](everyday-v16-normal-timings.json)、[実ブラウザー](everyday-v16-browser-timings.json)、[FP32/INT8数値照合](everyday-v16-parity.json)
- 本文・かな・segments・失敗と全要求の時間は `build/everyday-v16-evaluation/*.raw.json`、各90要求のCPU測定は `build/v16-normal-*.json` に保持し、hashを保存した。

全Pythonテスト121件、標準モデル整合性確認、2モデル×FP32/INT8の数値照合、全10モデル×30件のWASM smoke、実Edgeで全10モデル・3難易度・標準への再切替を検証した。Rustソースは変更しておらず、Rust workspaceの全テストは今回再実行していない。独立した人間の採点、複数学習seed、新しい確認seed、同義語・係り受けの漏洩検査、公開Pagesのデプロイは未実施。
