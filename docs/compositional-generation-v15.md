# v15：名詞入力マスクとデータ比率の比較

2026-10-30。[v14](compositional-generation-v14.md)では、尤度で候補を選ぶと自然さの点検結果が上がる一方、コピーと重複が増えた。今回は推論時の計算を増やさず、学習中の名詞入力マスクとデータ比率を2×2で比較した。**自然で読みが合い、訓練文と非一致の出力を増やす改善は確認できず、4モデルとも標準には採用しない。比較用として全てデモに追加した。**

後続の[v16](compositional-generation-v16.md)では、比率変更モデルを対照に、行数／実際の抽選質量に基づく分岐補正を比較した。既存の不変な出力・評価票をhashで確認して再利用し、独立した新規採点とは区別している。

## デモ

[ローカルデモ](http://127.0.0.1:8000/)で「v15 比較基準」「v15 名詞入力マスク」「v15 データ比率変更」「v15 比率変更＋入力マスク」を選択し、「読み込む」→難易度→「お題を生成」で比較する。サーバーを起動していない場合は `python scripts/serve_demo.py` を実行する。

catalogには標準v2、比較用v3、v14の2種類、今回のv15の4種類を保持する。標準は引き続きv2。全8モデルを含めて再ビルドし、WASMと実ブラウザーで選択・読み込み・3難易度生成・標準への再切替を検証した。公開GitHub Pagesへのpush・デプロイは実施していない。

## 仮説と比較条件

名詞の入力を学習中だけ10%の確率でUNK IDへ置き換え、元の文をnext-token正解として保つ既存の `noun_context_dropout` を使った。単語や読み・正解文を作り替えず、欠けた文脈から続きを予測する正則化がコピーを抑えるか調べる。意味的な知識を追加する手法や、新しい蒸留方式を実装したものではない。推論時にマスクや追加辞書は使わない。

もう一つの因子は既存コーパスの抽選比率。新しいデータ・教師生成・修正文・単語交換文は追加していない。「比率変更」を文章品質の検証済みフィルターと呼ばない。

| 条件 | Tatoeba | 既存教師合成文 | 自作日常文 | 自作paired文 | 名詞入力マスク |
|---|---:|---:|---:|---:|---:|
| 比較基準（base） | 65% | 15% | 15% | 5% | 0 |
| 入力マスク（context） | 65% | 15% | 15% | 5% | 0.1 |
| 比率変更（quality） | 25% | 45% | 25% | 5% | 0 |
| 比率＋マスク（quality-context） | 25% | 45% | 25% | 5% | 0.1 |

比率は難易度バランス内のcorpus mixture。validationの選択比率は全条件で65/15/15/5に固定し、比率変更したモデルだけ別のvalidation基準で有利に選ばないようにした。内部名 `quality` は実験の識別名で、品質改善を達成したラベルではない。

データはv14と同じv13の組合せ除去split。train 56,593、validation 2,692、test 2,642、family-test 145。語彙8,802 ID、4層・幅128・4head・FFN384、1,796,608パラメータ、seed42、各5,000step・batch128。全条件をランダム初期値から学び、既存checkpointからの初期化や修正補助データは使わない。

## 乱数と比較の検証

従来の学習では入力マスクを足すと同じglobal RNGを進め、次の学習文の抽選とモデルdropoutまで変わる。今回は `independent_data_rng: true` を明示し、学習文抽選seed143、入力マスクseed144、モデルdropoutの元のglobal RNGを分けた。この指定がない既存設定は従来の乱数経路を保つ。

初期stateと、更新に使う5,000batchの行番号列をSHA-256で記録した。全4モデルの初期state hashは共通で、通常比率の2条件同士、および比率変更の2条件同士の抽選hashも一致した。比率の異なる条件では抽選する文が変わるため、同じ行列とは主張しない。hashはモデル・語彙・入力hashと[学習記録](everyday-v15-training.json)に保存した。

同じ構成・学習量でも、v14は学習文抽選の乱数経路が異なるので、v14からの数値変化を入力マスク単独の効果と扱わない。今回の比較はv15の中で行う。

学習前の6テストで、小型モデルのloss低下・Python/Rust数値照合、名詞マスクが入力だけを変えること、乱数分離、初期重みhashを確認した。

## 学習と組合せ診断

PyTorch 2.9.0 / CUDA 12.8、bf16、RTX 4070 Ti SUPER。選択stepは全条件5,000。アプリの推論は引き続きCandle CPU/WASMのFP32で、INT8は保存形式。PythonやCUDAをアプリへ追加していない。

| 条件 | 共通validation loss | 学習時間 | 自然な側を選好した対 / 91 |
|---|---:|---:|---:|
| 比較基準 | 2.6296 | 68.3秒 | 74 |
| 入力マスク | 2.6240 | 69.9秒 | 73 |
| 比率変更 | 2.8599 | 70.8秒 | 67 |
| 比率＋マスク | 2.8475 | 71.6秒 | 67 |

組合せ診断はv13の100対のうち語彙に符号化できる91対について、共通prefixの最初に異なるトークンのlog確率をFP32 checkpointで測る。全splitの対象共起キー0件を再確認した。語形・格・後続動詞による検査であり、係り受けや同義語による意味漏洩は保証しない。語彙自体は元コーパスの情報を引き継ぐ。

以前から開発に使っている対照例を繰り返し診断しており、独立した新しい確認セットの結果ではない。全文の自然さ、自由生成の正答率、最終INT8モデルの対照得点とも区別する。今回はこの診断でも安定した改善を示さなかった。

## 自由生成の新規性・多様性

各条件600要求、easy/normal/hard各200、temperature0.8 / top-p0.95、履歴0、候補1。同じ要求番号iに `880000 + i*4` の独立seedを使う。候補は増やさず、訓練文との一致や品質を理由に出力を取り除かない。全4条件600/600成功。

| 条件 | 異なる本文 / 600 | 訓練文と完全一致 | 近似一致 | 単語Self-BLEU4 | 既知語のみの未出現bigram |
|---|---:|---:|---:|---:|---:|
| 比較基準 | 593 | 23.5% | 38.2% | 0.477 | 3.91% |
| 入力マスク | 592 | 17.7% | 34.2% | 0.485 | 3.78% |
| 比率変更 | 549 | 67.5% | 84.7% | 0.619 | 1.22% |
| 比率＋マスク | 546 | 65.2% | 84.8% | 0.656 | 1.20% |

完全一致はNFKC＋末尾句点のみの正規化。近似一致は文字bigram集合Dice係数0.8以上で、完全一致を含む。参照は今回のtrainだけで、validation/testをコピー参照に混ぜない。文字／単語n-gram、LCS、全件の近似訓練文は[測定JSON](everyday-v15-measurements.json)に保存した。これらは意味的な新規性や自然さの保証ではない。

入力マスクは完全一致を減らしたが、異なる出力や既知語の未出現bigramが一貫して増えたわけではない。比率変更は少量の既存合成文・自作文を繰り返し学ぶ割合を増やし、コピーと同じような文への集中が大きくなった。

## 自然さ・読みと「自然な非一致文」

生成を見る前に設定した乱数9041で各条件から同じ100要求を選び、条件名を隠して400行をシャッフルした。本文・かなが同一の行は一度点検し、同じ判定を反映したため、異なる本文・かなの組は379件。Codexによる点検であり、独立した人間の盲検評価ではない。定型的な文体から条件を推測できる限界もある。

通常の日常的な意味、普通の省略・口語、比喩の理解を許容し、格・活用・意味の破綻、解消できない読みは安全側で不合格とした。珍しさや代名詞だけで意味が不可能と決めつけない。読書帳に作業手順を書く例など、文脈を補えば成立し得る境界例も理由を保存している。

| 条件 | 自然さ / 100 | 読み / 100 | 両方合格 / 100 | 非一致かつ両方合格 / 100 | 非一致の点検件数 |
|---|---:|---:|---:|---:|---:|
| 比較基準 | 67 | 98 | 66 | **44** | 77 |
| 入力マスク | 60 | 100 | 60 | **44** | 84 |
| 比率変更 | 88 | 98 | 87 | **25** | 37 |
| 比率＋マスク | 88 | 98 | 87 | **29** | 40 |

新設したoffline集計 `scripts/review_novelty.py` は、明示的な自然さ・読みのラベルとtrainの完全／近似一致を結合する。新規でも不自然な文、自然でもコピーの文を、「自然で読みが合う非一致文」の件数には数えない。今回の全100件にラベルがあるが、未記入を正解で埋めず、agentと独立した人間の集計を分ける。

比較基準と入力マスクはどちらも44/100で、コピー率だけの改善は目的の達成につながらなかった。比率変更で自然さは上がったが、点検した非一致文は少なく、自然な非一致文も25/100に減った。近似一致も除いた両方合格は、順に32/100、28/100、8/100、9/100だった。

この「非一致」はtrainに同一文字列がないという意味で、完全に新しい意味・文型の証明ではない。1つの学習seedと100件の主観的な点検だけで小さな差を有意な改善と呼ばない。[全400行の採点と理由](everyday-v15-blind-agent-review.csv)、[379組の判定](everyday-v15-unique-agent-review.json)、[対応表](everyday-v15-review-map.json)、[集計](everyday-v15-agent-review.json)、[品質と非一致の結合集計](everyday-v15-joint-review.json)を保存。[人手評価用の未記入票](everyday-v15-human-review.csv)も別に残す。独立した人間の評価は未実施。

### 残った失敗

| 本文／かな | 問題 |
|---|---|
| 彼女は公園を延期した | 動詞と目的語の意味が合わない |
| 桜の葉が満開だ | 葉と満開の相性が不自然 |
| この本はあなたのですかと思いますか | 疑問と引用の接続が崩れる |
| 電車は空いているな／でんしゃはあいているな | 混雑が少ない意味では通常「すいている」 |
| この前、友達とカフェに行く予定だ／このぜん、… | 「この前」の誤読と時制の問題 |
| 夕暮れ時になると、空はオレンジ色に染まる／ゆうぐれときに… | 通常「ゆうぐれどき」と読む |
| 彼女は何を犠牲にしてもしすぎることはない／かのじょはなんを… | 文の構成と「何を／なにを」の読みが崩れる |

「金／きん」は金銭なら「かね」で、本文だけでは金属の文脈を確定できないため読みを保留・不合格とした。文脈があれば可能な読みを全て誤読と断定したものではない。構造的な連結が合っていても、こうした文脈読みの問題は残る。

## 容量と速度

全4モデルは重み・語彙・config・manifest・LICENSEの合計**各2,527,550 bytes**。推論に入力マスクやデータ表は加えない。

CPUは他の学習・検証処理が終わった後に、seed883000から同じ方式でnormal各90要求、履歴0を取り直した。Windows x86_64、Intel Family 6 Model 183、release/Candle FP32。下表は生成部分で、モデルloadや、候補1では使わない全文logits採点時間を除く。初回・再試行は含み、厳密なウォームのみの計測ではない。

ブラウザーはEdge 156.0.4314.56、実Web Worker、normal各30件、seed42の継続乱数と既定の履歴16。モデルload・通信・描画は除く。CPUとはseed・履歴・計測系が異なり、その差を速度改善の証拠にしない。

| 条件 | CPU p50 / p95 | ブラウザー p50 / p95 |
|---|---:|---:|
| 比較基準 | 30.9 / 46.4ms | 17.1 / 21.7ms |
| 入力マスク | 28.1 / 56.0ms | 16.2 / 21.9ms |
| 比率変更 | 30.0 / 40.8ms | 14.4 / 19.4ms |
| 比率＋マスク | 31.2 / 44.7ms | 14.3 / 21.7ms |

今回の測定p95は100ms以内だったが、少数の探索測定であり、他端末や長時間運用での上限の保証ではない。600要求の生成と一部の検証処理を同時に行った際の時間は探索値として全記録に保存し、上表には使っていない。

## 採否と次の仮説

全4モデルとも標準には採用しない。入力マスクは、学習文の再現を減らすだけでは自然な非一致文を増やせない。既存の少量の合成文・自作文へ抽選を偏らせるだけでも、自然さとコピーの交換になる。容量・速度だけを満たすことを品質達成とは扱わない。

次は、自然な文の異なる続きを十分な種類で学べるデータが不足しているか、既存train内の意味・語の相性を伝える信号が不足しているかを分けて調べる。新しい教師文や修正例を作る場合は、今回の評価出力を学習に流用せず、train側の独立した例を使い、同じ組合せ除去と漏洩検査を適用する。負例の自動的な単語交換は、自然な別解まで誤って禁止する危険があるため、その妥当性を検査してから小規模に比較する。

## 再現と記録

設定は `configs/everyday-v15-{base,context,quality,quality-context}.json`。学習環境・データの準備は[v14](compositional-generation-v14.md#再現手順)を参照。再実行する場合、既存checkpoint・export・配置先を保護するため、新しい出力パスを設定する。

```powershell
.venv/Scripts/python.exe -m pytest -q training/tests/test_controlled_rng.py training/tests/test_model.py -k 'controlled_rng or noun_corruption or pad_mask_and_training_loss_descent or python_rust_parity'
foreach ($variant in @('base','context','quality','quality-context')) {
  if (Test-Path "build/checkpoints/everyday-v15-$variant.pt") { throw '既存checkpointを保護するため新しい出力パスを設定してください' }
  .venv/Scripts/python.exe -m training.train --config "configs/everyday-v15-$variant.json"
  .venv/Scripts/python.exe -m scripts.export_experiment --checkpoint "build/checkpoints/everyday-v15-$variant.pt" --prefix "build/everyday-v15-$variant" --output "models/everyday-v15-$variant-3mb"
}
.venv/Scripts/python.exe -m scripts.evaluate_candidates --config configs/everyday-v15-evaluation.json --output build/everyday-v15-evaluation-recheck
.venv/Scripts/python.exe -m scripts.audit_combinations --data build/everyday-v13-combination-holdout-final --checkpoint build/checkpoints/everyday-v15-base.pt --checkpoint build/checkpoints/everyday-v15-context.pt --checkpoint build/checkpoints/everyday-v15-quality.pt --checkpoint build/checkpoints/everyday-v15-quality-context.pt --output build/v15-combination-recheck.json
```

再実行でモデル名を変えた場合は、対応するexportパス、評価設定のモデルパス、デモcatalogも揃える。評価票は同じ要求を固定抽出したまま全行を保持し、記入後に集計する。

```powershell
.venv/Scripts/python.exe -m scripts.review_candidates --report build/everyday-v15-evaluation-recheck/report.json --mapping build/everyday-v15-evaluation-recheck/review-map.json --ratings build/everyday-v15-evaluation-recheck/blind-review.csv --output build/v15-human-review.json
.venv/Scripts/python.exe -m scripts.review_novelty --report build/everyday-v15-evaluation-recheck/report.json --mapping build/everyday-v15-evaluation-recheck/review-map.json --ratings build/everyday-v15-evaluation-recheck/blind-review.csv --output build/v15-joint-human-review.json
python scripts/build_wasm.py --model-catalog configs/demo-models.json
node scripts/smoke_wasm.mjs --catalog configs/demo-models.json
$env:FUMIORI_REQUIRE_CATALOG='configs/demo-models.json'
node scripts/test_pages.cjs
```

WASM smokeもcatalogから全モデルを読み取るようにした。今後モデルを追加した際、手書きのモデルID列の更新漏れでスモーク対象から外れないようにする。WindowsのPlaywright導入とEdge指定は[v14の手順](compositional-generation-v14.md#再現手順)に従う。

保存した成果物:

- [学習設定・log・乱数trace・checkpoint hash](everyday-v15-training.json)、[生成・コピー診断・全入力hash](everyday-v15-measurements.json)、[比較CSV](everyday-v15-summary.csv)
- [組合せ診断と共起検査](everyday-v15-combination-audit.json)、[CPU測定](everyday-v15-normal-timings.json)、[実ブラウザー測定](everyday-v15-browser-timings.json)、[数値照合](everyday-v15-parity.json)
- 全600要求の本文・かな・segmentsと失敗情報は `build/everyday-v15-evaluation/*.raw.json`、初回を含む各90要求のCPU測定は `build/v15-normal-*.json` に保存し、そのhashを記録した。

全Pythonテスト119件、標準モデル整合性確認、4モデル×FP32/INT8の数値照合、全8モデル×30件のWASM smoke、実Edgeの全モデル・3難易度生成を検証した。ローカルURLのHTTP 200と全8選択肢も確認した。Rustソースは変更しておらず、今回Rust workspaceの全テストは再実行していない。独立した人間の採点、複数学習seed、同義語・係り受けの漏洩検査、完全な意味条件API、公開Pagesへのデプロイは未実施。
