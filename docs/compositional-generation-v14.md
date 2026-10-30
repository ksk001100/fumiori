# v14：組合せを除いた再学習と候補選別

2026-10-30。[Phase 1](compositional-generation-phase1.md)に続き、2モデルを実際に学習し、実験Aのサンプリング・正則化、実験Bの候補選別、実験Cへの入口となるprefix条件付き生成を比較した。**今回の候補は標準には採用しない。2モデルとも比較用としてデモに追加した。**

後続の[v15](compositional-generation-v15.md)では、推論時に候補を増やさず、名詞入力マスクとデータ比率を4モデルで比較した。モデルの再現率だけでなく、自然さ・読み・train非一致を同時に満たす件数を測っている。

## デモで試す

ローカルでは `python scripts/serve_demo.py` で [http://127.0.0.1:8000/](http://127.0.0.1:8000/) を開く。「組合せ実験 v14・通常CE」または「組合せ実験 v14・平滑化CE」を選び、「読み込む」→難易度を選択→「お題を生成」。標準v2、日常文v3にも切り替えられる。今回の作業環境ではこのURLで起動・HTTP応答を確認した。

一覧は `configs/demo-models.json`。ビルドとPagesワークフローは4モデルを扱い、ユーザーが選択したモデルだけを取得する。公開GitHub Pagesへのpush・デプロイは今回実行していないため、公開URLに新モデルが出ているとは扱わない。デモで試せる生成は各モデルの通常生成で、以下の尤度rerankは研究CLIだけである。

## 同じ条件で学び直した2モデル

v13の `build/everyday-v13-combination-holdout-final` を使用。train 56,593、validation 2,692、test 2,642、family-test 145。元データから評価用100対の自然側・負例側の「正規化名詞・隣接する格助詞・後続述語」の共起を全splitから除いた。今回も全splitを再走査し、対象キーの共起0件を確認した。構成語が全てtrainに残るのは96/100対で、語彙外等を除いて符号化できる91対には構成語欠落の4対は含まれない。

既存のtask学習済みcheckpointから初期化せず、両方ともseed42で同じモデル・データ抽選・5,000step・validation選択条件を使った。違いはlabel smoothing 0と0.05だけで、早期停止は無効。選択stepは両方5,000。validationの測定は平滑化なしのCEで共通である。

4層・幅128・4head・FFN384、8,802語彙ID、1,796,608パラメータ。bf16のCUDA/PyTorch 2.9.0、RTX 4070 Ti SUPERで学習した。アプリはINT8保存から復元したFP32推論で、CUDA・Python・教師を必要としない。

| 項目 | 通常CE | 平滑化CE |
|---|---:|---:|
| 学習時間 | 66.1秒 | 67.7秒 |
| peak CUDA allocated memory | 601,600,000 bytes | 601,604,096 bytes |
| 共通validation loss | 2.6373 | 2.6449 |
| 配布5ファイル合計 | 2,527,550 bytes | 2,527,550 bytes |
| 対照例で自然な側を選好 | 72 / 91 | 71 / 91 |
| 平均log odds | 3.576 | 2.795 |

学習前に小型モデルのloss低下・label smoothing/PAD・Python/Rust数値照合のスモーク3テストを実行した。新規学習はこの2つだけで、推論可能な両モデルを `models/everyday-v14-holdout-{ce,smooth}-3mb/` に配置した。テストfixtureは研究モデルとは区別する。

データはv13の親コーパスからの除去のみで、新しい外部データ・教師出力・修正文は追加していない。TatoebaのCC BY 2.0 FR、既存の自作文CC0、辞書由来のApache-2.0、合成教師の出典をmanifestとLICENSEに保持する。元データを一括CC0とはしない。[データの由来](data-licenses.md)も参照。

語彙は元コーパスのID・頻度を継承している。また共起検査は係り受け解析や同義語による意味漏洩の検出ではない。したがって「タスクの検査キーがtrainにない」と「意味的に完全に未見」を区別する。上表の確率診断も**共通prefixの最初の異なるトークン**だけで、全文の自然さや自由生成成功率ではない。診断はFP32 checkpointで行い、最終INT8保存モデルの組合せ得点を直接測ったものではない。

## 実験A/B：自由生成・サンプリング・候補選別

各条件600要求、easy/normal/hard各200、履歴0。要求iの候補jのseedを `870000 + i*4 + j` とし、候補数1/2/4は同じ候補集合の先頭1/2/4を共有する。温度やtop-pを変更する比較も同じ要求・seed・難易度。予約された確認seed830000台等には進んでいない。

通常はtemperature0.8 / top-p0.95。実験Bはモデル自身の**内容トークン＋句点＋EOSの平均log確率**が最大の候補を選ぶ。BOSと難易度予測は得点から除く。公開出力から取り除いた句点を採点時だけ復元し、読点かなを扱う。公開segmentsが語彙に厳密に戻せない場合は採点エラーとして残し、別の読みを推測しない。今回の採点エラー・生成失敗は全条件0件だった。

推論中に訓練文、近似文の索引、禁止文一覧、全文テンプレート、教師を参照しない。新規性指標のtrain索引は生成が終わった後の評価だけで使う。学生尤度は自然さ・意味・読みの正解ラベルではなく、今回はその限界を測る最小限のrerankである。

| 条件 | 固有本文 / 600 | 完全一致 | 近似一致 | 単語Self-BLEU4 | 自然さ・読みのagent点検 |
|---|---:|---:|---:|---:|---:|
| 通常CE・候補1 | 591 | 20.8% | 37.2% | 0.491 | 20 / 30 |
| 通常CE・候補2 | 584 | 33.5% | 60.3% | 0.578 | 24 / 30 |
| 通常CE・候補4 | 558 | 42.8% | 78.5% | 0.693 | 28 / 30 |
| 平滑化CE・候補1 | 594 | 22.3% | 37.0% | 0.479 | 19 / 30 |
| 平滑化CE・候補2 | 581 | 31.7% | 56.7% | 0.575 | 27 / 30 |
| 平滑化CE・候補4 | 564 | 40.8% | 75.3% | 0.677 | 24 / 30 |
| 通常CE・温度0.65・候補1 | 579 | 24.3% | 42.2% | 0.587 | 23 / 30 |
| 通常CE・top-p0.85・候補1 | 585 | 23.7% | 43.8% | 0.556 | 22 / 30 |

全条件600/600成功。完全一致はNFKC＋末尾句点のみの正規化、近似一致は文字bigram集合Dice係数0.8以上（完全一致を含む）。Self-BLEU等の詳細は[Phase 1の定義](compositional-generation-phase1.md)と共通。参照は今回学習に使ったtrainだけで、元の除去前trainやvalidation/testをコピー参照へ混ぜていない。標準v2とは学習データとモデルが異なるので、Phase 1の数値との差を特定の改善効果とは扱わない。

30要求は乱数9031で固定抽出し、全8条件で同じ要求番号を使う。条件名・候補数を隠した240行の評価票をシャッフルしてCodexが点検した。[全採点と理由](everyday-v14-blind-agent-review.csv)、[対応表](everyday-v14-review-map.json)、[hash付き集計](everyday-v14-agent-review.json)を保存。独立した人間の盲検評価ではなく、既知の文体・繰り返しから条件を推測できる限界もある。30件の小標本なので、有意な改善や採用ゲート合格の根拠にはしない。

今回の判定は通常の日常的な意味を基本とし、普通の省略・代名詞・折り畳める帽子等の解釈を許容する。一方、猫の草むしりや自分の死を聞いて喜ぶ文を、設定のない擬人化として不合格にした。童話調を許した標準v2の点検と同一の基準ではない。「料理の作り方を読書帳に書く」等、文脈を補えば成立し得る相性も安全側に判定しており、境界的な判断はCSVに残した。

読みの対応は全条件30/30と点検したが、意味が不自然な文も含む小標本であり、文脈読みを保証した結果ではない。自然さと読みを別項目に記録し、上表は両方合格した件数。[未記入の人手評価票](everyday-v14-human-review.csv)も保持する。agent採点を独立した人間の結果へ合算しない。

### 追加計算のコスト

全600要求のCPU測定中には他の検証処理も実行したため、その時間は探索値として全記録に残す。下表は検証処理の終了後に、seed873000から同じ方式でnormal 90要求ずつを取り直した値。

| CPU normal・90要求 | 通常CE p50 / p95 | 平滑化CE p50 / p95 |
|---|---:|---:|
| 候補1 | 28.9 / 39.6ms | 29.6 / 43.0ms |
| 候補2＋採点 | 65.6 / 85.0ms | 66.2 / 93.8ms |
| 候補4＋採点 | 133.4 / 157.7ms | 133.2 / 176.2ms |

Windows x86_64、Intel Family 6 Model 183、release/Candle FP32。同じpoolの実測した候補生成時間と全文logits採点時間を足した診断で、モデルload、レポートJSON保存、選択の管理処理を含めない。候補1には使わなかった採点時間を加えない。厳密なエンドツーエンドのWeb性能ではない。少数の探索測定で他端末への100ms保証もしない。

Edge 156.0.4314.50の実Web Workerでは、通常生成normal各30件、モデル既定の履歴16で、通常CE p50 17.5ms / p95 23.1ms、平滑化CE 16.8ms / 22.3msだった。ロード・通信・描画は含めず、同じseed42で履歴と乱数状態を保持するデモ操作の測定。候補数4のrerankをブラウザーへ実装した測定ではない。

### 採否

label smoothingは今回のvalidation・組合せ診断・30件点検で安定した改善を示さず、採用しない。低温度／狭いtop-pにも自然さの点検上の差はあるが、コピーや重複が増え、30件だけでは最適設定と判断できない。

通常CEの候補4は点検28/30まで上がった一方、完全一致42.8%、近似一致78.5%、固有本文93.0%へ悪化し、normal CPU p95も100msを超えた。平滑化CEでは候補2→4で点検も下がる。**尤度だけを高める選別は、新規性・多様性・速度を同時に満たす改善策にはならなかった。** この選別方式を標準生成やデモの既定には追加しない。

## 実験Cの小規模な実現可能性確認

`scripts/probe_combination_generation.py` は91対から「名詞＋格」のprefixだけを与え、各FP32 checkpointでgreedyに続きを生成する。正解の述語を強制せず、全文テンプレートを使わない。難易度は自然側の文から決めて両モデルで固定する。全91件ずつEOSへ到達したが、EOS到達を意味や読みの合格にはしない。

| 条件prefix | 通常CEの続き込みの出力 | 平滑化CEの続き込みの出力 |
|---|---|---|
| 水を | 水を注ぐ | 水を注ぐ |
| ご飯を | ご飯を作る | ご飯を食べる |
| 顔を | 顔を取って下さい | 顔を出す |
| 公園で | 公園でブランコに乗る | 公園でブランコに乗る |
| 桜が | 桜が満開だった | 桜が満開だ |

「顔を取って下さい」のような意味の破綻も残る。prefixから自然な別の述語へ進むことと、除去した特定の組合せを生成することは別であり、参照文と異なるだけで誤りとしない。この生成はPython/FP32の診断で、Rust/WASMへの意味条件APIや主語・目的語・場所・時制の完全な制御は未実装。今回の点検だけで意味条件付きモデルの完成とはしない。

次は、自然な代案を残しつつ意味の相性を区別する学習信号の小実験を、同じ除去splitで比較する必要がある。確認例を教師ラベルや修正学習へ転用せず、独立した組合せと複数学習seedで再確認する。今回の結果だけで大型化や追加教師APIへ進まない。

## 再現手順

既存の学習venv、v11の準備データ、CUDAが必要。v13データがない場合は次のコマンドで準備する。既存ディレクトリを上書きしない。

```powershell
.venv/Scripts/python.exe -m scripts.prepare_combination_holdout --output build/everyday-v13-combination-holdout-final
```

学習前に `configs/everyday-v14-holdout-{ce,smooth}.json` のcheckpointが存在しないことを確認する。再実行時はconfigをコピーしてcheckpointを新しいパスに変更し、下記exportのcheckpoint引数も揃える。

```powershell
.venv/Scripts/python.exe -m pytest -q training/tests/test_model.py -k 'pad_mask_and_training_loss_descent or regularization_is_training_only or python_rust_parity'
foreach ($variant in @('ce', 'smooth')) {
  if (Test-Path "build/checkpoints/everyday-v14-holdout-$variant.pt") { throw '既存checkpointを保護するため新しい出力パスを設定してください' }
  .venv/Scripts/python.exe -m training.train --config "configs/everyday-v14-holdout-$variant.json"
  .venv/Scripts/python.exe -m scripts.export_experiment --checkpoint "build/checkpoints/everyday-v14-holdout-$variant.pt" --prefix "build/everyday-v14-holdout-$variant" --output "models/everyday-v14-holdout-$variant-3mb"
}
cargo build --release --locked -p fumiori-cli
.venv/Scripts/python.exe -m scripts.evaluate_candidates --output build/everyday-v14-evaluation-recheck
.venv/Scripts/python.exe -m scripts.audit_combinations --data build/everyday-v13-combination-holdout-final --checkpoint build/checkpoints/everyday-v14-holdout-ce.pt --checkpoint build/checkpoints/everyday-v14-holdout-smooth.pt --output build/v14-combination-recheck.json
.venv/Scripts/python.exe -m scripts.probe_combination_generation --audit build/v14-combination-recheck.json --checkpoint build/checkpoints/everyday-v14-holdout-ce.pt --checkpoint build/checkpoints/everyday-v14-holdout-smooth.pt --output build/v14-prefix-recheck.json
```

exportは既存のFP32・INT8・runtime・配置先を拒否する。別名で再学習・exportした場合は評価設定のモデルパス・対応するexportパスとデモcatalogも更新する。実験モデルを追加したまま、標準の切替は行わない。

```powershell
# 記入した評価票を条件へ戻し、入力hashとともに集計する
.venv/Scripts/python.exe -m scripts.review_candidates --report build/everyday-v14-evaluation-recheck/report.json --mapping build/everyday-v14-evaluation-recheck/review-map.json --ratings build/everyday-v14-evaluation-recheck/blind-review.csv --output build/v14-human-review.json
python scripts/build_wasm.py --model-catalog configs/demo-models.json
node scripts/smoke_wasm.mjs model-int8 model-experimental model-v14-ce model-v14-smooth
$env:FUMIORI_REQUIRE_EXPERIMENTAL='1'
$env:FUMIORI_REQUIRE_CATALOG='configs/demo-models.json'
node scripts/test_pages.cjs
python scripts/serve_demo.py
```

WindowsでPlaywrightを別のGit管理外ディレクトリへ導入する場合、`npm install --prefix build/browser-tests --no-save --package-lock=false playwright@1.56.1` の後、`$env:NODE_PATH=(Resolve-Path build/browser-tests/node_modules).Path` を設定する。今回は `$env:FUMIORI_BROWSER_CHANNEL='msedge'` を指定した。CIはワークフロー通りChromiumを導入する。モデル追加時はPagesのスモーク対象とcatalog必須検査も更新する。

## 保存した記録と検証

- [学習条件・log・checkpoint hash](everyday-v14-training.json)、[生成・コピー診断・各入力hash](everyday-v14-measurements.json)、[比較CSV](everyday-v14-summary.csv)
- [組合せ確率と全splitの共起監査](everyday-v14-combination-audit.json)、[prefix生成全件](everyday-v14-prefix-generation.json)
- [独立したnormalのCPU測定](everyday-v14-clean-normal-timings.json)、[実ブラウザー測定](everyday-v14-browser-timings.json)
- 全候補とかな・segments・生成／採点時間は `build/everyday-v14-evaluation/*.raw.json` に保存。これらのhashをJSONへ保持し、成功例だけを残さない。
- Rust fmt・clippy・workspace test（34件）、全Pythonテスト113件、標準モデル整合性確認、2モデル×FP32/INT8の数値照合、WASM各モデル30件、実Edgeで全4モデル・3難易度・標準への再切替を検証した。数値照合はそれぞれ88,020値、許容絶対差0.0002以内。

独立した人間の評価、複数学習seed、同義表現・係り受けによる漏洩検査、完全な意味slot制御、rerankのWASM実装、公開Pagesへのデプロイは未実施。
