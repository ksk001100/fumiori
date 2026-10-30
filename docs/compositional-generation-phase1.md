# 構成的一般化の研究：現状分析とPhase 1

2026-10-30。今回の範囲は、依頼の最初の段階である現状調査、評価基盤の追加、標準モデルのベースライン測定。Phase 2〜5の学習・A/B比較・意味条件付き生成は下記の計画に残る。モデル改善や未見への転移を達成した結果ではない。

## 現行実装の分析

| 項目 | 現状と根拠 |
|---|---|
| 標準モデル | `models/stories-v2-diverse-3mb`、2,624,832パラメータ。配布5ファイルは2,903,289 bytes。小型の日常文v3は比較用で、標準ではない |
| アーキテクチャ | GPT-2型、pre-LayerNorm、causal attention、GELU、共有入出力埋め込み。4層・幅224・8head・FFN672・最大64位置。`training/model.py` と配布configで確認 |
| トークナイザー | 2,644 ID。既存の表記・かな・品詞を一組の語彙IDに保持する。標準モデルの読みは別モデルの生成結果ではない。次期試作にはSentencePiece＋語彙trie、Llama型もあるが標準と混同しない |
| データ | 自作対応付き散文2,500文＋SimpleStories-JAの生成可能な28,300文。train 30,800、validation 800、test 800、family-test 300。自作部分CC0、SimpleStories部分MIT、取得revision・原本hashはmanifestに保持 |
| 前処理 | 明示語彙・限定した活用形で対応付け。未知漢字、生成不能な文字／固有名詞トークン、問題のある境界等を除去。元物語hashによるsplitと既存の本文重複除去を使用。全体の独立した人間の点検はない |
| 学習 | next-token CE、PAD無視、AdamW。seed42、最大10,000step、batch128、lr0.001、weight decay0.1、label smoothing0.02、dropout0.1。warmup200＋cosine、難易度バランス、名詞context dropout0.1、lexical balance0.5・最大比3。validationで選択。設定は `configs/stories-v2-diverse-3mb.json` |
| 生成 | 語彙IDの自己回帰、temperature0.8 / top-p0.95、通常は履歴16。かな長・終端・語境界・局所的な読み補正を適用。Top-kは現在の公開オプションにない。学習文リストや全文テンプレートを推論に渡さない |
| 読みの限界 | segmentの連結・かな長・助詞の正書法を構造検証する。多義語の文脈解決は一般には保証しない。辞書一致・構造検証・今回の読み点検は別物 |
| 実行環境 | アプリはCandle CPU/WASMのFP32演算。INT8は保存形式で、読み込み時に復元。Python/PyTorch/CUDAはアプリには不要。KV cacheは未実装でprefixを再計算する |
| 既存評価 | Rustの構造・モデル破損・数値照合、Pythonのデータ／学習テスト、完全一致・本文新規率・語彙集中度、独立seed測定、Web Workerのブラウザーテスト。自然さ・読みの既存点検はCodexによるもの |
| 既存の組合せ研究 | v13に100対、共起検査、除去splitと学習済み重みの初期化拒否がある。91対の確率診断は第一分岐のみ。自然側86種類が元trainにあり、未見評価ではない。除去後の構成語被覆96/100 |

詳しい既存設計は[architecture](architecture.md)、[多様性研究](diversity.md)、[v11](compact-generation-research.md)、[v12](compact-repair-research.md)、[v13](combination-generalization-research.md)、[容量と速度](lightweight-generation.md)を参照する。古いv1説明の10万文を現行v2の学習件数として扱わない。

## 研究から採用する評価方針

- [Keysers et al. (2020)](https://arxiv.org/abs/1912.09713)の構成要素と組合せを分ける考え方に沿い、語彙の未出現と既知語の未出現ペアを別集計する。今回の表層bigram集計はDBCA/MCDの再現ではない。
- [Xu et al. (2023)](https://aclanthology.org/2023.findings-emnlp.623/)は未知の述語組合せに対する忠実性を扱う。意味条件付き生成へ進む場合、自然さと別に引数の欠落・追加・役割の入れ替わりを採点する。
- [McCoy et al. (2023)](https://aclanthology.org/2023.tacl-1.38/)を参考に、全文一致だけでなくn-gramと連続コピー長を調べる。今回の簡易実装はRAVEN全体の再現ではなく、意味的な言い換えの検出器でもない。
- [Lee et al. (2022)](https://aclanthology.org/2022.acl-long.577/)を参考に、重複除去とsplitの重複を点検する。別規模・別言語での効果を、このモデルの改善実績として転用しない。
- [Zhang et al. (2021)](https://aclanthology.org/2021.humeval-1.3/)を参考に、サンプリングを変えた結果は品質と多様性を並べて比較する。今回、サンプリングの最適化や蒸留の効果はまだ測っていない。

## 追加した評価基盤

`scripts/novelty_metrics.py` は標準ライブラリだけで使用できる。`scripts/novelty_evaluate.py` は既存のrelease CLIを呼び、JSON設定の条件で生成する。実行器は既存のruntime identity検査を再利用するため、学習用venvを使う。

| 指標 | 実装した定義と限界 |
|---|---|
| Exact Match Rate | NFKCと末尾「。」だけを正規化した訓練本文との一致。時刻・副詞・活用は消さない。既存の `core_key` 新規率とは別指標 |
| Near-Duplicate Rate | 訓練文に対する最大文字bigram集合Dice係数が0.8以上。完全一致を含む割合と、完全一致以外の件数を保存。候補の近似打切りはしない。意味的な言い換えには対応しない |
| Longest Common Substring | 1つの訓練文との最長の連続部分文字列、文字数、生成文に占める割合。文境界をまたがず、部分列とは区別。接尾辞オートマトンを使用 |
| n-gram Novelty | 文字と既存の語彙区切りのn=1〜4。出現回数ベースと種類ベースを分け、分母0はnull。単語版は記号を除外し、文字IDを単語扱いしない。活用形は別の語 |
| 既知語の未知組合せ | 構成する各語がtrainにあるbigramに限定し、未出現bigramの割合を別集計。単純な隣接語診断であり、名詞・格・述語の意味一般化ではない |
| Self-BLEU | 自分の観測1件だけを除いた全出力を参照とする文BLEU4の平均。重複した他の出力は残す。短文はeffective order、2-gram以上はadd-one平滑化、0〜1。標本数・難易度を揃えて比較する |
| Naturalness | 自動の意味・自然さ判定器は追加しない。固定抽出した評価票で0/1と理由を記録。未記入はnullで、agentと独立した人間の評価を分ける |
| Reading Accuracy | 上記評価票で本文とかなを文脈に沿って別採点。構造roundtrip件数は独立した項目。多義語・曖昧な読みを無条件に正解にしない |
| Inference Latency | 既存CLIの各要求の生成・シリアライズ・結果記録時間。失敗と初回を含むp50/p95。モデルload・通信・評価コードは含まない。WASMや厳密なウォーム測定ではない |

語彙・重み・config・manifest・LICENSE、コーパスの全split、CLI・Rustソース、評価コード・設定、raw生成結果のSHA-256を記録する。配布モデルに評価用コーパスや索引は追加しない。全指標のコピー参照はtrainだけで、validation/testの文を参照へ混ぜない。将来、蒸留や修正文を使う比較では、追加の全学習ソースも参照に含める拡張が必要。

生成失敗を隠さず、要求数・成功数・失敗seedを保存する。新規性の率は返ったモデル出力を分母にするため、成功率と必ず併記する。今回の全要求は成功。失敗も含む要求番号から事前に固定した乱数で評価対象を選び、品質や新規性を理由とする選別をしない。

## データ・分割の再現

今回、新しい外部データの取得・転載や学習splitへの文の追加はない。標準モデルの既存準備データを再利用し、manifestのSHA-256と全split・語彙のhashを検証する。設定にモデル重みとデータmanifestのhashを固定した。準備データの再現は[training README](../training/README.md)と[多様性の再現手順](diversity.md)に従う。元データの取得ライセンスは[data-licenses](data-licenses.md)を参照。

train内重複と、4splitの6通りの間の完全一致は今回の正規化でもすべて0だった。これは言い換え、近似文、文型、意味関係の漏洩ゼロを意味しない。既存準備コードの物語hash分割は継承するが、今回の新しい検査で上流物語の分離を独立に再証明したとはしない。

構成的一般化の専用セットは `data/combinations-v13/pairs.tsv` と既存の `scripts.prepare_combination_holdout` を再利用する。再現コマンドは[v13記録](combination-generalization-research.md#再現と検証)。除去後の構成語被覆を満たす例だけを既知語の未見組合せとして採点し、語彙外・構成語欠落を別記する。語彙自体に元コーパスの情報が残る点、同義語・係り受けの漏洩が未解決な点は引き継ぐ。既存のv2/v11/v12重みをこのsplitの未見評価へ流用しない。

## 実行と評価票

リポジトリルートで実行する。既存の結果を上書きしないので、新しい出力先を指定する。

```powershell
cargo build --release --locked -p fumiori-cli
.venv/Scripts/python.exe -m pytest -q training/tests/test_novelty_metrics.py
.venv/Scripts/python.exe -m scripts.novelty_evaluate run --config configs/novelty-baseline-v1.json --output build/novelty-baseline-recheck
# baseline-mixed/review.csvの全行を残し、評価者が記入した後に実行
.venv/Scripts/python.exe -m scripts.novelty_evaluate review --report build/novelty-baseline-recheck/baseline-mixed/report.json --ratings build/novelty-baseline-recheck/baseline-mixed/review.csv --output build/novelty-baseline-recheck/human-review.json
```

`natural` / `reading_ok` は0/1、`reviewer` は評価者名、`reviewer_kind` は `independent_human` または `agent`、`notes` は判断理由を記入する。未記入の行は未評価として残す。改変された本文・かな、標本hash、欠落行、重複行、失敗への採点、評価者の属性なしの採点を拒否する。集計JSONは入力レポートと評価票のhashに紐づける。

自然さは単独のお題としての文法・語の相性・意味を点検する。普通の主語／目的語省略と童話調・日常的な比喩は許容するが、接続の崩れ、文末の不成立、意味の不整合、無意味な反復は不合格にする。現行標準モデルが童話調であるため、日常文専用モデルと同じ評価基準だとは扱わない。読みは助詞「は・へ・を」、読点、漢字の文脈を確認し、曖昧な場合は理由付きで安全側の0とする。

各条件の `raw.json`、`report.json`、`review.csv` と全条件の `summary.csv` を保存する。品質点検前の固定抽出はseed9021、混合条件は600件から100件。seed860000〜860599を開発用に使用し、normal探索測定は861000〜861089。既存の予約済み確認seed830000台等は使わない。

## Phase 2〜5の実装計画

| 段階 | 既存実装を使う作業と比較 | 先へ進むための条件 |
|---|---|---|
| Phase 2 / 実験A | まず同じ600seed・履歴0でtemperatureとtop-pを1因子ずつ変更。Top-kは回帰試験付きの研究オプションとして必要時に追加する。既存の重複除去／生成可能性フィルターを起点に近似文の頻度とデータ品質を監査し、効果の見込みがあれば同じ初期値・学習量で再学習を比較 | 自然さと読みを並べて点検。未知率だけで設定を選ばない。近似除去は曖昧な読みや自然な代案を失わないか確認。予約した確認seedは使わない |
| Phase 3 / 実験B | 同じ学生モデルの候補数1/2/4を比較。まず学生の長さ正規化尤度と構造・語境界の失敗を診断する小実験から始める。尤度を自然さの正解ラベルにしない。学習文照合・修正表・全文一覧を実行時の選別に使わない。候補生成全体の時間、失敗・重複、盲検の品質差を保存 | 品質を改善し、normal p95と配布容量を同時に満たす見込みがあること。新規性は事後のoffline評価に限る。候補を増やすだけで100msを超える案は採用しない |
| Phase 4 / 実験C | 既存v13の格・述語キーで意味条件の被覆と漏洩を再点検。現行公開APIは難易度以外の意味条件を持たないため、最初は研究用のprefix条件／尤度診断を比較する。完全なslot制御が必要なら学習専用control tokenの小規模なfrom-scratch実験を分ける | 100対の既存重み診断を未見生成と呼ばない。構成語がtrainに残ることと全学習ソースでの組合せ除去が前提。語順を固定したテンプレート生成で代用しない。意味条件の忠実性と自由生成を別評価 |
| Phase 5 | 有望なA/B/Cだけを同じデータ・サンプリング・難易度・履歴で比較し、必要なら複数学習seed。既存の採用ゲート、5ファイルの整合性、モデルを追加する場合のWeb選択UI・WASM smoke・実ブラウザー確認を通す | 容量3,000,000 bytesとnormal 100msを品質と同時に評価。独立した人間の点検を募集し、少数のagent採点差で改善扱いしない。標準切替は既存の開発・確認ゲート通過後 |

大規模な学習前には、新しい出力先・checkpoint名で少量データ／数stepの学習→export→Rust読み込みを確認する。既存checkpointや実験成果を上書きしない。蒸留はPhase 2/3の効果と速度・容量の限界を見てから判断し、現時点では外部教師APIや追加データを必要条件にしない。

読みの自動棄却は、辞書一致だけで正解にせず、再現可能な曖昧性の判定が設計できてから回帰例とともに比較する。解消できない候補は再試行し、全候補が失敗すればエラーを返す。今回、その推論挙動の変更は実施していない。

今回の実測・評価票・失敗例は[Phase 1結果](compositional-generation-phase1-results.md)を参照する。
