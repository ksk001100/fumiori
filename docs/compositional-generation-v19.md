# v19：学習データの整理と生成品質の比較

2026-10-30。[監査](training-data-diversity-audit.md)で見つかった近似文・構文集中と、trainで確認した読み・修飾の問題を整理した。**研究用trainを59,111→57,354件へ整理し、自作日常文の近似率は標本で82.4→64.4%に低下した。ただし生成の自然な非一致文には安定した改善がなく、標準モデルは維持する。**

## 整理済みデータとデモ

今後比較に使える整理済みデータは `build/everyday-v19-refined-data/`。元の `build/everyday-v17-expanded-data/` は変更していない。trainの各行の本文・かな・segments・出典・著者等をそのまま保持して選別し、書き換えた文を追加していない。除外理由、元行番号、行内容hash、統合先を `curation.json` に保存した。[全報告と出典対応](everyday-v19-data.json)も参照。

[ローカルデモ](http://127.0.0.1:8000/)を再読み込みし、「v19 データ整理・修正版」を選択→「読み込む」→難易度→「お題を生成」。初版の問題も比較できるよう、「v19 データ整理・初版」も残す。両モデルは各 **2,528,193 bytes**。全15モデルのWASM生成・実Edgeでの読み込みと3難易度生成を確認した。標準はstories-v2-diverse-3mb。公開GitHub Pagesへのpush・デプロイは未実施。

## 整理方針

全てtrain内で行い、validation/test/family-test/teacher-testはバイト列・SHA-256を変えない。評価票の生成出力を新しい学習用修正文に使わない。

| 処理 | 件数 | 扱い |
|---|---:|---|
| 確認済みの読み・修飾の問題 | 6 | 文脈上「かぜ」なのに「ふう」の5行、不自然な修飾1行を理由付きで除外 |
| 題材の隔離 | 29 | 「自殺」を含む行をタイピング用のsubsetから隔離。否定・予防の文も含み、文法不良というラベルではない |
| 読点・丁寧さ等の近似 | 207 | Sudachiの語と活用を使い、同じ内容の代表行を選ぶ |
| 構文への集中 | 1,515 | 名詞を伏せた同じ構文からの行数を抑え、有益な語・組合せは残す |
| **除外合計** | **1,757** | 元データは保持 |

近似の代表化では時制・否定・受身・命令・意志・疑問を区別する。名詞、助詞、述語を自由に交換する生成処理ではない。代表は元の短い文、同長なら行hashで決定的に選ぶ。「読む／読まない」「行く／行け」「新たに税を課す／新たな税を課す」を同じ扱いにしないテストを加えた。辞書解析による規則であり、全ての意味の違いを保証するものではない。

構文の上限は、2名詞以上・6単語以上の文を対象に、群×難易度×名詞伏せパターンごとに32行を目安とする。語の表記・品詞と、隣接する名詞・助詞・末尾動詞の特徴が失われないよう貪欲に代表を選び、被覆に必要なら上限超過を明示的に認める。超過の合計は85行。短い「これは本です／これは地図です」等や、語の違いを学ぶための例を一律に削らない。

被覆検査は**代表化後の候補**の66,394特徴が間引き後にも残ることを確認したもの。原本の全語彙ID・全読み・全意味関係が残る保証ではない。実際にtrainで使われるIDは8,700→8,694となり、配布語彙8,802 ID自体は維持した。隣接・末尾動詞の特徴は係り受けや意味役割の正解ラベルではない。

## 点検で統合規則を修正した

整理初版は57,352件で学習を終えたが、統合した25組のCodex点検で「政府はワインに新たに税を課した／政府はワインに新たな税を課した」の誤統合が見つかった。行為の修飾と名詞の修飾が異なるため、助動詞「だ」の連用・連体の区別を追加し、回帰テストを通した。

修正版では2行を復元し57,354件となった。修正版の保持文を各群15件ずつ計60件、統合候補25組を学習前に再点検し、この標本では自然さ・読み60/60、統合内容25/25を許容した。独立した人間の評価ではなく、全件の品質保証ではない。標本を見た後に都合のよい行へ入れ替えたり、生成の評価票を学習へ追加したりしていない。

初版モデルは既に推論可能だったため、規則の既知の問題を明示してデモへ残す。初版の自由生成品質は今回の主比較に含めず、修正版の方を評価する。初版と修正版の2行差が学習文抽選全体を変えるため、重みの違いを特定の文だけの効果と断定しない。

[初版の保持文点検](everyday-v19-initial-kept-agent-review.json)、[初版の統合点検](everyday-v19-initial-merged-pairs-agent-review.json)、[修正版の保持文点検](everyday-v19-final-kept-agent-review.json)、[修正版の統合点検](everyday-v19-final-merged-pairs-agent-review.json)、[除外6行と理由](../data/curation-v19/README.md)に判断を残す。

## データの偏りはどれだけ減ったか

| 群 | 整理前 | 修正版 |
|---|---:|---:|
| Tatoeba | 49,314 | 49,091 |
| teacher | 4,444 | 4,399 |
| everyday | 3,375 | 1,886 |
| paired | 1,978 | 1,978 |
| **合計** | **59,111** | **57,354** |

自作日常文の「4単語以上で同じ位置の1単語だけ違う別文がある行」は3,276→1,660件。全trainでは13,665→11,811件。各群250件の文字bigram Dice≥0.8の近似率は、自作日常文82.4→64.4%、教師文22.0→19.6%だった。近似は意味の同一性や低品質を直接表すものではない。

各群250件は同じseedの群別無作為抽出だが、母集団の変更により選ばれる行も変わる。差を対応のある同一文の検定として扱わない。pairedは全行を維持しており、標本上70.0→71.2%という差は削除効果を示さない。

通常比率で「1単語差の別文がある行」が占める期待抽選質量は31.8→30.4%、名詞伏せパターン上位10群の期待質量は6.3→4.9%。行数だけでなく抽選時の集中も減ったが、大幅な解消ではない。元の評価splitにあるtrain近似文は再配置していない。[整理後の全監査](everyday-v19-data-diversity-audit.json)、[整理前](training-data-diversity-audit.json)を参照。

## 同条件で学び直した比較

対照はv18 expanded-base、試験はv19 refined。通常比率65/15/15/5、同じ初期state hash、seed42、batch128、4層・幅128・1,796,608パラメータ、固定最終5,000step。既存重みから継続せずランダム初期値から学習した。学習行抽選はデータの変更により異なる。RTX 4070 Ti SUPER、PyTorch2.9.0+cu128、bf16。

修正版の学習時間83.3秒、peak CUDA allocated601,607,168 bytes。最終の共通validation lossは対照2.5763に対し2.5876。100対のうち符号化可能な91対で自然側を選好するFP32診断は80→79で改善していない。全splitの対象共起キー0件を再検査し、構成語被覆96/100も維持したが、同義語・係り受けを含む意味漏洩の保証ではない。

各モデル600要求、3難易度各200、履歴0、候補1、temperature0.8/top-p0.95。各100件を出力を見る前に抽出した。既存開始seed887000と、新しい開始seed900000の両方を使い、要求ごとに+4。全条件600/600成功。

コピーの共通参照は**整理前の全train**。整理で外した文も参照に残し、除外した文を出すだけで新規率が上がった扱いにしない。修正版自身のtrain参照による値は測定JSONへ別に保存する。照合・除外表・構文表はオフライン処理専用で推論には入れない。

| seed群 | 条件 | 完全一致 | 近似一致 | 自然さ / 100 | 読み / 100 | 両方合格かつ非一致 / 100 | 両方合格かつ近似も除外 / 100 |
|---|---|---:|---:|---:|---:|---:|---:|
| 887000 | 整理前v18 | 18.0% | 35.5% | 68 | 99 | **52** | **39** |
| 887000 | 整理後v19 | 19.8% | 36.8% | 74 | 100 | **47** | **36** |
| 900000 | 整理前v18 | 18.3% | 37.3% | 77 | 100 | **61** | **36** |
| 900000 | 整理後v19 | 21.3% | 36.7% | 79 | 99 | **62** | **44** |

自然さは両群で点検上上がったが小標本・同じ学習seedの主観的評価で、有意な改善とは呼ばない。コピー率は両方で上がり、自然な非一致文は一方で減少、他方でほぼ同じ。データの構造的な冗長性の改善と、未知の自然文を生成する能力の改善は別である。今回も標準には昇格しない。

既存seedの対照100件は配布5ファイル・生成条件・標本hashを照合してv18の判定を再利用した。本文・かなが同じ出力は旧票と矛盾しない判定を7件再利用した。残る293組をCodexが新規点検した。400行は400件の独立新規採点ではない。条件列は隠しているが、新規採点の構成や既知の文体から条件を推測できる。独立した人間の採点・複数学習seedは未実施。判断基準は従来の通常の日常文・普通の省略や比喩を許容する運用を継承し、全理由を保存した。

## 配布・速度・検証

初版・修正版とも5ファイル計2,528,193 bytes、INT8保存・CPU/WASM演算FP32。教師・Python・CUDA・除外表はアプリに不要。CPU normal各90件、開始seed905000、履歴0では修正版p50/p95 **23.3/33.4ms**、初版24.0/32.9ms。実Edge normal各30件、seed42・履歴16では修正版 **15.1/22.5ms**、初版15.8/23.8ms。専用計測は学習・評価・テスト終了後に実行し、モデルロード・描画を除く。CPUは初回・再試行を含みJSON直列化を除く。標本・環境内の結果で全要求の時間上限ではない。

全Python135件、標準モデル整合性、2モデル×FP32/INT8照合、全15モデル×30件のWASM smoke、実Edge全15モデル×3難易度と標準への再切替を通過。Rustの変更はなくworkspace全テストは再実行していない。独立人手による言語品質・題材適性の全件審査は未実施。今回除いた題材が生成から完全に消える保証もない。

## 再実行と証跡

`configs/everyday-v19-refined-data.json` と `configs/everyday-v19-refined.json` が最終設定。初版の設定・モデル・報告は比較と経緯の追跡用として残す。新しい外部データ・教師出力は追加せず、全出典とライセンスを親から継承した。除外6行の点検表をコーパス全体の品質認証や新しいCC0データと扱わない。

再実行時はmerged/output/checkpoint/export先を新しいパスにする。既存成果は上書きしない。

```powershell
.venv/Scripts/python.exe -m scripts.curate_training_data --config configs/everyday-v19-refined-data.json
.venv/Scripts/python.exe -m training.train --config configs/everyday-v19-refined.json
.venv/Scripts/python.exe -m scripts.export_experiment --checkpoint build/checkpoints/everyday-v19-refined.pt --prefix build/v19-export-recheck --output build/v19-runtime-recheck
.venv/Scripts/python.exe -m scripts.evaluate_candidates --config configs/everyday-v19-evaluation.json --output build/v19-development-recheck
.venv/Scripts/python.exe -m scripts.evaluate_candidates --config configs/everyday-v19-fresh-seeds.json --output build/v19-fresh-recheck
.venv/Scripts/python.exe scripts/build_wasm.py --model-catalog configs/demo-models.json
node scripts/smoke_wasm.mjs --catalog configs/demo-models.json
```

- [事前条件と点検後の修正記録](everyday-v19-protocol.json)、[学習設定・trace・checkpoint hash](everyday-v19-training.json)、[全行の保持・除外と出典](everyday-v19-data.json)
- 887000側：[全生成診断](everyday-v19-development-measurements.json)、[比較CSV](everyday-v19-development-summary.csv)、[対応表](everyday-v19-development-review-map.json)、[採点票](everyday-v19-development-blind-agent-review.csv)、[品質](everyday-v19-development-agent-review.json)、[結合集計](everyday-v19-development-joint-review.json)、[再利用対応](everyday-v19-development-review-reuse.json)、[未記入人手票](everyday-v19-development-human-review.csv)
- 900000側：[全生成診断](everyday-v19-fresh-measurements.json)、[比較CSV](everyday-v19-fresh-summary.csv)、[対応表](everyday-v19-fresh-review-map.json)、[採点票](everyday-v19-fresh-blind-agent-review.csv)、[品質](everyday-v19-fresh-agent-review.json)、[結合集計](everyday-v19-fresh-joint-review.json)、[再利用対応](everyday-v19-fresh-review-reuse.json)、[未記入人手票](everyday-v19-fresh-human-review.csv)
- [新規293組の点検](everyday-v19-new-unique-agent-review.json)、[組合せ監査](everyday-v19-combination-audit.json)、[CPU計測](everyday-v19-normal-timings.json)、[ブラウザー計測](everyday-v19-browser-timings.json)、[数値照合](everyday-v19-parity.json)、[検証logのhash](everyday-v19-verification.json)

元コーパス、初版の処理ソースsnapshot、checkpoint、全出力segmentsは `build/` に保持し、hashで結び付けた。次の増量では、この整理済みsubsetを比較候補に使えるが、今回の削除量や上限32を最適値・採用済み品質と扱わない。
