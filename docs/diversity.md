# お題の多様性改善

現行モデルは **stories-v2-diverse-3mb**。学習データの整合と語彙の再配分を行い、開発・確認の採用基準を満たしたモデルを標準配置と `models/stories-v2-diverse-3mb/` に反映した。配布容量は2,903,289 bytes。生成設定はtemperature=0.8、top_p=0.95。

## 測定結果

同じ現行Rust実行ファイル・開発seed42/2026/73・各200文・履歴16件で比較した。v1は従来の既定設定0.75/0.95、v2は開発段階で選んだ0.8/0.95。データ・学習重み・生成設定をまとめた比較で、語彙再配分単独の効果を示すものではない。

| 開発用600文 | v1 | v2 |
|---|---:|---:|
| 名詞の種類 | 135 | 210 |
| 名詞の上位10種類への集中 | 46.5% | 36.7% |
| 述語の種類 | 228 | 274 |
| 述語の上位10種類への集中 | 27.8% | 24.6% |
| 文末の上位10種類への集中 | 27.0% | 24.7% |
| 異なる本文 | 579 | 582 |
| 学習にない正規化本文 | 63.5% | 57.3% |

語彙の広がりは改善したが、未知本文の割合は下がった。単一指標だけで改善を主張しない。[v1の全出力](diversity-baseline-measurements.json)、[v2の開発結果](stories-v2-diverse-3mb-development-measurements.json)、[確認結果](stories-v2-diverse-3mb-confirmation-measurements.json)、[採用判定](stories-v2-diverse-3mb-promotion.json)に記録する。

自然さ・読みのCodex点検は開発92/100（未知本文60/68）、確認90/100（未知本文53/61）。確認600文は584種類、未知本文319件（53.2%）。独立した人間の点検ではなく、誤読や不自然な構文は残る。

0.75での試行は全体91/100、未知本文41/49、新規率49.7%で不採用。開発評価で0.8を選び、その後は重み・設定を変えず確認seed3432/811/1907で検証した。点検票と不採用記録も `*-t075-*` に残す。

## 集中度の測り方

本文の一致率だけでは、同じ登場人物・もの・動作への集中を捉えられない。既存の区切りと品詞を使って、名詞・述語・文末の上位語への集中度を全体および難易度別に測る。名詞・述語は1文に同じ語が複数あっても1回と数える。文末は記号を除いた末尾2区切り。表層形の集計で、活用の違いは別の語として扱う。

`top10_share` は各軸の全出現数に対する上位10種類の割合。`effective_types` は出現比率のエントロピーの指数で、少数の語に集中すると小さくなる。`top` には文数と全文に対する出現率を残す。語彙数はサンプル数に依存するため、比較は同じ生成条件・同じ難易度別件数で行う。

```powershell
.venv/Scripts/python.exe -m scripts.audit_diversity build/stories-v1-data/train.jsonl build/name-free-samples.jsonl --output build/diversity-audit.json
```

入力はJSONL、`samples` を含む生成レポート、または生成文のJSON配列。全行を対象にし、入力ファイルのSHA-256も記録する。未知本文や見た目の良さによる選別はしない。

## 学習での調整

`configs/stories-v2-diverse-3mb.json` は語彙の再配分と、学習・生成のトークン条件の整合を試す候補モデル。従来の10万文のうち79,093文は、現在の生成では使えない「文字」・人名・固有名詞トークンを含んでいた。文字単位で学習した単語を生成途中で遮断するため、「雪だ校庭」「雪だ落ちていました」のような破綻も点検で見つかった。

現行の `scripts.prepare_stories` で同じ上流シャードを再走査し、条件を満たす30,800文で学習する。validation/testは各800文、family-testは300文。要求上限10万文に達する前にシャードを走査し終えるため、10万文を得たという主張はしない。`require_generation_eligible` により、禁止トークンを含む古いデータを誤って使うと学習開始前にエラーにする。[コーパス監査](diversity-corpus-audit.json)に入力のhashと件数を記録する。

難易度ごとに、各文に含まれる名詞・述語の文単位出現頻度を学習splitだけで数え、その対数平均の逆数方向に重みをつける。`strength=0.5` とし、同じ難易度内の最大重み比を `max_ratio=3` に制限する。最後に難易度ごとに重みを正規化して、各難易度の期待比率1:1:1を維持する。

元の文・かな・区切り・正解トークンは変更しない。文章中の単語を無作為に交換したり、稀な語を無制限に優先したりはしない。`strength=0` は従来の難易度バランスと一致する。語彙特徴がない文の係数は1。

```powershell
.venv/Scripts/python.exe -m scripts.prepare_stories --out-dir build/stories-v2-data --train-limit 100000
.venv/Scripts/python.exe -m training.train --config configs/stories-v2-diverse-3mb.json
.venv/Scripts/python.exe -m training.export --checkpoint build/checkpoints/stories-v2-diverse-3mb.pt --output build/stories-v2-diverse-3mb --generation-mode autoregressive --polite-boundary-guard --normalize-morph-readings
.venv/Scripts/python.exe -m training.quantize --model build/stories-v2-diverse-3mb --output build/stories-v2-diverse-3mb-int8
target/release/fumiori-cli.exe parity --model build/stories-v2-diverse-3mb
target/release/fumiori-cli.exe parity --model build/stories-v2-diverse-3mb-int8
.venv/Scripts/python.exe scripts/stories_evaluate.py --model-name stories-v2-diverse-3mb --data-dir build/stories-v2-data --temperature .8
```

生成評価にも `diversity` を記録する。自然さ・読みは固定seedの無作為100文で別途点検し、モデル・生成方針・点検対象のhashを照合する。採用には従来と同じ開発・確認の両評価が必要。多様性指標の改善だけで採用しない。

既存の10万文をそのまま語彙再配分だけした最初の探索試行は、無作為100文の自然さ・読みの合格が80文だったため不採用。[点検結果](stories-v2-lexical-only-development-review.json)と[採用判定](stories-v2-lexical-only-development-gate.json)を残す。この試行は読点処理の修正前の古い実行ファイルを使っていたため、現行Rustでの比較には使わない。最終比較は再ビルド後にすべて取り直した。現在の評価はソースより古いCLIを拒否し、実行ファイルとRustソースのSHA-256を記録する。独立した人間の評価ではない。

## 限界

この集計は意味や話題の分類ではない。「花」と「桜」などの近い題材、主語の省略、表記の揺れを同一視できず、社会的属性や題材の公平さを証明するものではない。語の頻度を調整しても、元データに少ない生活・仕事・学習などの題材は新たに増えない。童話由来という偏りを解消するには、出典・読み・文脈を確認した異なる分野の文章を追加し、話題ごとの点検も行う必要がある。

学習の再配分で稀な不自然な文を拾いやすくなる可能性もあるため、自然さと読みの点検結果を多様性と並べて判断する。
