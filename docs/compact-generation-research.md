# 3MBで自然さと未知文を両立するための実験

2026-10-29。上限は配布3,000,000 bytes、ふつうのお題の生成100ms以内。固定文型は推論に使わない。「自然」は日本語の自然さであり、自然や風景の題材も残す。

## 今回の結果

**容量と速度には余裕があるが、今回の新候補は自然さが不足し、採用しなかった。** デモの小型v3と標準v2は継続する。新規率だけを改善して文章品質を下げる変更は採用しない。

| 候補 | 配布バイト | 600seed中の重複件数 | 新規正規化本文 | 自然さ・読みの点検 |
|---|---:|---:|---:|---:|
| 比較用の小型v3 | 2,423,935 | 70 | 36.7% | 95/100 |
| v11・通常学習 | 2,527,477 | 6 | 82.3% | 64/100 |
| v11・分岐を残す学習 | 2,527,477 | 7 | 76.5% | 64/100 |
| v11・日常文と教師文を重視して追加学習 | 2,528,978 | 43 | 56.5% | 84/100 |

全候補でseed 730000〜730599、easy/normal/hard各200件。履歴を無効化・消去し、別seed間の重複を回避する処理なしで測った。通常学習と分岐学習は同じ初期seed・構成・データ・バッチ抽選・temperature 0.8 / top-p 0.95。v3は従来の推奨temperature 1.0 / top-p 0.95なので、v3との差はモデル・データ・設定を合わせた比較であり、特定の変更だけの効果ではない。

新規本文は、v3とv11の両方のタスク学習文の和集合に、表記を正規化した本文がない割合。どこにも存在しない文章の証明ではない。今回この共通参照での値は各候補の自分の学習文に対する値と一致した。

点検対象は出力を見る前に乱数20261029で600件から100件を選んだ。Codexによる日本語・読みの点検であり、独立した人による盲検評価ではない。破綻した接続、助詞、語の相性、不要な重複を不合格とした。基準の運用には主観が残るため、本文と全点検票を保存している。童話調は各候補とも点検100件中0件。

## 容量の使い方を変更した

従来の小型v3は語彙約1,000項目。v11は、既存v9データの形態素区切りを利用し、**trainだけから8,795語と7特殊ID**を作る。かな・品詞を単語IDに対応させるため、subwordの辞書探索や別の読みモデルは不要。学習文の全文一覧は配布しない。

幅128・4層・FF384、入出力埋め込み共有で1,796,608パラメータ。INT8保存した重み約1.87MB、表記・かな・品詞の語彙約0.64MB、設定と出典・LICENSEを含め約2.53MBとなった。幅を細くして語彙へ容量を割り振る試みであり、事前学習済み日本語本体を継承したモデルではない。

学習59,599件、validation 3,040件、test 2,940件。親データのsplitは維持し、学習splitにない語を含むvalidation 5件・test 4件は除外した。全testに対する性能を測った扱いにはしない。入力・出力のhashと除外数は比較JSONに記録した。

## 用途向けの仮説：分岐を残す学習

同じ全文prefixに対して学習文に複数の続きがあるとき、その条件付き頻度を `p` とし、`q ∝ p^(1−α)` に平坦化する。正解トークンのcross entropyを `q/p` で重み付けする。今回は `α=0.5`、重みは0.25〜4に制限した。続きが1通りのprefixは重み1で文法を維持する狙い。difficultyと文頭からの全prefixを使い、末尾の数語だけが似た無関係な文脈は混ぜない。

表はtrainだけから作り、学習終了後には配布しない。推論時のトークン選択、履歴処理、文型の規則は増やしていない。これはコーパスに対する重み付けの実験であり、新しい原理の発明や、既存論文のOPDを再現したものとは主張しない。語彙拡張と分岐学習を区別するため、同じ小型構成の通常学習を対照にした。

**今回の分岐学習は不採用。** 通常学習より新規率が低く、自然さも改善しなかった。学習文内の少数派を残すことと、未見の文で意味のつながりを守ることは別の問題だった。

## 自然さを回復する追加実験

通常学習の候補から、教師生成文55%・日常文30%・Tatoeba 10%・対応付き散文5%に抽選比率を変更して追加学習した。通常学習時は順に15%・15%・65%・5%。250stepがvalidationによる最良で、1,500step時点で早期停止した。

この変更で点検は64→84/100に回復したが、新規率は82.3→56.5%、重複は6→43件と後退した。validationの混合比も変わるため、重み付けされたlossの値を通常学習と直接比較しない。探索用seed群での結果であり、採用候補としての別seed確認試験はまだ行っていない。

誤りの例は「その機械は穴を浴びた」「パソコンがかかってきた」。単語や助詞を組み合わせるだけの広さは増えたが、対象と動作の相性が崩れている。次の改善では語彙の追加だけより、こうした関係の学習を優先する。

## 速度

追加学習候補を、ライブデモとは別の `build/v11-preview` で実Edge Web Worker測定。normal 30件、temperature 0.8 / top-p 0.9、履歴16件、seed42。生成時間の中央値13.7ms、p95 20.5ms。ロード・描画を除き、今回の環境では100ms以内だった。品質評価はtop-p 0.95なので、速度測定と設定が異なる。

Windows x86_64 / Intel Family 6 Model 183 / Edge 156.0.4314.50。native releaseの探索測定は同一seed42を30回、履歴0で中央値24.0ms・p95 39.1ms、cold load 18.9ms。nativeは異なる文を30件測った値ではない。他の端末への速度保証や、新候補の品質合格を意味しない。

## 調べた研究と次の具体案

- [Entropy-Aware On-Policy Distillation, ICML 2026](https://proceedings.mlr.press/v306/jin26e.html)：高い確信の出力への集中を避け、教師の不確実な箇所でforward KLも使う研究。数学推論での結果なので、日本語のお題で同じ効果が出るとは仮定しない。
- [Influence-Directed Distillation, 2026-08 preprint](https://arxiv.org/abs/2608.29846)：更新が多様性を狭める方向かどうかを扱う。今回の頻度平坦化とは異なる方法であり、未実装。
- [On-Policy Self-Distillation with Sampled Demonstrations Reduces Output Diversity, 2026-06 preprint](https://arxiv.org/abs/2606.26091)：自己蒸留で既に強い出力へ集中する失敗を分析。単一の正解例を模倣させるだけでは今回の目的に不十分かもしれない。

次の実験案は **複数の自然な修復を残す対比学習**。学習用seedだけで小型モデルを動かし、教師に「機械→穴を浴びる」のような誤った関係と、自然に続く複数の候補を対で点検させる。不自然な続きの確率を下げつつ、自然な代案を一つに絞らず学習する。教師や候補表は推論には含めず、配布サイズと実行手順を維持する狙い。

この案は未実装であり、既存の対比学習・unlikelihood・蒸留をこの課題に合わせる仮説。新規性や改善効果は未証明。今回点検した730000台のseedは学習用の誤り収集に流用せず、次候補の判断には別の確認用seedも使う必要がある。

## 記録と再現

[比較・設定・速度のJSON](everyday-v11-comparison.json)、[通常学習600seed](everyday-v11-ce-independent-seeds.json)、[分岐学習600seed](everyday-v11-branch-independent-seeds.json)、[追加学習600seed](everyday-v11-quality-independent-seeds.json)、[旧v3の同seed比較](everyday-v11-reference-independent-seeds.json)。各モデルの点検票は同名の `*-review.json`。

既存のv9準備データとCUDA環境を用意して実行する。データの出典・ライセンスは親manifestを継承する。

```powershell
python -m scripts.prepare_compact_words --source build/everyday-v9-data --output build/everyday-v11-data
python -m training.train --config configs/everyday-v11-ce-3mb.json
python -m training.train --config configs/everyday-v11-branch-3mb.json
python -m scripts.initialize_compact --checkpoint build/checkpoints/everyday-v11-ce.pt --data-dir build/everyday-v11-data --output build/everyday-v11-data/ce-initialization.pt
python -m training.train --config configs/everyday-v11-quality-3mb.json
```

exportは `python -m training.export` に `--generation-mode autoregressive --lexical-boundary-guard --polite-boundary-guard --normalize-morph-readings`、続いて `training.quantize` のINT8、`training.package --strip-token-metadata`。分岐重みの条件付き質量・文脈分離・PAD無視・勾配、trainだけの語彙作成とchecksum、初期化の語彙照合をテストした。新候補の保存重みはPython/Candleの88,020 logitで絶対許容差0.0002以内を確認した。
