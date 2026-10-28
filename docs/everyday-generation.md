# 自然な文と多様性の改善実験

標準モデルは `stories-v2-diverse-3mb` のまま。以下は検証中で、Pagesの配布設定も変更していない。

「自然」は、文法・語の組み合わせ・読みが自然な日本語を指す。風景、動植物、天気のお題も残す。童話調への集中を抑え、家事・料理・買い物・仕事・学習・移動・趣味など、話題と文体の両方を広げる。次期モデルは品質優先で32,000,000 bytes以内を許容する。推論は固定文型に依存しない学習モデル主体とする。

## 方式

v6/v8は [rinna/japanese-gpt2-xsmall](https://huggingface.co/rinna/japanese-gpt2-xsmall) のsubword埋め込みとTransformerを継承する。6層・幅512・8head・FFN2304・最大64位置、28,714,496パラメータ。16,000個のsubword IDを予測する。学習用の原本は配布に含めず、公開リポジトリのPythonコードも実行しない。

7,993項目の単語辞書で表記・かな・品詞を保持し、モデルが出す文字列を単語として最後まで分割できるよう制約する。辞書のtrieは単語の綴りだけを制約する。文型・全文候補・学習文の除外リストは推論に使わない。文中の語句と順序はモデルが選ぶ。アプリ実行時にPythonや外部LLMは不要。

同じ表記の読みは、学習データで観測した直前2語の文脈と語の頻度で解決する。かな長に合わせるために短い読みへ切り替えない。未見の文脈の誤読は残るため、構造整合性と読みの品質は別に評価する。

## データと帰属

`fetch_tatoeba.py` は [Tatoeba日本語エクスポート](https://downloads.tatoeba.org/exports/per_language/jpn/) と読みを保存しSHA-256を記録する。`prepare_tatoeba.py` は提供された読みとSudachiの全文一致を要求し、人名・固有名詞・未知語・対応しない読みを除く。読みが一致していても文章の自然さを審査済みという意味ではない。

正規化本文のSHA-256でtrain/validation/testを先に分割し、語彙・頻度はtrainだけから計算する。既存の評価本文も保護する。subwordのv6/v8はtrain57,148文、validation2,920文、test2,844文、family-test152文。難易度を均等にし、各難易度の期待比率をTatoeba85%、自作日常文10%、既存自作散文5%とする。頻出語の調整幅には上限を設ける。

Tatoebaには翻訳例文や「彼・彼女」を主語にした文が多く、出典を変えるだけでは偏りを解決できなかった。自作の補助データには、語の相性を指定した116種類の文型・分野の組によるv3データも含む。これは学習時の補助であり、推論の文型ではない。未出力の組み合わせを残して学習するが、未見構文への汎化の証明にはならない。

TatoebaはCC BY 2.0 FR、自作散文はCC0-1.0、SudachiDictはApache-2.0、rinnaモデルはMIT。出典・revision・ハッシュをmanifestに記録し、配布LICENSEに帰属と第三者ライセンスを含める。

## 結果と採用条件

v3はCodexの無作為点検で開発100/100、確認98/100が自然さと読みを両方満たし、童話調は両方0/100。ただし未学習の正規化本文は約33〜34%で不足した。v3の評価ファイルの合格は新規率30%以上の小型モデル用実験基準に対するもので、次期モデルとしての採用を意味しない。

v4のスクラッチ学習、v5の単語埋め込みへの変換、v6のsubword継承、追加学習なしのv7も比較。v4/v5は不自然な語句の組み合わせが多く、v7は60要求中22件の生成失敗と反復があり不採用。v6では100個の独立seedで1〜3件の重複が残った。v8は事前学習済み本体の学習率を下げ、文法を保持しながら長さと辞書へ適応させる試験。

次期モデルの事前目標は600文で新規正規化本文95%以上、独自本文99%以上、配布32MB以内。無作為100文で自然さと読みが両方正しい文95以上、未学習本文の点検30以上かつその合格率95%以上、童話調5以下、8分野以上、最大分野30以下を要求する。開発と確認を分ける。数値を満たすために基準を下げて採用しない。

「ある日」「少年」「夢」などの語の集計は補助情報で、これだけで童話調を判定しない。自然・風景の題材自体を不合格にしない。点検はCodexによるもので、独立した人間の点検は未実施。

v8の履歴なし600seed（530000〜530599）では生成失敗0、578種類・重複22件（独自本文96.33%）、未学習正規化本文462/600（77%）だった。新規率はeasy54%、normal92.5%、hard84.5%。無作為100文の探索的なCodex点検は自然さ・読みの合格88文で、基準未達。これは昇格用の開発・確認点検ではない。反復、意味の矛盾、時制の不整合、行き先と行動の不適合が残る。分野・童話調の正式点検まで進めず、標準モデルへの反映を見送った。

配布5ファイルの合計は30,818,748 bytes（約30.8MB）。INT8保存とRustのパリティ検証、WASMの30文の構造確認、実ブラウザーのWorkerによる3難易度の生成は通過した。実行時はFP32へ復元するため、32MBはメモリ使用量の上限ではない。

評価原本は [独立seed測定](everyday-v8-32mb-independent-seeds.json)、点検は [探索的点検](everyday-v8-32mb-exploratory-review.json)。次の改善対象は短いお題の集中と、翻訳例文への適応だけでは補えない語句・意味の自然さである。

## seed間の重複

CLI `evaluate-seeds` は毎回別のseedを指定し、履歴を無効化して各要求の前にも消去する。学習文に一致する出力も除外しない。失敗を分母から除かず、出力・seed・重複例・処理時間を保存する。

`scripts.independent_seeds` は実行ファイル・Rustソース・モデル・学習参照ファイルのハッシュも記録し、難易度別の重複率と新規率を計算する。時制や既知の時間・様態表現の違いは正規化本文でも確認する。この正規化は限定的な近似で、意味の同一性の判定ではない。「未学習」は追加学習ファイルに本文がないという意味で、上流の事前学習にないことは確認できない。

同じseedの再現性には同一モデル・設定・実行環境と同じ履歴状態が必要。通常利用の履歴で避けられる重複と、モデルが独立seedから生成する多様性を区別する。32MB候補の昇格には履歴なし600seedで独自本文99%以上も要求する。

## 再現

学習用依存関係に加えて `training/prose-requirements.txt` を導入する。設定はCUDA学習。原本の取得にはネットワークが必要で、保存後のハッシュ変更はエラーにする。

```powershell
.venv/Scripts/python.exe -m scripts.fetch_tatoeba
.venv/Scripts/python.exe -m scripts.fetch_pretrained
.venv/Scripts/python.exe -m scripts.prepare_everyday
.venv/Scripts/python.exe -m scripts.prepare_tatoeba
.venv/Scripts/python.exe -m scripts.prepare_subwords
.venv/Scripts/python.exe -m training.train --config configs/everyday-v8-32mb.json
.venv/Scripts/python.exe -m training.export --checkpoint build/checkpoints/everyday-v8-32mb.pt --output build/everyday-v8-32mb --generation-mode autoregressive --polite-boundary-guard --normalize-morph-readings --lexical-boundary-guard --subword-start-temperature 1.0
.venv/Scripts/python.exe -m training.quantize --model build/everyday-v8-32mb --output build/everyday-v8-32mb-int8
.venv/Scripts/python.exe -m training.package --model build/everyday-v8-32mb-int8 --output build/everyday-v8-runtime
cargo build --release --locked -p fumiori-cli
.venv/Scripts/python.exe -m scripts.independent_seeds --model build/everyday-v8-runtime --training-data build/everyday-v6-data/train.jsonl --samples 600 --seed 530000 --output docs/everyday-v8-32mb-independent-seeds.json
```

通常の開発・確認評価は `scripts/stories_evaluate.py --model-name everyday-v8-32mb --data-dir build/everyday-v6-data --temperature .5 --top-p .9 --everyday-review` と `--confirmation` で実行する。生成後、事前に無作為抽出された点検対象を全件評価する。点検票なしでは昇格しない。
