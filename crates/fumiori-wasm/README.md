# fumiori-wasm

共通のCandle CPU推論を `wasm-bindgen` 経由で使う。`Generator` はmanifest・設定・語彙・重みのバイト列を受け取り、ネイティブ推論と同じハッシュと構造を検証する。`generate(difficulty, seed)` はお題のJSONを返し、`logits(ids)` は数値照合に使う。破棄時は `free()` を呼ぶ。

リポジトリルートで、同梱の標準・試作全モデルからデモをビルドする。学習exportやPyTorchは不要。

```powershell
python scripts/build_wasm.py --model-catalog configs/demo-models.json
node scripts/smoke_wasm.mjs --catalog configs/demo-models.json
python scripts/serve_demo.py
```

`wasm32-unknown-unknown` と `wasm-bindgen-cli 0.2.127` が必要。[ローカルデモ](http://127.0.0.1:8000/)を開き、モデルを選択して「読み込む」→「お題を生成」。モデル一覧は `configs/demo-models.json`。GitHub Pagesも同じ一覧からビルドし、モデルのmanifestハッシュを検証する。[公開手順](../../README.md#github-pagesへの公開)を参照。

単一の標準モデルだけを確認する場合は `python scripts/build_wasm.py --runtime-model models/stories-v2-diverse-3mb` を使える。全モデルのデモへ戻すときは一覧指定で再ビルドする。

デモはmodule Web Worker内で単一スレッドの推論を行う。INT8の保存重みはロード時にFP32へ戻す。直近のv21確認は実Edgeで実施しており、全ブラウザーの動作保証ではない。[性能の測定条件](../../docs/lightweight-generation.md)、[最新の比較と制限](../../docs/compositional-generation-v21.md)を参照。
