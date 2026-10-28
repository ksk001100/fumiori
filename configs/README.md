# 学習設定

現行設定は `stories-v2-diverse-3mb.json`。`stories-v1-3mb.json` とそのデータ・checkpointは従来モデルの設定として保持する。

`data/paired-v1` の対応付き散文と日本語SimpleStoriesを入力に、`scripts/stories_pipeline.py` でデータ準備・学習・export・評価を実行する。

[設計・評価・再現手順](../docs/three-mb-generation.md)

`stories-v2-diverse-3mb.json` は語彙の集中を抑える現行モデル用。現行の生成で使えないトークンを含む文を除いて作り直した `build/stories-v2-data` を使い、`lexical_balance` で名詞・述語の文単位出現頻度を考慮する。難易度の期待比率は1:1:1、同じ難易度内の最大重み比は3倍。v1の設定・checkpointを上書きしない。[多様性改善](../docs/diversity.md)を参照。
