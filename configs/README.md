# 学習設定

`everyday-v16-branch-{count,mass}.json` はv15比率変更モデルと同じ条件で、分岐の頻度源 `uniform_rows` / `sampling_mass` だけを比較する。`everyday-v16-evaluation.json` はv15の不変な対照2種類と新モデル2種類を同じ開発seedで評価する。[v16記録](../docs/compositional-generation-v16.md)を参照。

`everyday-v15-{base,context,quality,quality-context}.json` は名詞入力マスクと学習データ比率の2×2比較。validation基準と容量・学習量を固定し、`independent_data_rng` で入力マスクによる学習文抽選のずれを避ける。`everyday-v15-evaluation.json` が各600要求・100件点検の設定。[v15記録](../docs/compositional-generation-v15.md)を参照。

`everyday-v14-holdout-ce.json` / `everyday-v14-holdout-smooth.json` は組合せを除去したデータでの通常CE／平滑化CEの比較。`everyday-v14-evaluation.json` は同じ要求ごとの候補を使う1/2/4候補比較とサンプリング比較。`demo-models.json` は標準・試作を含むデモのモデル一覧。モデルを追加したらこの一覧とPagesの検証対象も更新する。[v14記録](../docs/compositional-generation-v14.md)を参照。

`novelty-baseline-v1.json` は学習設定ではなく、標準モデルのPhase 1評価条件。重み・データmanifestのhash、難易度、独立seed、サンプリング、評価票の抽出条件を固定する。[実行手順と指標の定義](../docs/compositional-generation-phase1.md)を参照。

現行設定は `stories-v2-diverse-3mb.json`。`stories-v1-3mb.json` とそのデータ・checkpointは従来モデルの設定として保持する。

`data/paired-v1` の対応付き散文と日本語SimpleStoriesを入力に、`scripts/stories_pipeline.py` でデータ準備・学習・export・評価を実行する。

[設計・評価・再現手順](../docs/three-mb-generation.md)

`stories-v2-diverse-3mb.json` は語彙の集中を抑える現行モデル用。現行の生成で使えないトークンを含む文を除いて作り直した `build/stories-v2-data` を使い、`lexical_balance` で名詞・述語の文単位出現頻度を考慮する。難易度の期待比率は1:1:1、同じ難易度内の最大重み比は3倍。v1の設定・checkpointを上書きしない。[多様性改善](../docs/diversity.md)を参照。
