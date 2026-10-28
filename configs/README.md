# 学習設定

`stories-v1-3mb.json` がstories-v1-3mbの学習設定。データは `build/stories-v1-data`、checkpointは `build/checkpoints/stories-v1-3mb.pt` に出力する。

`data/paired-v1` の対応付き散文と日本語SimpleStoriesを入力に、`scripts/stories_pipeline.py` でデータ準備・学習・export・評価を実行する。

[設計・評価・再現手順](../docs/three-mb-generation.md)
