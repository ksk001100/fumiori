# 明示的fallback API用fixture

`sentences.jsonl`は自作30文（CC0-1.0）。`TypingGenerator::fallback_only()`の既存データとして保持する。
標準モデルの生成失敗時に自動で使うデータではない。独立した人間の確認は未実施。
ライセンスは[LICENSE](LICENSE)。データ準備・辞書fixture・生成スクリプトは
[fumiori-models/data/examples](https://github.com/ksk001100/fumiori-models/tree/main/data/examples)へ移した。
