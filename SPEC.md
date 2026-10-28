# Fumioriの仕様

日本語タイピング用の短文をオフラインで生成するRustライブラリ。モデルはstories-v1-3mb。

- 出力は表示本文 `text`、正書法の入力かな `kana`、表記・かな・品詞の `segments` を持つ。
- 難易度はかな長でeasy=5〜12、normal=13〜22、hard=23〜40文字。
- Rust/CandleによるCPU推論とWASM/Web Workerに対応する。
- モデルが語彙IDを逐次予測する。全文候補、学習文一覧、外部LLMは推論に使わない。
- 配布モデルは重み・語彙・設定・manifest・LICENSEを含め3,000,000 bytes以内。実行ファイルと実行時メモリは別。
- INT8は保存形式。演算はFP32。
- モデル破損・不整合・生成失敗はエラーを返す。静的fallbackを使わない。
- 表記・かなの構造整合性と、文脈上の自然さ・読みの品質は区別する。独立した人間の評価は未実施。

利用は [README](README.md)、APIは [構成とAPI](docs/architecture.md)、学習・評価条件は [モデルの説明](docs/three-mb-generation.md) を参照。
