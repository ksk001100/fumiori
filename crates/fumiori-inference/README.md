# fumiori-inference

Candle CPU FP32 Transformer、export済みsafetensors/設定/語彙/manifest読込、temperature/top-p/seed、有限再試行、検証、履歴、明示fallback。標準モデルは生成失敗時にエラーを返し、静的fallbackは使わない。公開APIは [設計](../../docs/architecture.md) を参照。`cargo test -p fumiori-inference`。
