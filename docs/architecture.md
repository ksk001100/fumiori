# 構成とAPI

現行モデルは `stories-v1-3mb`。`JSONL → fumiori-data → vocab + split + manifest → PyTorch → checkpoint → safetensors → Candle` の順で作成する。表記・かな・品詞を同じ語彙IDに持たせ、別の読みモデルは実行しない。

## Rust利用

```rust,no_run
use fumiori_core::Difficulty;
use fumiori_inference::TypingGenerator;

fn main() -> anyhow::Result<()> {
    let mut generator = TypingGenerator::load("build/fumiori-stories-v1-3mb")?;
    let mut options = generator.default_options();
    options.difficulty = Difficulty::Normal;
    options.seed = Some(42);
    let prompt = generator.generate(&options)?;
    println!("{} / {}", prompt.text, prompt.kana);
    Ok(())
}
```

`TypingGenerator::from_bytes(manifest, config, vocab, weights)` はメモリから読み込む。`include_bytes!` でモデルを実行ファイルに埋め込める。重み・語彙・設定のSHA-256と形式を検証し、欠損・破損はエラーとして返す。

`generate(&mut self, &GenerateOptions)` は同期処理。生成器を使い回すことで履歴と乱数状態を保持する。`default_options()` はモデルの推奨temperature/top-pを返し、難易度やseedなどは呼び出し側で変更できる。

`set_history_size(n)` / `clear_history()` で重複履歴を管理する。`last_stats` は試行数・棄却数・トークン数を保持する。全エラーは `Result` で返す。

返り値 `TypingPrompt` は `text` / `kana` / `segments` / `source` / `model_version` / `reading_validation`。`model_version` はデータ形式のバージョンであり、配布モデル名のv1とは別。公開出力は末尾の句点を省き、区切りの連結と本文・かなを一致させる。

## v1の推論

4層・幅224・8head・FFN672・最大64位置、2,624,832パラメータ、2,644語彙ID。CPUのCandleでFP32演算する。INT8保存版は読み込み時にFP32へ戻す。KV cacheはなく、各ステップでprefixを再計算する。

モデルが次の語彙IDを予測する自己回帰方式。全文候補・trie・学習文一覧・学習文の除外リストを推論に渡さない。かな長、終端、漢字名詞の境界、丁寧形の連結、一部の読み補正を適用する。意味や文脈的な読みは保証しない。

このモデルは静的fallbackを禁止し、生成失敗はエラーになる。

学習はCUDA/PyTorch、アプリ推論はRust/Candle CPU。WASM版も同じgeneratorを呼び、Web Workerで同期推論を実行する。詳細は [モデルの設計と評価](three-mb-generation.md)。
