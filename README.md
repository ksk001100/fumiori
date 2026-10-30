# Fumiori — Tiny Japanese Typing Generator

日本語タイピング用の短文を、本文・かな・品詞付きの区切りとして生成するRustライブラリ。
CandleによるCPU/WASM推論を提供する。アプリ実行時にPython・PyTorch・CUDAは不要。

学習・配布モデル・品質評価・比較Webデモは別リポジトリの
[fumiori-models](https://github.com/ksk001100/fumiori-models)で管理する。
標準モデルは引き続き **stories-v2-diverse-3mb**。v21などの試作は比較用であり、標準には採用していない。

## Rustアプリへの組み込み

```toml
[dependencies]
anyhow = "1"
fumiori-core = { git = "https://github.com/ksk001100/fumiori.git" }
fumiori-inference = { git = "https://github.com/ksk001100/fumiori.git" }
```

Cargo依存にモデルは含まれない。[モデルリポジトリ](https://github.com/ksk001100/fumiori-models/tree/main/models/stories-v2-diverse-3mb)の
`model.safetensors`、`vocab.json`、`model-config.json`、`manifest.json`、`LICENSE`の5ファイルをまとめて取得し、アプリに同梱する。
リポジトリ全体を取得する場合、モデルを使うだけならsubmoduleの取得は不要。

```powershell
git clone https://github.com/ksk001100/fumiori-models.git
cargo run --release -p fumiori-cli -- generate --model fumiori-models/models/stories-v2-diverse-3mb --difficulty mixed --count 10 --seed 3432
```

```rust,no_run
use fumiori_inference::TypingGenerator;

fn main() -> anyhow::Result<()> {
    let mut generator = TypingGenerator::load("fumiori-models/models/stories-v2-diverse-3mb")?;
    let options = generator.default_options();
    let prompt = generator.generate(&options)?;
    println!("{} / {}", prompt.text, prompt.kana);
    Ok(())
}
```

モデルパスはプロセスの作業ディレクトリからの相対パス。`TypingGenerator::from_bytes`と`include_bytes!`による埋め込みにも対応する。
生成器は読み込み後に使い回す。同期処理のためGUIではワーカースレッドで実行する。

`options.difficulty`は`Difficulty::Easy` / `Normal` / `Hard`。かな長は5〜12 / 13〜22 / 23〜40文字。
返り値は`text`、`kana`、`segments`、`source`、`model_version`、`reading_validation`。
助詞「は・へ・を」は正書法を維持し、読点「、」もかな長に数える。公開出力の末尾の「。」は省く。
manifestのSHA-256を検証し、モデル破損・標準モデルの生成失敗はエラーで返す。
構造検証は文の自然さや文脈的な読みを保証しない。

## 構成と開発

| パス | 役割 |
|---|---|
| `crates/fumiori-core` | 共通型・構造検証・ハッシュ |
| `crates/fumiori-inference` | モデル読み込み・CPU推論・生成・履歴 |
| `crates/fumiori-cli` | 生成・評価・性能計測のCLI |
| `crates/fumiori-wasm` | ブラウザー向けWASM API |
| `data/examples` | 明示的なfallback APIの既存fixtureとライセンス |

```powershell
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo test --workspace --locked
cargo build --locked --release -p fumiori-wasm --target wasm32-unknown-unknown
```

Rustの単体テストにモデルやPythonは不要。CIでは固定したモデルリポジトリの標準モデルを取得し、実モデルの読み込み・生成も確認する。
WASMには`wasm32-unknown-unknown`が必要。モデル比較デモのビルドとPages公開は
[fumiori-modelsの手順](https://github.com/ksk001100/fumiori-models#生成とwebデモ)を参照。
[公開デモ](https://ksk001100.github.io/fumiori-models/)でモデルを選び、「読み込む」→「お題を生成」で試せる。

[APIと構成](docs/architecture.md)、[分割・互換性・更新手順](docs/repository-split.md)。
