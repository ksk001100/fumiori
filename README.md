# Fumiori — Tiny Japanese Typing Generator

日本語タイピング用の短文を、本文・かな・品詞付きの区切りとして生成するRustライブラリ。
モデルは **stories-v1-3mb**。

配布用モデルは重み・語彙・設定・manifest・LICENSEを含め約2.90MB。10万文で学習した自己回帰モデルが語彙IDを逐次予測し、全文候補や学習文の除外リストは推論に使わない。INT8は保存形式で、CPU/WASMではFP32で計算する。Python・PyTorch・CUDAはアプリ実行時に不要。

## Rustアプリへの組み込み

```toml
[dependencies]
anyhow = "1"
fumiori-core = { path = "../fumiori/crates/fumiori-core" }
fumiori-inference = { path = "../fumiori/crates/fumiori-inference" }
```

```rust,no_run
use fumiori_inference::TypingGenerator;

fn main() -> anyhow::Result<()> {
    let mut generator = TypingGenerator::load("build/fumiori-stories-v1-3mb")?;
    let options = generator.default_options();
    let prompt = generator.generate(&options)?;
    println!("{} / {}", prompt.text, prompt.kana);
    Ok(())
}
```

上のモデルパスはプロセスの作業ディレクトリからの相対パス。配布時はモデルの5ファイルをアプリの配置先に同梱し、そのパスを指定する。`TypingGenerator::from_bytes` と `include_bytes!` による実行ファイルへの埋め込みにも対応する。生成器は読み込み後に使い回す。同期処理のためGUIではワーカースレッドでの実行が適する。

`options.difficulty` に `Difficulty::Easy` / `Normal` / `Hard` を指定できる。かな長はそれぞれ5〜12 / 13〜22 / 23〜40文字。返り値は `text`、`kana`、`segments`、`source`、`model_version`、`reading_validation` を持つ。かなは助詞「は・へ・を」を正書法のまま返す。読点「、」は `kana` と各区切りの `kana` にも含み、かな長に1文字として数える。公開出力は末尾の「。」を省く。

## 生成とWebデモ

```powershell
cargo run --release -p fumiori-cli -- generate --model build/fumiori-stories-v1-3mb --difficulty mixed --count 10 --seed 3432
.venv/Scripts/python.exe scripts/build_wasm.py
.venv/Scripts/python.exe -m http.server 8000 --bind 127.0.0.1 --directory build/web
```

ブラウザで http://127.0.0.1:8000 を開く。Web Workerで生成する。ビルドには `wasm32-unknown-unknown` と `wasm-bindgen-cli 0.2.127` が必要。

### GitHub Pagesへの公開

`.github/workflows/pages.yml` が `main` へのpushでWebデモをビルドし、GitHub Pagesへ自動デプロイする。
PRではビルドと推論・ブラウザ検証のみを実行する。Actionsの「Deploy web demo to GitHub Pages」から手動実行もできる。

初回はGitHubリポジトリの **Settings → Pages → Build and deployment → Source** を **GitHub Actions** に設定する。
公開URLはデプロイジョブの出力から確認できる。プロジェクトサイトの `/fumiori/` のようなサブパスにも対応する。

公開ビルドはGit管理された `models/stories-v1-3mb/` の約2.90MBのINT8配布モデルを使う。学習・PyTorch・`build/` 内の既存ファイルは不要。
同じ構成をローカルで確認する場合:

```powershell
python scripts/build_wasm.py --runtime-model models/stories-v1-3mb
node scripts/smoke_wasm.mjs model-int8
python -m http.server 8000 --bind 127.0.0.1 --directory build/web
```

モデルの更新手順は [models/README.md](models/README.md) を参照。

## モデルと学習

| パス | 用途 |
|---|---|
| `build/fumiori-stories-v1-3mb/` / `.zip` | アプリ配布用の5ファイル |
| `build/stories-v1-3mb/` | FP32の診断用export |
| `build/stories-v1-3mb-int8/` | INT8の診断用export |
| `build/model/` / `build/model-int8/` | 同じv1モデルの標準読み込み先 |
| `build/checkpoints/stories-v1-3mb.pt` | v1の学習済みcheckpoint |
| `build/stories-v1-data/` | 学習・検証データ |
| `build/current-model.json` | 標準配置元と重みのSHA-256 |

再現手順は [学習README](training/README.md)、設計・評価は [3MBモデルの説明](docs/three-mb-generation.md)、出典は [データライセンス](docs/data-licenses.md) を参照。

## 品質と検証

確認用600文では595種類、学習にない正規化本文59.2%。無作為抽出100文のCodex点検で90文が自然さ・読みの両方に合格した。独立した人間の評価は未実施で、童話調への偏り、不自然さ・読み間違いは残る。学習文の再現や重複を完全には防がない。生成失敗やモデル破損は `Result` で返し、このモデルは静的fallbackを使わない。

```powershell
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo test --workspace --locked
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe scripts/check_default_model.py
```

[ドキュメント一覧](docs/README.md)。
