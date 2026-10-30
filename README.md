# Fumiori — Tiny Japanese Typing Generator

日本語タイピング用の短文を、本文・かな・品詞付きの区切りとして生成するRustライブラリ。
標準配布モデルは **stories-v2-diverse-3mb**。最新の比較用モデルは **everyday-v21-source10-3mb** で、標準への採用はしていない。

配布用モデルは重み・語彙・設定・manifest・LICENSEを含め約2.90MB。30,800文で学習した自己回帰モデルが語彙IDを逐次予測し、全文候補や学習文の除外リストは推論に使わない。INT8は保存形式で、CPU/WASMではFP32で計算する。Python・PyTorch・CUDAはアプリ実行時に不要。

## Rustアプリへの組み込み

```toml
[dependencies]
anyhow = "1"
fumiori-core = { git = "https://github.com/ksk001100/fumiori.git" }
fumiori-inference = { git = "https://github.com/ksk001100/fumiori.git" }
```

モデルは [GitHubリポジトリの `models/stories-v2-diverse-3mb/`](https://github.com/ksk001100/fumiori/tree/main/models/stories-v2-diverse-3mb) にある5ファイル（`model.safetensors`、`vocab.json`、`model-config.json`、`manifest.json`、`LICENSE`）を取得し、アプリの `models/stories-v2-diverse-3mb/` に配置する。CargoのGit依存とは別にモデルの配置が必要。

```rust,no_run
use fumiori_inference::TypingGenerator;

fn main() -> anyhow::Result<()> {
    let mut generator = TypingGenerator::load("models/stories-v2-diverse-3mb")?;
    let options = generator.default_options();
    let prompt = generator.generate(&options)?;
    println!("{} / {}", prompt.text, prompt.kana);
    Ok(())
}
```

上のモデルパスはプロセスの作業ディレクトリからの相対パス。配布時はモデルの5ファイルをアプリの配置先に同梱し、そのパスを指定する。`TypingGenerator::from_bytes` と `include_bytes!` による実行ファイルへの埋め込みにも対応する。生成器は読み込み後に使い回す。同期処理のためGUIではワーカースレッドでの実行が適する。

`options.difficulty` に `Difficulty::Easy` / `Normal` / `Hard` を指定できる。かな長はそれぞれ5〜12 / 13〜22 / 23〜40文字。返り値は `text`、`kana`、`segments`、`source`、`model_version`、`reading_validation` を持つ。かなは助詞「は・へ・を」を正書法のまま返す。読点「、」は `kana` と各区切りの `kana` にも含み、かな長に1文字として数える。公開出力は末尾の「。」を省く。

## 生成とWebデモ

リポジトリを取得し、ルートディレクトリで実行する。同梱モデルを使うため、学習は不要。

```powershell
git clone https://github.com/ksk001100/fumiori.git
cd fumiori
```

```powershell
cargo run --release -p fumiori-cli -- generate --model models/stories-v2-diverse-3mb --difficulty mixed --count 10 --seed 3432
python scripts/build_wasm.py --model-catalog configs/demo-models.json
python scripts/serve_demo.py
```

ブラウザで [ローカルデモ](http://127.0.0.1:8000/)を開く。Web Workerで生成する。ビルドには `wasm32-unknown-unknown` と `wasm-bindgen-cli 0.2.127` が必要。

「モデル」で標準v2、小型の日常文v3、v14〜v21の試作を選び、「読み込む」→「お題を生成」で試せる。選んだモデルだけをダウンロードする。モデル一覧は `configs/demo-models.json` で管理する。v21は同じ10万文・4層で、追加した物語ソースの抽選を各難易度10%に抑えた約2.53MBの比較用モデル。会話・説明などの新規データを補った版ではなく、出力全体の物語率が10%以下になる保証もない。v20は10万文の2/4/8層と小データ対照の各約2.53〜2.56MBの4モデルで、標準への採用はしていない。[v21の物語ソース比率の比較](docs/compositional-generation-v21.md)、[v20の比較結果と制限](docs/compositional-generation-v20.md)、[v19の整理内容と比較](docs/compositional-generation-v19.md)、[v18の結果と制限](docs/compositional-generation-v18.md)、[v17の結果と制限](docs/compositional-generation-v17.md)、[v16の実験結果と制限](docs/compositional-generation-v16.md)、[v15](docs/compositional-generation-v15.md)、[v14](docs/compositional-generation-v14.md)を参照。標準は引き続きv2。目標はモデル1組の配布5ファイル合計3,000,000 bytes以内・ふつうのお題0.1秒以内。デモの全モデルを合わせた配信容量とは区別する。[軽量化と速度](docs/lightweight-generation.md)、[日常文モデルの実験](docs/everyday-generation.md)も参照。

### GitHub Pagesへの公開

`.github/workflows/pages.yml` が `main` へのpushでWebデモをビルドし、GitHub Pagesへ自動デプロイする。
PRではビルドと推論・ブラウザ検証のみを実行する。Actionsの「Deploy web demo to GitHub Pages」から手動実行もできる。

初回はGitHubリポジトリの **Settings → Pages → Build and deployment → Source** を **GitHub Actions** に設定する。
公開URLはデプロイジョブの出力から確認できる。プロジェクトサイトの `/fumiori/` のようなサブパスにも対応する。

公開ビルドは `configs/demo-models.json` に登録した標準v2、比較用v3、実験用v14〜v21を含む全20モデルを使う。学習・PyTorch・`build/` 内の既存ファイルは不要。
同じ構成をローカルで確認する場合:

```powershell
python scripts/build_wasm.py --model-catalog configs/demo-models.json
node scripts/smoke_wasm.mjs --catalog configs/demo-models.json
python scripts/serve_demo.py
```

モデルの更新手順は [models/README.md](models/README.md) を参照。

## モデルと学習

| パス | 用途 |
|---|---|
| [`models/stories-v2-diverse-3mb/`](models/stories-v2-diverse-3mb/) | GitHubに同梱されたアプリ配布用の5ファイル |
| `build/fumiori-stories-v2-diverse-3mb/` / `.zip` | アプリ配布用の5ファイル |
| `build/stories-v2-diverse-3mb/` | FP32の診断用export |
| `build/stories-v2-diverse-3mb-int8/` | INT8の診断用export |
| `build/model/` / `build/model-int8/` | 同じv2モデルの標準読み込み先 |
| `build/checkpoints/stories-v2-diverse-3mb.pt` | v2の学習済みcheckpoint |
| `build/stories-v2-data/` | 学習・検証データ |
| `build/current-model.json` | 標準配置元と重みのSHA-256 |

再現手順は [学習README](training/README.md)、最新の実験は [v21](docs/compositional-generation-v21.md)、標準v2の採用時の改善と測定は [多様性改善](docs/diversity.md)、従来設計は [3MBモデルの説明](docs/three-mb-generation.md)、出典は [データライセンス](docs/data-licenses.md) を参照。

## 品質と検証

標準v2の採用時の確認用600文では584種類、学習にない正規化本文53.2%。無作為抽出100文のCodex点検で90文が自然さ・読みの両方に合格した。独立した人間の評価は未実施で、童話調への偏り、不自然さ・読み間違いは残る。学習文の再現や重複を完全には防がない。生成失敗やモデル破損は `Result` で返し、このモデルは静的fallbackを使わない。

最新v21とv20の4層を比べた各100件のCodex点検では、物語調は30→4件、自然・読み一致・非物語・学習文と完全非一致は46→54件。一方、全600件の学習文との完全一致率は8.83→13.17%へ増え、文字列近似も除いた日常の良文は43→38件だった。文体の偏りを抑える効果と、自然さ・新規性の改善は区別する。独立人手・複数学習seedでの確認は未実施。[条件と全結果](docs/compositional-generation-v21.md)を参照。

Rust・Pythonの開発用検証は次のとおり。Pythonは既存のvenvと `training/requirements.txt` を使用する。辞書を使うテストは `training/prose-requirements.txt` も必要で、SudachiPyまたは辞書がなければ該当テストだけスキップする。

```powershell
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo test --workspace --locked
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe scripts/check_default_model.py
```

[ドキュメント一覧](docs/README.md)。
