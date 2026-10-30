# AGENTS.md

このファイルはリポジトリ全体で作業するエージェント向けの開発指針です。

## 基本方針

- ユーザーへの説明とドキュメントは原則として日本語で記述する。
- 変更前に関連コード・テスト・READMEを読み、既存の設計と命名に合わせる。
- 依頼に必要な範囲に変更を絞り、無関係なリファクタリングや生成物の更新を混ぜない。
- 挙動や手順を変える場合は、対応するREADMEや `docs/` も更新する。
- 実施した検証と未実施の検証を区別し、測定していない品質や性能を達成済みと記述しない。

## プロジェクト構成

Fumioriは日本語タイピング用の短文を、本文・かな・品詞付きの区切りとして生成するプロジェクト。

| パス | 役割 |
|---|---|
| `crates/fumiori-core/` | 共通型、表記・かな・区切りの構造検証、ハッシュ |
| `crates/fumiori-inference/` | CandleによるCPU推論、モデル読み込み、生成と履歴管理 |
| `crates/fumiori-cli/` | 生成・評価などのCLI |
| `crates/fumiori-data/` | 学習データの準備、語彙・split・manifestの作成 |
| `crates/fumiori-wasm/` | ブラウザー向けWASM API |
| `training/` | PyTorchによる学習・export・量子化・評価とPythonテスト |
| `scripts/` | データ取得・実験・検証・WASMビルド |
| `web/` | WebデモのHTMLとWeb Worker |
| `configs/` | モデルと実験の設定 |
| `models/` | 配布モデルと比較用モデル |
| `docs/` | 設計・評価・研究記録・ライセンス情報 |
| `build/` | ローカル生成物。Git管理対象外 |

設計は `docs/architecture.md`、学習は `training/README.md`、配布モデルの更新は `models/README.md` を参照する。`SPEC.md` や過去の研究記録を読む場合は、現行READMEと後続の実験記録も確認する。

## 維持する仕様

- 標準配布モデルは `models/stories-v2-diverse-3mb/`。`models/everyday-v3-3mb/` は小型の日常文の比較用試作であり、標準への昇格済みモデルではない。
- 現在の研究目標は配布モデル合計 **3,000,000 bytes以内**、難易度normalのお題1件 **0.1秒以内**。測定条件と限界は `docs/lightweight-generation.md` を参照する。
- アプリ実行時にPython・PyTorch・CUDAを要求しない。INT8/INT4は保存形式であり、現行CPU/WASM推論はFP32演算を行う。
- 全文候補、固定文型、学習文一覧やその除外リストを推論に追加しない。単語の綴りを制約する辞書trieとは区別する。
- 標準モデルで生成に失敗した場合はエラーを返す。静的fallbackで成功扱いにしない。
- 公開出力の末尾の「。」は省き、区切りの連結を本文・かなと一致させる。
- 助詞「は・へ・を」のかなは正書法を維持する。読点「、」はかなにも含め、かな長に1文字として数える。
- 難易度別のかな長はEasyが5〜12、Normalが13〜22、Hardが23〜40文字。
- 構造検証の合格を、自然さや文脈的な読みの正しさの保証として扱わない。

## モデル・データを変更するとき

- 実験で作成した推論可能なモデルは、実験ごとに必ずWebデモへ追加する。試作・比較用・標準採用を見送ったモデルも対象とし、評価結果の良し悪しだけを理由にデモへの追加を省略しない。各モデルを識別して選択できるようにし、実験条件・評価結果・既知の制限を利用手順に記録する。標準モデルへの昇格は別途の採用ゲートに従う。
- モデルを更新・追加したら、ユーザーがWebデモでそのモデルを選択・読み込み・生成して試せる状態まで整える。必要なモデル配置、選択UI、ビルドスクリプト、GitHub Pagesのワークフロー、テスト、利用手順も合わせて更新する。試作モデルは比較用として追加し、標準への採用条件とは区別する。
- 更新・追加したモデルを含めてデモを再ビルドし、WASMのスモークテストと実ブラウザーで読み込み・生成を確認する。完了報告には試せるURLと操作手順を記載し、未検証や起動できない点があれば明示する。
- 配布モデルは `model.safetensors`、`vocab.json`、`model-config.json`、`manifest.json`、`LICENSE` の5ファイルを整合した状態で扱う。manifestのSHA-256検証を維持する。
- 学習・データ取得・標準モデルの切り替えは、必要な作業として依頼された場合に実施する。通常のコード修正やデモ確認のために再学習しない。
- 標準への採用は、学習READMEと関連パイプラインの開発・確認ゲートに従う。評価票とモデル・語彙・生成方針のハッシュの対応を保つ。
- データを追加・変更する場合は出典、ライセンス、splitの扱いを確認し、必要に応じて `docs/data-licenses.md` とmanifestを更新する。
- 研究用の大型モデルや過去の評価結果を、現在の容量目標や採用済み品質の根拠として扱わない。
- `target/`、`build/`、`.venv/`、キャッシュをコミットしない。既存のcheckpointや実験成果を不用意に削除・上書きしない。

## 検証

コマンドはリポジトリルートで実行する。変更対象に応じた検証を選び、依存関係やモデルがなく実行できない場合はその理由を報告する。

Rustの変更時:

```powershell
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo test --workspace --locked
```

Pythonの学習・データ処理・検証スクリプトを変更した場合は、既存の `.venv` を利用して関連テストを実行する。全Pythonテストと標準モデルの整合性確認:

```powershell
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe scripts/check_default_model.py
```

WASM・Webデモや配布モデルを変更した場合:

```powershell
python scripts/build_wasm.py --runtime-model models/stories-v2-diverse-3mb --experimental-model models/everyday-v3-3mb
node scripts/smoke_wasm.mjs model-int8 model-experimental
$env:FUMIORI_REQUIRE_EXPERIMENTAL = '1'
node scripts/test_pages.cjs
```

WASMビルドには `wasm32-unknown-unknown` と `wasm-bindgen-cli 0.2.127` が必要。ブラウザーテストのPlaywrightとChromiumのセットアップは `.github/workflows/pages.yml` に合わせる。WebデモはWeb Workerで生成し、GitHub Pagesの `/fumiori/` のようなサブパスでも動作させる。

推論や生成制約の変更では、関係する回帰テストに加えて必要に応じて実モデルで生成を確認する。性能比較ではモデル、seed、難易度、履歴、生成設定、実行環境と計測範囲を記録する。
