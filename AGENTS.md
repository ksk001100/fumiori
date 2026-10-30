# AGENTS.md

このリポジトリはFumioriのRustライブラリ（共通型・推論・CLI・WASM）を管理する。
説明とドキュメントは原則日本語。変更前に関連コード・テスト・READMEを読み、既存の設計と命名に合わせる。
変更範囲を依頼に絞り、手順変更時はREADME/docsも更新する。実施・未実施の検証を区別する。

モデル・学習・fumiori-data・評価・比較デモは https://github.com/ksk001100/fumiori-models へ移した。
構成はREADME.md、APIはdocs/architecture.md、連携はdocs/repository-split.mdを参照。

## 維持する仕様

- 標準モデルはstories-v2-diverse-3mb。比較用試作を標準へ自動昇格しない。
- アプリ実行時にPython・PyTorch・CUDAを要求しない。INT8/INT4は保存形式で、CPU/WASM演算はFP32。
- 全文候補・固定文型・学習文一覧・除外リストを推論に追加しない。単語辞書trieとは区別する。
- 標準モデルの生成失敗はエラー。静的fallbackで成功扱いにしない。明示的fallback APIの既存fixtureは保持する。
- 公開出力末尾の「。」を省き、区切りの連結と本文・かなを一致させる。
- 助詞「は・へ・を」は正書法。読点「、」はかなにも含め、1文字として数える。
- かな長はEasy 5〜12、Normal 13〜22、Hard 23〜40文字。
- 構造検証は自然さ・文脈的な読みの正しさの保証ではない。
- manifestのSHA-256と形式検証を維持する。モデルは重み・語彙・設定・manifest・LICENSEの5ファイル一組。
- モデル変更を伴う場合はモデル側のAGENTS.mdと採用ゲートに従い、試作も比較デモへ追加し、全モデルのWASM・実ブラウザー検証を行う。
- 通常の修正のために再学習しない。build/、target/、.venv/、キャッシュをコミットせず、既存checkpointや実験成果を削除・上書きしない。

## 検証

```powershell
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo test --workspace --locked
cargo build --locked --release -p fumiori-wasm --target wasm32-unknown-unknown
```

推論や制約変更時は回帰テストと実モデルで生成を確認する。モデル連携のCIは固定コミットのモデルを取得する。
WASMにはwasm32-unknown-unknown、JS生成にはwasm-bindgen-cli 0.2.127が必要。
性能比較ではモデル・seed・難易度・履歴・生成設定・環境・計測範囲を記録する。
研究目標3,000,000 bytes・Normal 0.1秒は達成済み品質の保証ではない。条件はモデル側docs/lightweight-generation.mdを参照。
