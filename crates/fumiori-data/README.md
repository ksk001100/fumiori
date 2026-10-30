# fumiori-data

ローカルJSONLのprepareと、UTF-8/IPADIC形式ソースからのbuild-dictionary。Linderaは0.38.1、辞書内蔵なし。信頼度・ライセンス・除外理由・近似重複split・決定的語彙・manifestを生成する。コマンドと引数はリポジトリルートで `cargo run -p fumiori-data -- --help` を参照。標準モデルと最近の実験のデータ準備は [学習README](../../training/README.md) のPythonスクリプトを使う。`cargo test -p fumiori-data`。
