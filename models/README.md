# アプリ・Webデモ用の配布モデル

`stories-v2-diverse-3mb/` は `build/fumiori-stories-v2-diverse-3mb/` の配布用5ファイル。
GitHub Actionsはこのモデルを使い、学習・データ取得なしでWebデモをビルドする。
ビルド時にmanifestのSHA-256と実ファイルの一致を検証する。

`everyday-v8-32mb/` は比較用の試作モデルの5ファイル。標準モデルへの昇格ではなく、デモでv2と切り替えて試すために同梱する。生成設定は評価時と同じtemperature 0.5、top-p 0.9、冒頭temperature 1.0。デモでは履歴16件の重複回避を使う。履歴なしでの評価と品質の限界は [実験結果](../docs/everyday-generation.md) を参照。

両方を含むデモのビルド:

```powershell
python scripts/build_wasm.py --runtime-model models/stories-v2-diverse-3mb --experimental-model models/everyday-v8-32mb
node scripts/smoke_wasm.mjs model-int8 model-experimental
```

アプリに組み込む場合は [GitHubの `models/stories-v2-diverse-3mb/`](https://github.com/ksk001100/fumiori/tree/main/models/stories-v2-diverse-3mb) の5ファイルをまとめて取得し、アプリの配置先に同梱する。読み込み例とCargoのGit依存の指定は [READMEの組み込み手順](../README.md#rustアプリへの組み込み) を参照。

モデルの更新時は、検証済みの標準モデルから次の手順で5ファイルを更新し、まとめてコミットする。

```powershell
python -m training.package --model build/model-int8 --output build/fumiori-stories-v2-diverse-3mb --strip-token-metadata
Copy-Item build/fumiori-stories-v2-diverse-3mb/* models/stories-v2-diverse-3mb/
python scripts/build_wasm.py --runtime-model models/stories-v2-diverse-3mb
node scripts/smoke_wasm.mjs model-int8
```

出典・学習データのライセンスは同梱manifestと[データの由来](../docs/data-licenses.md)を参照。
