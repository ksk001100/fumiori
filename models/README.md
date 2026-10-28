# アプリ・Webデモ用の配布モデル

`stories-v2-diverse-3mb/` は `build/fumiori-stories-v2-diverse-3mb/` の配布用5ファイル。
GitHub Actionsはこのモデルを使い、学習・データ取得なしでWebデモをビルドする。
ビルド時にmanifestのSHA-256と実ファイルの一致を検証する。

アプリに組み込む場合は [GitHubの `models/stories-v2-diverse-3mb/`](https://github.com/ksk001100/fumiori/tree/main/models/stories-v2-diverse-3mb) の5ファイルをまとめて取得し、アプリの配置先に同梱する。読み込み例とCargoのGit依存の指定は [READMEの組み込み手順](../README.md#rustアプリへの組み込み) を参照。

モデルの更新時は、検証済みの標準モデルから次の手順で5ファイルを更新し、まとめてコミットする。

```powershell
python -m training.package --model build/model-int8 --output build/fumiori-stories-v2-diverse-3mb --strip-token-metadata
Copy-Item build/fumiori-stories-v2-diverse-3mb/* models/stories-v2-diverse-3mb/
python scripts/build_wasm.py --runtime-model models/stories-v2-diverse-3mb
node scripts/smoke_wasm.mjs model-int8
```

出典・学習データのライセンスは同梱manifestと[データの由来](../docs/data-licenses.md)を参照。
