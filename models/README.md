# アプリ・Webデモ用の配布モデル

`everyday-v21-source10-3mb/` はv20と同じ10万文・4層を、追加した物語データの抽選比率10%で学び直した比較用モデル。出典の比率と出力のジャンル比率は区別する。デモでは「v21 物語ソース10%・4層」で選択できる。[v21記録](../docs/compositional-generation-v21.md)を参照。

`everyday-v20-{control4,layers4,layers2,layers8}-3mb/` は約5.7万文の対照4層と、10万文の2/4/8層の比較用4モデル。容量は各約2.53〜2.56MB。今回の標本では4層が品質面で先行、2層が速い。8層はCPU90件中2件で100msを超えた。全モデルをデモへ追加したが標準への採用ではない。[v20記録](../docs/compositional-generation-v20.md)を参照。

`everyday-v19-{curated,refined}-3mb/` は各2,528,193 bytesのデータ整理モデル。初版curatedは一部の意味が異なる修飾を統合した問題があり比較用に残す。最終候補はrefinedだが、整理前に対する自然な非一致文の安定改善はなく標準へは採用しない。両モデルともデモで選択できる。[v19記録](../docs/compositional-generation-v19.md)を参照。

`everyday-v18-expanded-base-3mb/` はv17で追加した2,518文を通常比率65/15/15/5で学ぶ比較用モデル。文追加と抽選比率の2×2比較を補い、新しい生成seedでも確認する。コピー率は両seed群で低下したが、自然な非一致文の増加は新seedで再現せず、標準へは昇格していない。[v18記録](../docs/compositional-generation-v18.md)を参照。

`stories-v2-diverse-3mb/` は `build/fumiori-stories-v2-diverse-3mb/` の配布用5ファイル。
GitHub Actionsは標準モデルを含む `configs/demo-models.json` の全モデルを使い、学習・データ取得なしでWebデモをビルドする。
ビルド時にmanifestのSHA-256と実ファイルの一致を検証する。

`everyday-v3-3mb/` は2,423,935 bytesの小型試作の5ファイル。以前の日常文v3を、3MB・0.1秒という最新目標に対する比較基準としてデモに使う。生成設定は点検時のtemperature 1.0、top-p 0.95、履歴128件。自然さの点検結果は良好だったが新規率は約33〜34%であり、新規生成の目標を達成したモデルとしての昇格ではない。[速度](../docs/lightweight-generation.md) と [品質の限界](../docs/everyday-generation.md) を参照。

`everyday-v8-32mb/` は過去の比較用の研究資産。最新のデモビルドには含めない。

`everyday-v14-holdout-ce-3mb/` と `everyday-v14-holdout-smooth-3mb/` は各2,527,550 bytesの比較用モデル。語の組合せを除去したv13データから同じ初期値・5,000stepで学習し、後者のみlabel smoothing 0.05を追加した。標準への採用はしていない。品質・読み・新規性の課題と実測は[v14記録](../docs/compositional-generation-v14.md)を参照。

`everyday-v15-{base,context,quality,quality-context}-3mb/` は各2,527,550 bytesの4条件の比較用モデル。乱数を分離した共通初期値から、名詞入力マスクとデータ比率変更を2×2で比較した。自然な非一致文を増やす改善は確認できず、標準には採用しない。[v15記録](../docs/compositional-generation-v15.md)を参照。

`everyday-v16-branch-{count,mass}-3mb/` は各2,527,550 bytesの2モデル。従来の行数基準と実際の抽選質量を使う分岐補正を比較した。標準には採用しないが、比較用として両方デモへ追加している。[v16記録](../docs/compositional-generation-v16.md)を参照。

`everyday-v17-{small,expanded}-3mb/` は各2,528,193 bytes。語彙・構造・5,000stepを保ち、ローカル教師の新規文を500件／2,518件追加して再学習した比較用2モデル。自然で読みが合う非一致文は同じ混合比の対照より増えたが、近似文が多く、独立人手評価もなく標準には採用しない。両モデルをデモへ追加した。[v17記録](../docs/compositional-generation-v17.md)を参照。

登録した全20モデルを含むデモのビルド（一覧は `configs/demo-models.json`）:

```powershell
python scripts/build_wasm.py --model-catalog configs/demo-models.json
node scripts/smoke_wasm.mjs --catalog configs/demo-models.json
```

アプリに組み込む場合は [GitHubの `models/stories-v2-diverse-3mb/`](https://github.com/ksk001100/fumiori/tree/main/models/stories-v2-diverse-3mb) の5ファイルをまとめて取得し、アプリの配置先に同梱する。読み込み例とCargoのGit依存の指定は [READMEの組み込み手順](../README.md#rustアプリへの組み込み) を参照。

モデルの更新時は、検証済みの標準モデルから次の手順で5ファイルを更新し、まとめてコミットする。

```powershell
python -m training.package --model build/model-int8 --output build/fumiori-stories-v2-diverse-3mb --strip-token-metadata
Copy-Item build/fumiori-stories-v2-diverse-3mb/* models/stories-v2-diverse-3mb/
python scripts/build_wasm.py --model-catalog configs/demo-models.json
node scripts/smoke_wasm.mjs --catalog configs/demo-models.json
$env:FUMIORI_REQUIRE_EXPERIMENTAL = '1'
$env:FUMIORI_REQUIRE_CATALOG = 'configs/demo-models.json'
node scripts/test_pages.cjs
```

試作モデルを追加するときは標準モデルとは別のディレクトリへ5ファイルを置き、`configs/demo-models.json` に固有のID・パス・表示名・制限を登録する。上記と同じ一覧指定で再ビルドし、WASM・実ブラウザーの読み込みと生成を確認する。Pagesのワークフローもこの一覧を参照する。PlaywrightとChromiumの準備は `.github/workflows/pages.yml` に合わせる。

出典・学習データのライセンスは同梱manifestと[データの由来](../docs/data-licenses.md)を参照。
