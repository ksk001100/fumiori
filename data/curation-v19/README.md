# v19 学習文の点検による除外

`exclusions.json` はv17-expandedのtrainだけを直接点検して確認した6行。本文と元のJSONL行のSHA-256、除外理由を保持する。5行は風を「ふう」と読む文脈上の誤り、1行は修飾の不自然さ。生成出力の評価票から修正文を学習へ追加したものではない。

元行は `build/everyday-v17-expanded-data/train.jsonl` に保持する。全てローカル教師による合成文で、元行のsourceに教師revision・原本hashがある。コーパス全体や教師出力をCC0とする主張ではなく、親manifestとモデルLICENSEの各出典を継承する。

追加で「自殺」を含む29行をタイピング用の学習subsetから隔離する方針を `configs/everyday-v19-data.json` に指定した。この題材方針と文法・自然さのラベルは区別する。否定・予防の文も隔離対象であり、誤った日本語だと判定したものではない。

詳しい整理方針、保持・除外の件数、比較と制限は [v19記録](../../docs/compositional-generation-v19.md) を参照。除外表・近似群・文型群は推論へ配布しない。

整理済みの最終データは `build/everyday-v19-refined-data/`、保持・除外対応は同ディレクトリの `curation.json`。`configs/everyday-v19-refined-data.json` から再作成できるが、merged/outputは新しいパスへ変更する。旧データと評価splitは保持する。
