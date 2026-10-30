# ドキュメント案内

標準配布モデルstories-v2-diverse-3mbの利用・組み込みと、v21までの比較実験・学習・評価に関する文書。

| 文書 | 内容 |
|---|---|
| [ルートREADME](../README.md) | Rustへの組み込み、CLI、Webデモ |
| [architecture.md](architecture.md) | モデル構成と公開API |
| [three-mb-generation.md](three-mb-generation.md) | データ、モデル、品質評価、再現手順 |
| [evaluation.md](evaluation.md) | 評価条件と検証コマンド |
| [compositional-generation-v21.md](compositional-generation-v21.md) | 同じ10万文の追加物語ソースを10%抽選に抑える比較、品質・新規性と文体の点検 |
| [compositional-generation-v20.md](compositional-generation-v20.md) | 近似を抑えた学習10万文、ほぼ同容量の2/4/8層と小データ対照の比較 |
| [compositional-generation-v19.md](compositional-generation-v19.md) | 重複・構文集中・確認済み誤りの整理、修正版の再学習とデモ追加 |
| [training-data-diversity-audit.md](training-data-diversity-audit.md) | 学習文の完全・近似重複、構文集中、実際の抽選質量とsplit間の近似 |
| [compositional-generation-v18.md](compositional-generation-v18.md) | 文追加×学習比率の2×2比較、新しい生成seedでの点検とデモ追加 |
| [compositional-generation-v17.md](compositional-generation-v17.md) | 新規教師文500件／2,518件の追加、共通コピー参照による比較とデモへの追加 |
| [compositional-generation-v16.md](compositional-generation-v16.md) | 行数／抽選質量による分岐補正、固定した対照評価票の再利用とデモへの追加 |
| [compositional-generation-v15.md](compositional-generation-v15.md) | 名詞入力マスク・データ比率の2×2比較、自然な非一致文の評価とデモへの追加 |
| [compositional-generation-v14.md](compositional-generation-v14.md) | 除去splitからの再学習、サンプリング・候補選別比較とデモへの追加 |
| [compositional-generation-phase1.md](compositional-generation-phase1.md) | 現状分析、コピー・多様性評価基盤、A/B/Cの段階的な改善計画 |
| [compositional-generation-phase1-results.md](compositional-generation-phase1-results.md) | 標準モデルのベースライン実測、評価票と失敗例 |
| [diversity.md](diversity.md) | 語彙の集中度測定、学習の偏り調整、候補モデルの比較 |
| [limitations.md](limitations.md) | 品質・性能・対応環境の限界 |
| [data-licenses.md](data-licenses.md) | 学習データとfixtureの出典 |
| [combination-generalization-research.md](combination-generalization-research.md) | v13の語の組合せ診断、学習データとの重なり、再学習用split |
| [学習README](../training/README.md) | 学習・export・標準配置 |
| [学習設定](../configs/README.md) | 設定ファイルと出力先 |

`stories-v2-diverse-3mb-*.json` は現行モデルの測定・点検・採用記録。v1と初期試行の記録も保持する。`current-model-check.json` は標準配置したモデルの生成確認結果。
