# Training

[v19のデータ整理](../docs/compositional-generation-v19.md)は `scripts.curate_training_data` で元trainを保持したまま別のsubsetを作る。最終設定は `configs/everyday-v19-refined-data.json`、データは `build/everyday-v19-refined-data/`、学習は `configs/everyday-v19-refined.json`。除外理由・元行hash・代表行を追跡し、評価splitと語彙は維持する。初版には修飾の誤統合があり、修正版とともにデモへ残した。生成の自然な非一致文に安定した改善はなく、標準へは採用しない。

[学習データの多様性監査](../docs/training-data-diversity-audit.md)は完全重複・1単語差・文字列近似・名詞を伏せた構文群を調べ、現在の抽選比率での偏りも計算する。`scripts.audit_data_diversity` はオフライン診断のみで、データを削除したり推論を変更したりしない。

[v18](../docs/compositional-generation-v18.md)はv17の追加データを通常比率65/15/15/5で学び、文追加×抽選比率の2×2比較を完成する。語彙・構造・初期値・最終5,000stepを維持する。既存seedで4条件、新しい887000台で通常比率の対照と追加モデルを評価し、採点票の再利用と新規点検を区別する。新モデルもデモへ追加した。新seedでは自然な非一致文の増加が再現せず、標準には採用しない。

[v17](../docs/compositional-generation-v17.md)はローカル教師の新規文500件／2,518件を追加し、同じ語彙・構造・最終5,000stepで比較する。`scripts.prepare_word_teacher` が原本・few-shot train由来・辞書・語彙hashを照合し、旧splitと既存評価出力の重複・評価用組合せを除外する。新teacher-testは学習に含めない。比較のコピー参照は各条件trainの和集合で、`evaluate_candidates` の `common_reference_data` と条件別 `data` で指定する。全モデルをデモへ追加した。

[v16](../docs/compositional-generation-v16.md)は分岐補正の頻度を数える基準を比較する。`branch_balance.frequency_source` は省略時 `uniform_rows`、`sampling_mass` では難易度・corpus mixture等を適用した実際のtrain行抽選質量を使う。明示的な抽選重みがない場合は後者を拒否する。表は学習時だけ使い、配布モデルへ入れない。試作2モデルはデモへ追加している。

[v15](../docs/compositional-generation-v15.md)は名詞入力マスクとcorpus mixtureを2×2で比較する。`independent_data_rng: true` により学習文抽選と入力マスクの乱数を分離し、初期重みと更新batch行番号列のSHA-256をcheckpointへ保存する。同じ混合比の対照実験でマスクの有無が抽選を変えないことを検査できる。省略した既存設定の乱数経路は従来どおり。試作4モデルは全てデモに追加している。

[v14](../docs/compositional-generation-v14.md)は既存v13の除去splitから通常CEと平滑化CEを同じ初期値・学習量で比較する。`configs/everyday-v14-holdout-{ce,smooth}.json` が学習設定、`configs/everyday-v14-evaluation.json` がサンプリング・候補選別の比較設定。新モデル2種類は比較用としてデモへ追加している。

構成的一般化の研究は[Phase 1の現状分析・評価手順・改善計画](../docs/compositional-generation-phase1.md)を参照。既存の学習を変更せず、標準モデルの部分コピー・文字／単語新規性と自然さ・読みの評価票を追加している。[実測結果](../docs/compositional-generation-phase1-results.md)は学習改善の成果と区別する。

現行モデルは `configs/stories-v2-diverse-3mb.json` / `scripts/stories_pipeline.py` を使用する。日本語SimpleStoriesの公開trainの一部と対応付き散文を合わせ、生成条件を満たす30,800文で学習する。

```powershell
py -3.12 -m pip --python .venv/Scripts/python.exe install -r training/stories-requirements.txt
.venv/Scripts/python.exe scripts/fetch_stories.py
.venv/Scripts/python.exe scripts/stories_pipeline.py
# 既存のv2 checkpointを使ってexportと評価を再実行
.venv/Scripts/python.exe scripts/stories_pipeline.py --skip-training
# 開発・確認の採用条件を通過した場合だけ標準へ配置
.venv/Scripts/python.exe scripts/stories_pipeline.py --skip-training --activate
```

CUDA版PyTorchと `training/requirements.txt` の依存も必要。学習設定はCUDAを指定し、GPUが使えなければエラーになる。推論にはPython・PyTorch・CUDAは不要。

checkpointは `build/checkpoints/stories-v2-diverse-3mb.pt`、データは `build/stories-v2-data`、exportは `build/stories-v2-diverse-3mb` / `build/stories-v2-diverse-3mb-int8`。配布用は `build/fumiori-stories-v2-diverse-3mb`。標準の `build/model` / `build/model-int8` も同じv2を参照する。`python -m training.activate` の既定もv2で、標準の読み込み先に配置する。

自作の対応付き散文は `data/paired-v1` に収録する。入力のSHA-256を照合してから、日本語SimpleStoriesの短文を追加する。

構造はcausal mask、pre-LayerNorm、GELU、tied embedding。PADを無視したnext-token cross entropyで学習し、validation lossでcheckpointを選択する。点検票とモデル・語彙・生成方針のhashが合わなければ採用しない。出典・容量・評価・再現条件は [3MBモデルの説明](../docs/three-mb-generation.md) を参照。

語彙の偏りを抑える候補の学習・比較手順は[多様性改善](../docs/diversity.md)を参照。`lexical_balance` は学習splitのみから重みを計算し、文章自体を書き換えない。指定しない既存設定のサンプリング動作は従来どおり。

最新の3MB実験は [小型モデルの改善実験](../docs/compact-generation-research.md)。`branch_balance` はtrainの同一prefixに対する次トークンの頻度から学習lossの重みを作る。推論ファイルには含めない。今回の設定では通常学習より改善せず、標準への採用は行っていない。

続く [v12の修正学習](../docs/compact-repair-research.md) では `repair_auxiliary` により、通常のCEへ複数修正のsuffix CEと誤り箇所のunlikelihoodを加えられる。明示的に `allow_model_reviewed_repairs` を指定し、親データと修正splitのhashを照合する。修正表は推論へ配布しない。研究用であり、今回の候補は採用していない。

修正データ準備の単体テストは形態素解析をモック化し、SudachiPy未導入でも保留データの除外と学習用seedの検証を実行する。実辞書を使う解析テストには `training/prose-requirements.txt` の依存が必要で、未導入時はスキップする。

[v13の組合せ診断](../docs/combination-generalization-research.md)は100対の共起検査と再学習用splitを用意する。`scripts.prepare_combination_holdout` が付ける `combination_holdout` manifestでは、`initialize_from` と `repair_auxiliary` を拒否し、`holdout.json` のSHA-256を検証する。元データを学習済みの重みから続けても未見の組合せにはならないため、共通の乱数初期値から比較する。現時点で再学習による転移の確認は未実施。
