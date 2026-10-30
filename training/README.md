# Training

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
