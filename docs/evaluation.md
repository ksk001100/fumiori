# 評価

完全一致以外のコピー・文字／単語n-gram・Self-BLEUと評価票の実行手順は[構成的一般化のPhase 1](compositional-generation-phase1.md)、標準モデルの新しいベースラインは[測定結果](compositional-generation-phase1-results.md)を参照する。これらの診断は採用ゲートの置き換えではない。

現行モデルは **stories-v2-diverse-3mb**。生成設定はtemperature=0.8、top_p=0.95。FP32/INT8重みのSHA-256、PyTorch/Candleのlogits整合性、固定seedの生成結果を確認する。

開発seed42/2026/73、確認seed3432/811/1907、各seed200文。確認600文では584種類、学習にない正規化本文319件（53.2%）。無作為抽出100文のCodex点検では90文が自然さ・読みの両方に合格し、未知本文の部分は53/61。独立した人間の評価ではない。

```powershell
cargo test --workspace --locked
.venv/Scripts/python.exe -m pytest -q
cargo run --release -p fumiori-cli -- parity --model build/model
cargo run --release -p fumiori-cli -- parity --model build/model-int8
.venv/Scripts/python.exe scripts/check_default_model.py
.venv/Scripts/python.exe scripts/stories_gate.py --require-pass
```

モデル・語彙・生成方針・点検対象のhashを照合し、別の重みに既存の点検票を流用しない。採用条件は配布3MB以内、600件すべてモデル出力、新規正規化本文50%以上、異なる本文90%以上、点検100文中90文以上合格、未知本文を30文以上点検して85%以上合格。

[改善内容と再現手順](diversity.md)、[開発結果](stories-v2-diverse-3mb-development-measurements.json)、[確認結果](stories-v2-diverse-3mb-confirmation-measurements.json)、[採用判定](stories-v2-diverse-3mb-promotion.json)。
