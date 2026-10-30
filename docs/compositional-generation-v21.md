# v21: 追加した物語データの抽選比率を抑える比較

v20で増やした42,646文はSimpleStories由来で、表現の近似や構文の集中を抑えても、会話・説明・案内などの文体の多様さは保証できなかった。v21では同じ10万文を残し、追加したstoriesソースの学習時の比重を下げる。出典の比重を制御する実験であり、元の57,354文を非物語と認定したわけではない。

## 結果と限界

各100件の新規Codex点検で、物語調と判定した出力は30→4件へ減った。同じモデル形状・同じデータで出典の抽選を変えた結果なので、追加ソースの学び方が文体の偏りに影響しているという説明を支持する。ただし学習seedは1つで、出典内のどの特徴が原因かまでは分離していない。

| 指標 | v20 4層 | v21 source10 |
|---|---:|---:|
| 生成成功 | 600/600 | 600/600 |
| 異なる本文 | 598/600 | 598/600 |
| 学習文と完全一致・全生成 | 8.83% | 13.17% |
| 学習文と文字列近似Dice≧.8・全生成 | 23.0% | 27.5% |
| 自然で読みが合う・点検100件 | 74 | 70 |
| 物語調・点検100件 | 30 | 4 |
| 自然・読み一致・非物語・点検100件 | 54 | 70 |
| 上記に加え学習文と完全非一致 | 46 | 54 |
| 上記に加え文字列近似もなし | 43 | 38 |

物語調を抑える効果は今回の標本で見えた。一方、全体の自然さは向上せず、コピー・近似は増えた。日常文の「完全非一致」は増えたが、「近似もなし」では増えない。文字列の非一致は意味の新しさを保証しない。物語調の境界例を含めて理由を全行保存し、判定の再利用は0件。独立人手・複数学習seed・新しい生成seed群での再確認は未実施。以前のv20点検とは生成seedと点検者の判断が異なるため、過去の86/100等との差を今回のモデル劣化と断定しない。今回は同じ票・同じ判断での74対70を比較する。

例としてv21には「ベランダで洗濯物を干す」「作業の条件を記録帳に書き留める」等が出るが、これらを未学習の文の例とは主張しない。「授業でなされたすることで、一番重要な説明ができる」のような破綻も残る。比率変更を品質問題の解決や標準への採用とは扱わない。次に必要なのは、残りの小さなプールを繰り返すだけでなく、重複せず自然な説明・会話・依頼・案内の原本を増やす比較である。

候補の5ファイル合計は2,529,769 bytes。学習255.1秒、最終共通validation loss 2.575239（対照2.592931）。lossは自然さの保証ではない。実Edge normal30件はp50 29.6ms/p95 43.9ms、最大45.1ms。測定ブラウザー156.0.4314.56、seed42/履歴16、Worker内生成時間でロード・表示を含まない。CLIの比較評価中は他の検証が並走しており、専用の速度比較としては使わない。

[全測定](everyday-v21-measurements.json)、[集計CSV](everyday-v21-summary.csv)、[採点200行](everyday-v21-blind-agent-review.csv)、[対応表](everyday-v21-review-map.json)、[品質](everyday-v21-agent-review.json)、[新規性との結合](everyday-v21-joint-review.json)、[文体との結合](everyday-v21-genre-review.json)、[点検方法](everyday-v21-review-method.json)、[未記入人手票](everyday-v21-human-review.csv)、[学習記録](everyday-v21-training.json)。

## 条件

対照はv20の4層・10万文。候補は同じ4層、幅128、FF384、1,796,608パラメータ、seed42、バッチ128、最終10,000step、early stopなし。データ・語彙・評価split・難易度抽選・validation基準・生成方針は同じ。初期状態SHA-256は両者 `9a91a4df45bbf0fdaf2cd8e847583142fb90d4a5a387c83553c3b57dee6ae298`。

`corpus_group_map` でtatoeba/teacher/everyday/pairedを学習時のみbaseへまとめる。`corpus_mixture` はbase .9/stories .1。各難易度でbase内の各行を一様抽選し、出典や本文を変更しない。辞書・評価用文・モデル出力による選別を推論へ持ち込まない。

| 追加storiesの抽選比率 | v20 | v21 |
|---|---:|---:|
| Easy | 20.73% | 10% |
| Normal | 34.86% | 10% |
| Hard | 65.56% | 10% |
| 全体・難易度均等 | 40.38% | 10% |

[実際の抽選質量](everyday-v21-source-mass.json)。行数では42.646%だが、難易度均等抽選により学習確率と異なる。チャット途中の約46%という説明は誤りで、計算後に40.4%へ訂正した。元データの学習回数が増えるため、コピー率との引き換えも確認する。

新しい本文の追加はしていない。会話・説明・依頼・案内を補った10万文への置換は未実施で、この実験と区別する。手元のTatoeba原本248,941件中、表記条件を満たす225,911件は全てHrkt記録があり、未転写の別プールはなかった。ただし、それ以外の不採用理由や追加採用可能数をこの集計では判定していない。[原本件数](everyday-v21-source-inventory.json)。読み照合・語彙・長さ等の既存基準は緩めていない。

## 評価方法

生成前の[条件](everyday-v21-protocol.json)を保存。両モデルを開始seed940000、各600要求・3難易度200ずつ、履歴0、候補1、temp .8/top-p .95で比較する。コピー参照は同じ学習10万文。各モデルの100件をreview_seed9103で選び、モデル名を隠したCSVでCodexエージェントが新規点検する。独立した人手評価ではない。過去票の再利用はしない。

自然さ・文脈読みと別に `fiction_or_narrative` と理由を記入する。明示的な幻想・擬人化、少年・王などの人物を配した物語場面、筋立て・情緒的展開を含む場合に物語調とする。単なる観察・依頼・説明・一人称の日常報告は含めない。境界例は主観的で、単語だけの機械分類より広く点検するが、ジャンルの客観的正解を保証しない。自然で読みが合い、物語調でなく、学習文に完全一致しない件数も探索的に集計する。

## デモ・再実行

ローカルデモは http://127.0.0.1:8000/ 。モデル選択で「v21 物語ソース10%・4層」を選び、「読み込む」→「お題を生成」。比較対象は「v20 10万文・4層」。GitHub Pagesへの公開はこの作業では行わない。標準はv2のまま。

再学習には既存の `build/everyday-v20-selected-100k-data/` とCUDA環境が必要。既存成果を上書きせず、checkpointと各出力パスを変更して実行する。

```powershell
.venv/Scripts/python.exe -m training.train --config configs/everyday-v21-source10.json
.venv/Scripts/python.exe -m scripts.export_experiment --checkpoint build/checkpoints/everyday-v21-source10.pt --prefix build/v21-recheck --output build/v21-runtime-recheck
.venv/Scripts/python.exe -m scripts.evaluate_candidates --config configs/everyday-v21-evaluation.json --output build/v21-evaluation-recheck
.venv/Scripts/python.exe scripts/build_wasm.py --model-catalog configs/demo-models.json
node scripts/smoke_wasm.mjs --catalog configs/demo-models.json
```

新しい文を追加するときは出典・話題ラベルだけでなく、発話目的（説明・質問・依頼・案内・報告など）を点検する。内容・読み・近似文の基準を緩めて10万件を埋めることはしない。物語語彙を消すだけでは日常文の意味関係を学べないため、比率変更で足りなければ不足する文体の原本を補う比較が必要。

## 検証

Python全145テスト、標準モデルの整合性、候補のFP32/INT8数値照合、全20モデル×30件のWASM smoke、実Edge全20モデル×3難易度の読み込み・生成と標準への戻しを通過した。WASMもモデル一覧から再ビルド済み。配布の出典・MIT notice・語彙とmanifestの整合を保持。Rustコードの変更はなくworkspace全テストは再実行していない。

[数値照合](everyday-v21-parity.json)、[ブラウザー測定](everyday-v21-browser-timings.json)、[検証logのhashと未実施範囲](everyday-v21-verification.json)。

採点を再集計するには `python -m scripts.review_genre --report docs/everyday-v21-measurements.json --mapping docs/everyday-v21-review-map.json --ratings docs/everyday-v21-blind-agent-review.csv --protocol docs/everyday-v21-protocol.json --output build/v21-genre-recheck.json`。採点票本文・かな・statusの改変や文体の未記入を拒否する。
