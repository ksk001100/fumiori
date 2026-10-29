# 32MB日本語短文モデルの研究と改善実験

調査日: 2026-10-29。配布32,000,000 bytes以内、Rust/CPU/WASM推論、固定文型を推論に使わない条件を維持する。「自然」は日本語の自然さであり、自然・風景の話題は残す。

## 調べた一次資料と、この用途への判断

| 資料 | 確認した内容 | このプロジェクトへの使い方 |
|---|---|---|
| [LLM-jp-4 Technical Report, 2026-10-14](https://arxiv.org/abs/2610.19331) | 日本語コーパスの重複除去、データ選別、コーパスごとの混合比、合成データの品質確認を記述 | 小型モデルでも、容量拡大だけでなくデータの内容と混合比を比較する。短文生成の性能が同様に上がるという実証ではない |
| [LLM-jp-4.1開発報告, 2026-09-28](https://llm-jp.nii.ac.jp/blog/llm-jp-4-1/) | SFTの候補データを小規模実験で個別に比較してから採用。データの加工で別の能力に影響する事例も報告 | 多数の変更を一度に入れて改善と断定しない。新しいデータ群の追加効果をv8と比較し、失敗も記録する |
| [Entropy-Aware On-Policy Distillation, ICML 2026](https://arxiv.org/abs/2603.07079) | reverse KLによる蒸留で多様性が落ちる問題と、教師の不確実性に応じforward KLを組み合わせる方法 | 正解率だけでなく履歴なしのseed重複を測る。論文の実験は主に数学推論であり、日本語タイピングへの効果は未検証 |
| [Stable On-Policy Distillation through Adaptive Target Reformulation, ACL Findings 2026](https://arxiv.org/abs/2601.07155) | 教師と生徒の分布の差が大きい際の不安定さと多様性の崩壊を分析 | 1.8Bの教師から約29Mの生徒へ一気にlogitを合わせる前に、保持データと混ぜた文章単位の蒸留を試す |
| [LLM-jp-3.1 instruct4開発元発表, 2025-05-30](https://llmc.nii.ac.jp/topics/llm-jp-3-1_instruct4/) | 継続事前学習と事後学習を改善した日本語モデル群。1.8Bも公開 | 今回のオフライン教師に採用。最新の4.1ではなく、手元のGPUで無量子化のまま動かせる公開日本語モデルという理由で選定 |
| [ParetoQ, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/83b17fb3369b1effa97ca5409526b02e-Abstract-Conference.html) | 低ビット量子化の精度とサイズのトレードオフを体系的に比較 | より大きな本体を4bit等で32MBへ入れる案は次の候補。現状のINT8保存を4bitへ単純変更して品質が保たれるとは判断しない |

これらの論文の手法をすべて実装したわけではない。特に今回の最初の実験はon-policy logit蒸留やVetoの再現ではなく、**オフラインの文章単位の蒸留**である。教師と生徒ではtokenizerが異なり、token IDを対応付けずにKLを取ることはしない。

## 今回の実装

1. `scripts/fetch_teacher.py`: LLM-jp-3.1-1.8b-instruct4の固定revisionの設定・tokenizer・safetensorsだけを取得。モデル本体は公開SHA-256で照合する。公開Pythonコードを実行しない。中断した取得はRangeで再開し、不完全な応答を完成品とみなさない。
2. `scripts/teacher_prose.py`: CUDA上の教師で自由な短文を生成。24分野、3つの長さ、分野ごとの複数の題材を組み合わせる。長い指示だけでは文字数を守れなかったため、既存のtrainから無作為に選ぶ3文を長さの見本にする。文型や語順は固定せず、見本と異なる内容を依頼する。指示・見本・seed・生の応答・生成設定・教師の出典を保存する。見本も指示も、推論には使わない。
3. `scripts/prepare_teacher_prose.py`: 単語辞書で解析し、人名・固有名詞・未知語・未対応subword・長さ違反等を除く。既存の全splitの正規化本文との重なりも除く。新しい本文はSHA-256で90% train、5% validation、5% testに分割する。
4. 表記と読みの辞書は**trainに現れた単語だけ**を追加する。神経モデルのsubword IDと埋め込みは変更せず、v8から追加学習できるようにする。かなは辞書による自動付与で、人による読み確認済みとは記録しない。
5. 教師、生成用の指示、全文一覧は配布推論に持ち込まない。配布ライセンスには教師の帰属も記録する。

原文の構造検査と辞書一致は、意味の正しさを判定するものではない。教師が作る文にも誤りや偏りはあり得るため、未知率・多様性・読み・自然さを分けて評価する。

## 評価方針

v8の比較基準は、独立600seedで578種類、未学習正規化本文77%、探索的な無作為100文の自然さ・読み88/100。異なるデータセットに対する「未学習率」だけを比べると不公平なので、追加学習後の参照に加え、v8の共通参照に対する率も残す。

同じseed・temperature/top-p・冒頭temperature・履歴0・3難易度を使う。候補を選ぶための探索と、採用するための確認は分ける。過去の無作為点検の失敗文を追加学習に流用し、同じ点検の点数向上を成果とすることはしない。

自然さ95/100、新規正規化本文95%、独自本文99%、配布32MB以内の事前目標を維持する。基準未達の場合はそのまま記録し、デモの試作モデルの差し替えを改善済みと称して行わない。

## データ準備と比較条件

長い指示だけを渡す最初の試験では、長さ違反・説明・童話調が混ざったため、その出力は学習に使わなかった。見本を使う最終試験は720要求、形式上抽出できた5,153文。重複や既存本文529文、未対応subword420文、長さ250文、辞書・語境界1,216文等を除き、追加train2,451文、validation125文、test100文を採用した。新規の辞書項目は902個。見本は既存trainのみから取得し、評価用文章は使わない。

教師trainから無作為100文をCodexが点検し、自然さ・読みの両方の合格は96文だった。意味の不自然なつながりが残っている。この入力点検は、追加学習後のモデルの品質評価ではない。点検した4件だけを個別に削除して点数を上げる処理はしていない。[入力点検票](everyday-v9-training-input-review.json) と [準備manifest](everyday-v9-corpus-manifest.json) を保存した。

v9はv8の重みから開始し、同じ学習率・step数・seed・データで、教師データの期待比率50%と25%を比較する。50%ではTatoeba35%、日常自作10%、既存自作5%。25%ではTatoeba60%、日常自作10%、既存自作5%。各難易度の総量は均等にする。サブワードIDと28,714,496個のパラメータ数は同じ。教師の要求した話題を、実際に生成された話題の正解ラベルとしては扱わない。

追加データが少ないため、検証全体の単純平均だけで重みを選ぶと既存コーパスの影響が大きくなる。`validation_corpus_mixture` を指定してデータ群ごとの検証損失を重み付けし、同時に各群の損失も記録する。この選択指標は自然さや多様性を保証せず、最終的な生成評価を代替しない。

## 再現手順

既存v8のcheckpoint・v6データ・rinnaの固定tokenizerが必要。通常の学習依存と `training/prose-requirements.txt` に加えて、オフライン教師用に `training/teacher-requirements.txt` を使う。

```powershell
python -m scripts.fetch_teacher
python -m scripts.teacher_prose --requests 720 --offset 72 --batch-size 16 --max-new-tokens 192 --output build/teacher-prose/corpus-01.jsonl
python -m scripts.prepare_teacher_prose --raw build/teacher-prose/corpus-01.jsonl --output build/everyday-v9-data
python -m training.train --config configs/everyday-v9-32mb.json
python -m training.train --config configs/everyday-v9-25-32mb.json
python -m training.export --checkpoint build/checkpoints/everyday-v9-32mb.pt --output build/everyday-v9-32mb --generation-mode autoregressive --polite-boundary-guard --normalize-morph-readings --lexical-boundary-guard --subword-start-temperature 1.0
python -m training.quantize --model build/everyday-v9-32mb --output build/everyday-v9-32mb-int8
python -m training.package --model build/everyday-v9-32mb-int8 --output build/everyday-v9-runtime
python -m scripts.independent_seeds --model build/everyday-v9-runtime --training-data build/everyday-v9-data/train.jsonl --comparison-training-data build/everyday-v6-data/train.jsonl --samples 600 --seed 530000 --temperature .5 --top-p .9 --output build/everyday-v9-independent.json
```

25%候補のexport・量子化・評価もモデル名を `everyday-v9-25-32mb` に置き換えて同様に行う。合成データの生成はGPU・ライブラリ・バッチ形状にも依存するため、生の応答とハッシュを保存してから下流を再現する。アプリの推論に教師用依存を追加しない。

## 600seedと無作為点検の結果

| 指標 | v8基準 | v9 教師50% | v9 教師25% |
|---|---:|---:|---:|
| 生成成功 | 600/600 | 600/600 | 600/600 |
| 独自本文数 | 578 | 528 | 570 |
| 重複数 | 22 | 72 | 30 |
| 各候補自身の学習データにない正規化本文 | 77.0% | 63.3% | 68.7% |
| v8の共通学習参照にない正規化本文 | 77.0% | 88.8% | 78.2% |
| 探索的な自然さ・読みの合格 | 88/100 | 89/100 | 87/100 |
| 配布容量 | 約30.8MB | 約30.9MB | 約30.9MB |

**どちらも不採用。標準モデルとデモの試作v8は変更していない。** 数点の品質差は、同一評価者による100文の探索的点検の揺れもあり、改善の証拠としない。目標の自然さ95/100、新規本文95%、独自本文99%を満たさないため、正式な分野・童話調の点検や昇格用の確認評価には進めなかった。

50%候補は共通参照に対する新規率だけなら上がって見えるが、その多くは新たに追加した教師データの再現であり、各候補自身の学習データを参照すると新規率は下がる。教師出力を追加しただけで未知の文章の生成能力が改善したとは言えない。25%では重複悪化は小さくなったが、意味の矛盾や不自然な因果・反復は残った。

両候補で、数値照合はFP32/INT8の許容誤差内。重み・語彙・設定・manifest・LICENSEの合計は30,927,744 bytes。今回は推論の文型や全文一覧、履歴による除外を加えて結果を改善することはしていない。測定は履歴0で、生成後の品質による除外もない。

再検証用に [比較集計](everyday-v9-comparison.json)、[50%の600出力](everyday-v9-32mb-independent-seeds.json)、[25%の600出力](everyday-v9-25-32mb-independent-seeds.json)、[50%の点検](everyday-v9-32mb-exploratory-review.json)、[25%の点検](everyday-v9-25-32mb-exploratory-review.json) を保存した。

次に優先するのは、日本語の事前学習済み本体と低ビット保存の組み合わせ。今回の結果だけで現行本体の限界が証明されたわけではないが、2,451文の追加と混合比の変更だけでは目標に届かないことは確認できた。下記の圧縮案は未実装であり、実現済みの改善として扱わない。

## 今回は実装しなかった比較候補

- **誤文の訂正・選好学習**: 生徒自身の新規rolloutを教師に訂正させ、正例・負例を作る方法。今回のv9は新規短文の追加だけで、誤文訂正やDPOを実施済みとはしない。既存の評価用600seedの出力を学習に戻さないこと、教師の訂正にも誤りがあることを検証設計に含める。
- **日本語の事前学習済み本体と4bit保存**: [LLM-jp-3-150mの公式設定](https://huggingface.co/llm-jp/llm-jp-3-150m/blob/main/config.json) は幅512・12層のLlama型。日本語で学習した本体を保持し、語彙の削減と低ビット保存で現行の約29Mパラメータより多くの容量を残す案は検討価値がある。ただし現行Rust実装のGPT-2型とはRMSNorm・RoPE・SwiGLU等が異なり、単なる重みの読み替えでは使えない。4bit化後の誤差・読み・自然さとWASMメモリを検証してから判断する。配布32MBと実行時FP32メモリは別である。
- **不確実性に応じた蒸留目的**: Entropy-Aware OPDやVetoは候補だが、本来のlogit蒸留には教師と生徒の語彙の対応が必要。今回の異なるtokenizerを持つ教師と生徒に、そのままtoken単位のKLを適用しない。まず今回のデータ群ごとの効果と多様性の変化を測る。
