# v13：語の組合せの評価と漏洩検査

2026-10-30。[次の実験計画](next-model-experiments.md)の優先1に着手し、100種類の対照例、既存checkpointの確率診断、再学習用の除去splitを作成した。新しいモデルの学習・配布・標準切替は実施していない。今回の結果は文章品質の改善を示すものではない。

## 既知の組合せなら選べるか

Codexが「水を飲む／水を読む」等の100対を作成・点検した。外部文例集や評価seedの出力は使っていない。文脈を補わない日常的・字義どおりの解釈を条件とし、比喩や擬人化まで単語単位で禁止しない。独立した人の評価ではない。例は[TSV](../data/combinations-v13/pairs.tsv)で全件確認できる。

既存語彙の表記・かな・品詞と一致した91対を、同じ条件で4つのFP32 checkpointに入力した。残り9対の必要な語形は語彙外だった。100対すべてを採点したとは扱わない。自然な側のかな長から難易度トークンを決め、負例にも**同じ難易度**を使う。

| checkpoint | 文脈ありで自然な側を選好 | BOS＋難易度だけの選好 | 文脈により対数オッズが増えた対 | 平均対数オッズ |
|---|---:|---:|---:|---:|
| v11追加学習・比較用 | 87/91 | 43/91 | 80/91 | 7.863 |
| v12修正SFT | 89/91 | 46/91 | 83/91 | 7.625 |
| v12修正SFT＋UL | 89/91 | 46/91 | 83/91 | 7.625 |
| v12大型本体・4bit QAT由来 | 86/91 | 43/91 | 86/91 | 7.620 |

共通prefixを与え、**初めて異なる次トークン**の `log p(自然な側) − log p(不自然な側)` を測った。両側で条件は同一で、全文尤度の長さの差を混ぜない。これは最初の異なるトークンだけの診断で、文末までの自然さや自由生成での正答率ではない。BOS＋難易度の条件は単語の事前選好を調べる粗い比較で、格や語順を揃えた厳密な対照ではない。

SFTとULは同じ2対（「コップを洗う／履く」「椅子に座る／咲く」）を比較用から修正し、逆方向の変化は0対。2例だけの差を改善の証拠とはしない。平均対数オッズはむしろ低下した。「卵を焼く／着る」「宿題を終える／飲む」は3つの小型checkpointとも誤った側を選好した。前回の自由生成で自然さ93/100だったモデルも、この簡単な対照で89/91となる。異なる評価を同じ品質尺度として扱わない。

全入力、除外理由、各対の確率、モデル・語彙・スクリプトのhashは[診断JSON](everyday-v13-combination-audit.json)。QAT候補も保存checkpointをFP32モデルにロードした診断で、最終INT4保存モデルの推論結果ではない。

## 高得点を未見への転移とは扱えない

親データの全4splitとv12修正データのtrain/validationを走査すると、自然な側100種類のうち**86種類**に共起があった。86種類すべて親trainにも現れていた。既存checkpointは既にこのデータを学習しているため、後から評価例を作っても未見の組合せにはならない。

検出は「正規化した名詞＋隣接する助詞＋同じ句読点区間の後続動詞」。活用形・辞書で正規化される表記差をまとめ、サ変名詞＋するを複合述語として扱う。「運転する」と「掃除する」は同じ「する」に潰さない。係り受け解析ではなく、過剰除去を許す共起検査である。負例側も1種類に共起があったが、例えば「本を読みながらコーヒーを飲んでいる」から「本・を・飲む」を拾う。この一致を元文の誤りとは判定しない。

句読点をまたぐ依存関係、受身・使役の格変化、同義語や言い換えによる意味上の漏洩は検出しきれない。「語形の共起がない」と「意味的に未見」を区別する。

## 再学習用のsplitと保護

自然側・不自然側の両方の共起がある行を親データから除いた。train **3,006行**、validation **348行**、test **298行**、family-test **7行**を除去し、trainは**56,593行**となった。複数の組合せに該当する行も1回だけ数える。追加した対照文は学習splitに入れていない。

別途、準備後の全splitを再走査し、対象200キーの一致が0件であることを確認した。[準備サマリー](everyday-v13-holdout-preparation.json)にmanifest、除去件数、100対の構成語被覆を、[再検査JSON](everyday-v13-holdout-audit.json)に各対の一致とsplitのhashを保存した。行番号付きの全除去記録はローカル出力の `holdout.json` に残る。

除去後も名詞・自然側の述語・負例の述語がそれぞれtrainに残るのは**96/100種類**。「沸く」を含む3対と「計る」を含む1対は構成語が欠けるため、既知語の未見組合せとして採点できない。語彙外を除いた今回の91対には、この4対は含まれない。100種類すべてを満たす完成した未見評価ではない。

語彙は元の8,802 IDとmetadataを保持する。語彙の選択と頻度には元コーパスの情報があるため、文の組合せを分ける診断に用途を限定する。親validation/testも除去したので、元のvalidation lossと直接比較しない。

このmanifestを `training.train` に渡した場合、既存checkpointの `initialize_from` と未検査の `repair_auxiliary` を拒否し、除去レポートのhashを検証する。今後の比較は同じ乱数初期値から学び直す必要がある。新たな教師文や別の学習ソースを加える場合にも同じ除去と検査が必要で、今回の保護だけで任意のデータ混入を防ぐわけではない。

## 再現と検証

既存の `.venv` とv11準備データ・v12 checkpointを利用する。外部データの取得はない。出力先は新しいファイル・ディレクトリを指定する。

```powershell
.venv/Scripts/python.exe -m scripts.audit_combinations --source data/repairs-v12/prepared/train.jsonl --source data/repairs-v12/prepared/validation.jsonl --checkpoint build/checkpoints/everyday-v11-quality.pt --checkpoint build/checkpoints/everyday-v12-repair-sft.pt --checkpoint build/checkpoints/everyday-v12-repair-ul.pt --checkpoint build/checkpoints/everyday-v12-qat4.pt --output build/v13-combination-recheck.json
.venv/Scripts/python.exe -m scripts.prepare_combination_holdout --output build/everyday-v13-combination-recheck
.venv/Scripts/python.exe -m scripts.audit_combinations --data build/everyday-v13-combination-recheck --output build/v13-holdout-recheck.json
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe scripts/check_default_model.py
```

形態素の活用・句読点・サ変複合、修正データの全代案、同一条件での採点、語彙外の明示、除去後の構成語、入力・出力hash、既存重みからの初期化拒否をテストした。全Pythonテストは **91 passed**、標準モデルの整合性・生成確認も成功した。今回Rust・WASM・Web・配布モデルは変更しておらず、ブラウザー再ビルドや新規モデルの速度測定は行っていない。

次は、自然な候補同士の微妙な相性と、格・文脈を変えれば自然になる例を加え、構成語と同義表現の漏洩を点検してから、共通の乱数初期値による通常CEと複数候補学習を比較する。この91対の高得点だけで採用ゲートや予約した確認seedへは進めない。
