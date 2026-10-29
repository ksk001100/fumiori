# v12：複数修正による誤り学習と4bitでの容量配分

2026-10-29。[v11](compact-generation-research.md) に続き、配布3,000,000 bytes・通常生成100ms以内で自然さと未知文を増やす実験を行った。今回も採用条件を満たす候補は得られず、標準v2・デモv3を継続する。

## 結果

| 候補 | 配布バイト | 600seed中の重複 | 新規正規化本文 | 自然さ・読み |
|---|---:|---:|---:|---:|
| 比較用v11追加学習モデル | 2,528,978 | 47 | 56.5% | 92/100 |
| 複数の修正文による追加学習（SFT） | 2,528,988 | 44 | 56.0% | 89/100 |
| 上記＋誤り箇所のunlikelihood（UL） | 2,528,988 | 43 | 57.0% | 93/100 |
| 本体を増やした4bit候補 | 2,801,219 | 30 | 57.8% | 81/100 |

全候補を新しい開発seed **820000〜820599** で再測定。easy/normal/hard各200件、temperature 0.8・top-p 0.95、履歴を無効化・消去して各seedを独立に生成した。前回の点検84/100等と今回の92/100を改善幅として比べない。

新規率は、v11タスク学習文・過去のv3学習文・今回の**trainに使った修正文**の和集合に、正規化本文がない割合。修正文を暗記した出力も未知文として数えない。SFTの1件はこの補正で新規から既知へ変更された。事前学習コーパスや世界中の文との非一致を証明するものではない。

出力を見る前に乱数20261041で600件から100件を選び、本文とかなをCodexが点検した。独立した人による盲検評価ではなく、文脈や語の相性には主観が残る。点検100件の童話調は各候補0件。自然や風景の話題、自然な詩的描写は排除していない。

ULは同じ100seedでSFTより良かった例4・悪かった例0、比較用v11より良かった例4・悪かった例3。探索的な対応あり二項検定ではそれぞれ両側p=0.125、p=1.0。複数実験の補正や独立した人の評価もなく、**93/100だけで改善を断定しない**。新規率95%・自然さ95/100に届かないため、予約した確認用830000台の評価には進めていない。

## 誤った箇所だけを下げ、代案を一つに絞らない

学習用seed 810000〜810255・811000〜811383で学生モデルの出力を収集。評価用seedを誤り収集に流用しない。Codexが誤り箇所のprefixと2〜3種類の自然な続きを注釈した。例えば「鳥の絵を木炭で買う」に対して「描く」「描いた」「描いている」。元から不自然なprefixが残る例や、判断が曖昧な例は点検で外した。

元の語彙8,802 IDは変えず、Sudachiによる分割・表記・かな・品詞が既存語彙と一致するものだけを使った。単語境界にないprefix、未知語、prefixが再分割された修正、棄却対象の単語をそのまま含む修正等は除外。prefixのhashでtrain **34種類・92修正文**、validation **9種類・23修正文**に分けた。全修正文を作成後に点検してから学習している。

補助lossは修正後のsuffixだけのcross entropy。prefixを一様に選び、その中の修正文を一様に選ぶので、一つの修正を勝者に固定しない。ULでは同じprefixに対する誤った次トークンの確率 `p_bad` に `−log(1−p_bad)` を加える。数値計算時は確率を最大 `1−10⁻⁶` に制限する。全単語の禁止リストやハードな文型制約ではなく、学習での弱い重み付けである。

v11追加学習checkpointから同じseed・600step・バッチ抽選・学習率で比較した。元コーパスのCEに修正suffix CEを0.3倍加え、UL候補だけULを0.2倍加える。修正バッチ32、元バッチ128、学習率5e-5から減衰。validationで都合のよいstepを選び直さず、両候補とも固定600stepを保存した。通常のvalidation lossは元モデルより少し悪化している。

## 修正の暗記と一般化を分けて調べた

保存したFP32 checkpointで、各prefixに対する「誤った続きの確率」を測った。最終配布の量子化後の生成品質とは別の診断である。

| 対象 | 比較用v11 | 修正SFT | 修正SFT＋UL |
|---|---:|---:|---:|
| 学習に使った34種類の平均誤り確率 | 4.54% | 0.54% | 0.27% |
| 別に残した9種類の平均誤り確率 | 2.01% | 3.44% | 3.42% |
| 学習34種類の自然な最初の単語への総確率 | 10.4% | 72.1% | 72.4% |
| 別の9種類の自然な最初の単語への総確率 | 14.1% | 15.0% | 15.0% |

学習した箇所は修正できたが、別の誤りへの転移は確認できず、むしろ誤り確率は上がった。9種類だけなので推定も粗い。今回の注釈は少数で、**語の相性の一般化を解決した成果とはしない**。

修正例を推論に配る処理は追加していない。92修正文・棄却表・教師・追加の分類器は実行時には不要で、2.53MBのモデルサイズと既存の生成APIを維持した。

## ローカル教師の自動判定を採用しなかった理由

既存のLLM-jp-3.1-1.8b-instruct4を、固定revision・ファイルhash照合・local_files_onlyで利用した。自然/不自然をA/Bで答える判定を予備試験し、ラベルの対応を逆にした場合も調べた。しかし「パソコンがかかってきた」等を自然側へ採点する例やラベル順の影響があった。

この分類器の点数を学生の教師ラベルには使わず、今回はCodexが作成・点検した小規模な修正データで補助lossの予備実験を行った。完全自動の高品質な修復蒸留が完成したわけではない。教師の流暢さを、校正ラベルの正確さと同一視しない。

## 4bitの余裕を本体へ配分

幅128・4層・FF384、約1.80MパラメータのINT8に対し、幅192・6層・FF576、**3,928,320パラメータ**を基本的なblock-128 INT4 QATで学習した。語彙とコーパスのsplitはv11と同じ。混合比はTatoeba 50%・教師25%・日常文20%・対応付き散文5%。6,000step中validation最良の5,500stepを保存した。

配布2,801,219 bytesに収まり、INT4もロード時にFP32へ戻すので演算自体が4bitになるわけではない。本体・混合比・量子化が同時に異なる探索候補であり、自然さの差を量子化だけの原因と断定しない。今回の自然さ81/100、新規率57.8%では不採用。

表面上の代名詞始まり（彼は・彼女は・私は等）は600件中、比較用15、SFT 15、UL 18、4bit候補159。これはジャンルの判定ではないが、童話調を除く以外にも偏りを測る必要がある。単に本体を大きくすれば文体の幅が広がる結果にはならなかった。

## 実ブラウザーの速度と検証

ライブデモとは別の `build/v12-preview` で、4bit候補を実Edge Web Workerで測定。normal 30件、seed42、履歴16件、temperature 0.8・top-p 0.9。中央値 **28.5ms**、p95 **42ms**、最大69.4ms。ロード・画面描画を除く。品質評価はtop-p 0.95なので設定が異なる。Windows x86_64 / Intel Family 6 Model 183 / Edge 156.0.4314.50での少数測定であり、全端末での保証ではない。

既存モデル→4bit候補→既存モデルの切替と3難易度の生成、キャッシュされた古いWorkerの検出を実ブラウザーで確認した。新しい3候補はいずれもPython/Candleの88,020 logitを絶対許容差0.0002以内で照合した。正しい数値計算と自然な文章の合格は別である。

## 参照した研究と、今回実装した範囲

- [CRINGE, ACL 2023](https://aclanthology.org/2023.acl-long.493/)：負例を学習に使う手法。今回の単純なsuffix CE＋単一トークンULはCRINGE全体の再現ではない。
- [Multi-Response Training, 2026-05 preprint](https://arxiv.org/abs/2606.00544)：同じ入力に複数の応答を残す考え方と、選択による多様性の減少を扱う。今回の一様な修正抽選の参考にしたが、日本語校正で同じ効果が出るとは仮定していない。
- [Concrete Score Distillation, ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/1bfc9f74afa91b9b8add5a97a97001a1-Abstract-Conference.html)：確率の模倣だけでなく、教師と学生の相対logit差を合わせる研究。今回は未実装。少数の誤りラベルだけでなく、語彙全体の関係を伝える案の候補として調べた。

次に検証する価値があるのは、名詞と述語をそれぞれ知っていても組合せは未見、という分割で語の相性の転移を直接測ること。また、単一の正誤ラベルに頼らず、教師の複数の自然な候補の関係を小型モデルへ伝える方法を比較すること。今回の小規模な修正学習の成功と混同せず、学習量・初期値・データ比を揃えた対照を用意する必要がある。

## 記録と再現

[比較JSON](everyday-v12-comparison.json) に設定、モデル・レポートhash、全測定、修正の確率診断、教師判定の予備試験を保存。各 `everyday-v12-*-independent-seeds.json` に全600出力、対応する `*-review.json` に100件すべての点検を保存した。注釈と実際に使ったprepared snapshotは [修正データ](../data/repairs-v12/README.md)。

```powershell
# v11の既存runtimeから、評価とは別の学習用出力を収集
target/release/fumiori-cli.exe evaluate-seeds --model build/everyday-v11-quality-runtime --samples 256 --seed 810000 --temperature .8 --top-p .95 --report build/v12-training-rollouts.json
target/release/fumiori-cli.exe evaluate-seeds --model build/everyday-v11-ce-runtime --samples 384 --seed 811000 --temperature .8 --top-p .95 --report build/v12-ce-training-rollouts.json
python -m scripts.prepare_repairs --output build/everyday-v12-repairs-reviewed
python -m scripts.initialize_compact --checkpoint build/checkpoints/everyday-v11-quality.pt --data-dir build/everyday-v11-data --output build/everyday-v11-data/quality-initialization.pt
python -m training.train --config configs/everyday-v12-repair-sft-3mb.json
python -m training.train --config configs/everyday-v12-repair-ul-3mb.json
python -m training.train --config configs/everyday-v12-qat4-3mb.json
```

exportのautoregressive・境界・読みガードはv11と同じ。修正候補はINT8、QAT候補は `training.quantize --bits 4`、続いて `training.package --strip-token-metadata`。既存のファイルは上書きせず、新しい出力先で再実行する。CUDAと親のv11準備データが必要。
