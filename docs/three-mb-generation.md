# 3MB以内での未知の日本語短文生成

モデル名は `stories-v1-3mb`。文を全文候補から選ぶ方式ではなく、モデルが語彙IDを逐次予測する。学習文との一致を理由に出力を捨てる処理は使わない。

## 容量と方式

4層・幅224・8head・FFN672・最大64位置、2,624,832パラメータ、2,644語彙ID。INT8で保存し、CPU/WASMでは読み込み時にFP32へ戻して計算する。

容量制限は、重み・語彙・設定・manifest・LICENSEを含む**配布用モデル一式の展開後サイズ3,000,000 bytes以内**。ZIPだけを3MB以内にする条件ではない。FP32診断用export、学習データ、checkpoint、parity検査用データ、WASM/JSコード、実行時メモリーはこのモデル配布容量に含めない。

最終配布一式は2,903,290 bytes（約2.90MB）、ZIP2,498,243 bytes。語彙の学習用metadataは配布から除き、ID・表記・読み・品詞を保持する。元のmetadataと出典はexportとmanifestのhashから追跡できる。

推論には全文候補・trie・学習文一覧・外部LLMを渡さない。静的fallbackも禁止する。局所的な次のチェックは明示的に使う。

- かな長・文末・予約ID等の既存の構造チェック。
- 別の漢字名詞を連結して未知の複合語読みを作らない境界チェック。「風」＋「船」から誤って「かぜふね」を作らず、語彙にある「風船／ふうせん」を使う。
- 「遊びましたいる」のような丁寧形の直後の連結を抑制。接続助詞や句読点は残す。
- 名詞「思い出／おもいで」と、「思い出しました」の動詞語幹「おもいだ」を局所的に区別する読み補正。

これらは一般的な日本語解析器や意味の正しさの保証ではない。新しい名詞の組み合わせ・述語・文型の全文を列挙する制約でもない。任意の新しい漢字語・未知の読みを自由に創作できるモデルとは区別する。

## データと研究

[SimpleStories（NeurIPS 2025）](https://proceedings.neurips.cc/paper_files/paper/2025/hash/34696303f5996d5c54e699c0b462ae48-Abstract-Datasets_and_Benchmarks_Track.html) と[日本語版データカード](https://huggingface.co/datasets/SimpleStories/SimpleStories-JA)を参考に、日本語の文章構造を増やした。これは人間が作文したコーパスではなく、主にgpt-4o-miniで生成された合成物語で、データカードのライセンス表記はMIT。

取得したのは公開trainの最初の1shard、178,169,487 bytes。upstream testは取得・学習していない。revisionは `d25370316dd77ec0ee4fd9db57bfe57075cf4b90`、ファイルSHA256は `54bd792d9442c0bc1471d061bdc2228c889d2e3dd5c9a9fe258cd7791bd8ae1b`。

学習は `data/paired-v1` の独自の対応付き散文2,500件と、追加97,500件の合計100,000件。validation800、test800、family-test300。元の物語のSHA256でsplitを分け、正規化本文の重複と既存の保護された本文を除く。同一の元物語や正規化本文がsplitを跨がないことを確認した。

表記と読みの対応には既存の明示語彙、限定した活用形、追加の明示読みを使う。未知の漢字や、読みが変わりうる未知複合語は推測せず除く。かな文字はそのまま読める文字IDとして扱い、`pos:文字` として単語の品詞を推測しない。文脈依存の読みをすべて解決したものではない。

318,190物語を走査して必要な学習短文を得た。構造上の候補に合わないものや、強制された冒頭語による不自然な修飾なども除外した。全文を独立した人間が点検したデータではなく、児童向け物語の題材・人物・表現への偏りが残る。一般の日本語全域の自然さを主張しない。

学習はCUDA、AdamW、200stepのwarmupとcosine learning-rate decay、最大10,000step、validation lossでcheckpointを選択。今回の選択stepは10,000、学習約124秒。学習lossは固定した2,048件の標本で記録し、validationは全件で計算する。評価用testや生成後の自然さ採点で重みを選んではいない。

## 評価

生成設定はtemperature0.75 / top-p0.95、easy/normal/hard巡回、完全一致履歴16。開発seed42/2026/73、確認seed3432/811/1907、各seed200文。文字列の新規率と自然さを別々に測る。

開発600文では全文候補なしで600件生成、594種類（99%）、学習本文と完全一致221件（36.8%）、正規化本文が学習にない374件（62.3%）。正規化は時刻・一部の活用・副詞を除く既存のヒューリスティックであり、意味的独立性の保証ではない。人物名や形容を変えただけの新規性もすべて除けるわけではない。

自然さと読みは、全600要求から固定乱数9011で無作為抽出した100文を点検した。新規文だけ・見栄えのよい文だけを選んでいない。**Codexによる点検**で、独立した人間の評価ではない。開発標本では自然さと読みの両方に問題なしと判定したものが90/100、未知本文の部分は57/67だった。

採用条件は、配布3MB以内、600件すべてモデル出力、新規正規化本文50%以上、異なる本文90%以上。さらに、点検100文中90文以上に問題がなく、未知本文を少なくとも30文点検し、その85%以上に問題がないこと。設定を変えず、確認seedでも同じ条件を満たした場合のみ標準へ反映する。

[開発の全出力とhash](stories-v1-3mb-development-measurements.json)、[開発点検](stories-v1-3mb-development-review.json)。[確認600文](stories-v1-3mb-confirmation-measurements.json)では595種類、学習本文との完全一致241件（40.2%）、未知の正規化本文355件（59.2%）。[確認点検](stories-v1-3mb-confirmation-review.json)も90/100、未知部分は53/61で、設定と重みを変えず条件を通過した。[採用判定](stories-v1-3mb-promotion.json)。

「参照集合に存在すること」を自然さの評価に使わない。今回は参照の外に新しい文を作ることが目標なので、参照外という理由で棄却せず、出力と読みをそのまま点検する。不自然さ・読みの誤りは依然残り、100文の点検を全生成への保証とはしない。未知文型への転移も未達の領域で、family-testには外部文章との構造的な重なりがありうるため、厳密な未見文型評価とは扱わない。

残った例は「きっかけは、ある日、森の中で小さな花を見つけました」の構成の崩れ、「春の雨が降ります」を「あめがおります」と読む文脈上の誤りなど。既知の読みが異なる「降ります」のような語をすべて解決したわけではない。児童向け物語の擬人化・幻想的な場面・短い問いかけを許容して採点した。一般的な実用文の品質や自然さ100%とは区別する。

## 再現

```powershell
# 追加のデータ取込ライブラリ（推論には不要）
py -3.12 -m pip --python .venv/Scripts/python.exe install -r training/stories-requirements.txt
# 公開train shardを取得し、版・サイズ・SHA256・ライセンス表記を記録
.venv/Scripts/python.exe scripts/fetch_stories.py
# データ準備、GPU学習、export、量子化、parity、生成と点検結果の照合
.venv/Scripts/python.exe scripts/stories_pipeline.py
# 既存checkpointを再利用
.venv/Scripts/python.exe scripts/stories_pipeline.py --skip-training
# 開発と確認の条件を通過した場合だけ標準へ配置
.venv/Scripts/python.exe scripts/stories_pipeline.py --skip-training --activate
```

重み・生成方針・点検対象のhashが一致しない既存の採点は再利用しない。新しい実験の点検票がなければ自動的に合格扱いせず停止する。

標準 `build/model` / `build/model-int8` に反映済み。配布には `build/fumiori-stories-v1-3mb` の5ファイルを使用する。FP32診断用exportとparityデータを一緒に配布する必要はない。

Rustでは `TypingGenerator::load` 後に `default_options()` でモデル別の推奨設定を取得できる。明示した生成オプションを上書きしない。学習データ・Python・PyArrow・CUDAはアプリ実行時に不要。`scripts/build_wasm.py` の既定はINT8標準モデルで、Web用にもmetadataを削減した語彙を使用する。

## 実装の検証

Rust/Pythonのテスト、fmt、Clippyで実装を検証する。FP32/INT8各26,440 logitsのPyTorch/Candle parityは最大誤差8.11e-6以下。WASMも再ビルドし、Web用モジュールをNodeで読み込んで標準モデル30文の生成、本文・かなの連結一致、静的fallbackなしを確認した。ブラウザのDOM/Worker操作を再検証したものではない。

[CPU測定](stories-v1-3mb-bench.json)はnormal30文、失敗0、モデルロード約13.4ms、warm p50約50.9ms / p95約509ms。再試行を含むこのPC上の一回の測定で、機器や文の長さによって変わる。実行時メモリーの上限を3MBにしたという意味ではない。

10万件の参照に対する近似文字列一致は高コストなので、この実験の評価では `--skip-near-statistics` で省き、値を0ではなくnullとして記録する。完全一致・正規化本文一致・出力本文は維持し、近似一致の省略が生成結果を変えないこともテストする。
