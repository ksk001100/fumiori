# データの由来と配布範囲

v19はv17-expandedのtrainから行を選別したsubsetで、新しい外部データ・教師文・修正文は追加しない。元の各行の出典・著者・ライセンスを保持し、除外した行の出典対応も[v19データ記録](everyday-v19-data.json)へ残す。合成文6行の点検表は[data/curation-v19](../data/curation-v19/README.md)。Tatoeba、合成教師、自作文、辞書等の親manifestと配布LICENSEを継承し、整理後コーパス全体をCC0とする主張ではない。[整理と比較](compositional-generation-v19.md)を参照。

v18はv17-expandedと同じ準備データを使い、新しい教師出力・外部データ・修正文を追加しない。変更は学習中のcorpus mixtureのみで、出典・ライセンスをv17から継承する。[v18記録](compositional-generation-v18.md)を参照。

v17の2モデルはv13除去済みtrainに、既存ローカル `llm-jp/llm-jp-3.1-1.8b-instruct4` の新規出力500件／2,518件を追加する。教師revision・モデルファイル・few-shot train由来・原本のhashは[v17データ記録](everyday-v17-data.json)に保持し、配布manifestとLICENSEにも教師ソースを追記した。教師モデルのメタデータはApache-2.0だが、合成文や混合コーパス全体をApache-2.0またはCC0とする主張ではない。教師・プロンプト・生成文一覧はアプリへ配布しない。元のTatoeba・自作文・辞書等の出典を継承する。[実験記録](compositional-generation-v17.md)を参照。

v16の2モデルはv15と同じv13除去splitを使い、変更はtrain内の分岐頻度を数える基準とlossの重み付けのみ。新しい文・教師生成・修正文は追加せず、親manifestと配布LICENSEの出典を継承する。[v16記録](compositional-generation-v16.md)を参照。

v15の4モデルもv13の同じ除去splitを使う。変更は学習時の抽選比率と入力マスクで、外部データ・教師生成・修正文の追加はない。親manifestと配布LICENSEの出典を継承する。[v15記録](compositional-generation-v15.md)を参照。

v14の2モデルは既存v13の除去splitを用い、外部文や新しい教師出力は追加していない。Tatoeba、既存の自作文、辞書、合成教師に関する親manifestの出典と配布LICENSEを継承する。[実験記録](compositional-generation-v14.md)を参照。

## 現行モデルと従来モデルの学習データ

現行のstories-v2-diverse-3mbは、自作の対応付き散文2,500文と、日本語SimpleStoriesから生成条件を満たす28,300文を追加した合計30,800文で学習した。従来のstories-v1-3mbは同じ自作散文と日本語SimpleStoriesから追加した97,500文の合計10万文。

自作散文は [data/paired-v1](../data/paired-v1/README.md) に表記・かな・品詞・出自とともに収録する。適用可能な権利の範囲でCC0-1.0として提供する。独立した人間の言語レビューは未実施。

[日本語SimpleStories](https://huggingface.co/datasets/SimpleStories/SimpleStories-JA) はThe SimpleStories Projectの合成物語データで、主にgpt-4o-miniにより生成された。取得時のデータカードのライセンス表記はMIT。公開trainの1shardを使用し、upstream testは取得・学習していない。revision・原本SHA-256・データカードは `build/simple-stories/source.json` / `README.md` に保持する。

配布用モデルの [`models/stories-v2-diverse-3mb/manifest.json`](../models/stories-v2-diverse-3mb/manifest.json) に `data_sources` として出典を記録する。語彙の学習用metadataは配布用パッケージから除き、ID・表記・かな・品詞を保持する。取得・対応付け・学習splitの詳細は [モデルの説明](three-mb-generation.md)。

## 同梱のテストfixture

[data/examples](../data/examples/README.md) の30文、明示segments、98語の自作辞書ソース、解析fixtureはライブラリの検証用。エージェント作成・読み点検の由来を保持し、適用可能な権利の範囲でCC0-1.0として提供する。v1の学習用散文と区別する。独立した人間の言語レビューは未実施。

## ソフトウェアと推論時の入力

[data/combinations-v13](../data/combinations-v13/README.md) の100対はCodexが作成・点検した評価専用の対照例。適用可能な権利の範囲でCC0-1.0として提供する。独立した人間の点検は未実施。既存の学習文と一致する組合せはあるが、外部文例集の転載ではなく、推論への配布や学習への追加は行わない。再学習用に元コーパスを除去したデータは親manifestの出典・ライセンスを継承し、データ全体をCC0としない。

ソフトウェアはMIT。推論はCandle、データ加工はLinderaを使用する。推論時はモデルに同梱した語彙と重みを読み、外部辞書・外部LLM・学習文一覧は使わない。
