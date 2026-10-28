# 自作PoCデータ

`sentences.jsonl`: 新規作成30文、明示segments/タイピング用かな/品詞付き。`review.status=agent_curated`、独立した人間の確認はまだない。CC0-1.0。自動解析の正解データを装っていない。

`lindera-src`: 98語の自作UTF-8 CSV、最小char/unknown/connection定義。全語のコスト/接続IDは0で言語統計を持たない。辞書化・解析経路の実験用。`analysis-cases.jsonl`: segmentsなし正常4行、読み不一致の異常1行。

生成元は `scripts/create_examples.py`, `scripts/create_dictionary_fixture.py`。ルートから明示実行できるが、人間が校正した後に再実行するとその編集を上書きする。学習には既定で採用されず、サンプル実験には `--allow-curated-examples` を指定する。
