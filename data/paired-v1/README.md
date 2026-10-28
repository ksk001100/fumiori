# v1の対応付き散文

`stories-v1-3mb` の学習入力となる自作散文と、対応する表記・かな・品詞を収録する。学習2,500文、validation300文、test300文、family-test300文。独立した人間の言語レビューは未実施。適用可能な権利の範囲でCC0-1.0として提供する。

`train.jsonl` などの本文と `vocab.json` はv1学習時の入力をそのまま保持する。`manifest.json` に各ファイルのSHA-256と件数を記録する。`scripts/prepare_stories.py` がこの入力に日本語SimpleStoriesの対応付け可能な短文を追加し、10万文の学習データを作成する。

文型・語句・出自などの行内metadataは学習データの由来として保持する。モデル推論時にはこの散文一覧を使わない。
