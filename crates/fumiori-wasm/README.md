# fumiori-wasm

共通のCandle CPU推論を`wasm-bindgen`経由で使う。
`Generator`はmanifest・設定・語彙・重みのバイト列を受け取り、ネイティブ推論と同じハッシュと構造を検証する。
`generate(difficulty, seed)`はお題のJSON、`logits(ids)`は数値照合用。破棄時は`free()`を呼ぶ。

```powershell
cargo build --locked --release -p fumiori-wasm --target wasm32-unknown-unknown
wasm-bindgen target/wasm32-unknown-unknown/release/fumiori_wasm.wasm --target web --out-dir build/pkg
```

`wasm32-unknown-unknown`と`wasm-bindgen-cli 0.2.127`が必要。
モデルを選択してWeb Workerで生成するデモ・ビルド・スモークテスト・ブラウザーテスト・Pages公開は
[fumiori-models](https://github.com/ksk001100/fumiori-models#生成とwebデモ)へ移した。
INT8/INT4は保存形式で、推論はFP32演算を行う。
