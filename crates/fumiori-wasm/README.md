# fumiori-wasm

`wasm-bindgen` wrapper for the shared Candle CPU generator. `Generator` accepts
manifest/config/vocabulary/weight byte arrays and checks the same bundle hashes
and limits as native inference. `generate(difficulty, seed)` returns prompt JSON;
`logits(ids)` supports numerical parity tests. Call `free()` before discarding it.

From the repository root, run `python scripts/build_wasm.py` after preparing
`build/model` and `build/model-int8`. Requires the installed
`wasm32-unknown-unknown` target and wasm-bindgen-cli **0.2.127**.
Serve `build/web` over localhost HTTP. See [measurements and limitations](../../docs/three-mb-generation.md).

For a fresh checkout without training exports, use
`python scripts/build_wasm.py --runtime-model models/stories-v2-diverse-3mb`.
This is also the GitHub Pages build; the bundled runtime files are checked against
the manifest hashes. See the [deployment instructions](../../README.md#github-pagesへの公開).

The demo performs single-thread inference inside a module Web Worker. INT8 storage
is dequantized to FP32 at load. Edge was tested; other browsers are not certified.
