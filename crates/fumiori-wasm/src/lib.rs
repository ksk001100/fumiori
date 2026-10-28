use fumiori_core::{Difficulty, GenerateOptions};
use fumiori_inference::TypingGenerator;
use wasm_bindgen::prelude::*;

fn js_error(e: impl std::fmt::Display) -> JsValue {
    JsValue::from_str(&e.to_string())
}

#[wasm_bindgen]
pub struct Generator {
    inner: TypingGenerator,
}
#[wasm_bindgen]
impl Generator {
    #[wasm_bindgen(constructor)]
    pub fn new(
        manifest: &[u8],
        config: &[u8],
        vocab: &[u8],
        weights: &[u8],
    ) -> Result<Generator, JsValue> {
        Ok(Self {
            inner: TypingGenerator::from_bytes(manifest, config, vocab, weights)
                .map_err(js_error)?,
        })
    }
    pub fn generate(&mut self, difficulty: &str, seed: u32) -> Result<String, JsValue> {
        let difficulty = match difficulty {
            "easy" => Difficulty::Easy,
            "normal" => Difficulty::Normal,
            "hard" => Difficulty::Hard,
            _ => return Err(js_error("unknown difficulty")),
        };
        let prompt = self
            .inner
            .generate(&GenerateOptions {
                difficulty,
                seed: Some(seed as u64),
                ..self.inner.default_options()
            })
            .map_err(js_error)?;
        serde_json::to_string(&prompt).map_err(js_error)
    }
    pub fn logits(&self, ids: Vec<u32>) -> Result<String, JsValue> {
        let ids: Vec<_> = ids.into_iter().map(|i| i as usize).collect();
        serde_json::to_string(&self.inner.logits(&ids).map_err(js_error)?).map_err(js_error)
    }
}
