use anyhow::{ensure, Context, Result};
use candle_core::{DType, Device, Module, Tensor};
use candle_nn::{LayerNorm, Linear, VarBuilder};
use fumiori_core::*;
use rand::{rngs::StdRng, Rng, SeedableRng};
use serde::Deserialize;
use serde_json::Value;
use std::{
    collections::{BTreeMap, VecDeque},
    fs,
    path::Path,
};

#[derive(Debug, Clone, Deserialize)]
pub struct ModelConfig {
    pub vocab_size: usize,
    pub d_model: usize,
    pub n_heads: usize,
    pub n_layers: usize,
    pub d_ff: usize,
    pub max_seq_len: usize,
    pub version: String,
}
impl ModelConfig {
    fn validate(&self) -> Result<()> {
        ensure!(
            self.version == VERSION && (8..=16000).contains(&self.vocab_size),
            "unsupported model version/vocabulary"
        );
        ensure!(
            self.d_model > 0
                && self.d_model <= 512
                && self.n_heads > 0
                && self.d_model.is_multiple_of(self.n_heads)
                && (1..=16).contains(&self.n_layers)
                && (1..=2048).contains(&self.d_ff)
                && (3..=64).contains(&self.max_seq_len),
            "invalid model dimensions"
        );
        Ok(())
    }
}
struct Block {
    ln1: LayerNorm,
    qkv: Linear,
    attn_out: Linear,
    ln2: LayerNorm,
    fc1: Linear,
    fc2: Linear,
}
impl Block {
    fn load(c: &ModelConfig, vb: VarBuilder<'_>) -> Result<Self> {
        Ok(Self {
            ln1: candle_nn::layer_norm(c.d_model, 1e-5, vb.pp("ln1"))?,
            qkv: candle_nn::linear(c.d_model, 3 * c.d_model, vb.pp("qkv"))?,
            attn_out: candle_nn::linear(c.d_model, c.d_model, vb.pp("attn_out"))?,
            ln2: candle_nn::layer_norm(c.d_model, 1e-5, vb.pp("ln2"))?,
            fc1: candle_nn::linear(c.d_model, c.d_ff, vb.pp("fc1"))?,
            fc2: candle_nn::linear(c.d_ff, c.d_model, vb.pp("fc2"))?,
        })
    }
    fn forward(&self, x: &Tensor, c: &ModelConfig) -> Result<Tensor> {
        let (b, t, d) = x.dims3()?;
        let dh = d / c.n_heads;
        let qkv = self.qkv.forward(&self.ln1.forward(x)?)?;
        let part = |i| -> candle_core::Result<Tensor> {
            qkv.narrow(2, i * d, d)?
                .reshape((b, t, c.n_heads, dh))?
                .transpose(1, 2)?
                .contiguous()
        };
        let (q, k, v) = (part(0)?, part(1)?, part(2)?);
        let scores = (q.matmul(&k.transpose(2, 3)?.contiguous()?)? / (dh as f64).sqrt())?;
        let mask: Vec<f32> = (0..t)
            .flat_map(|i| (0..t).map(move |j| if j > i { f32::NEG_INFINITY } else { 0.0 }))
            .collect();
        let mask = Tensor::from_vec(mask, (1, 1, t, t), x.device())?;
        let probs = candle_nn::ops::softmax_last_dim(&scores.broadcast_add(&mask)?)?;
        let z = probs
            .matmul(&v)?
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b, t, d))?;
        let x = (x + self.attn_out.forward(&z)?)?;
        let y = self
            .fc2
            .forward(&self.fc1.forward(&self.ln2.forward(&x)?)?.gelu_erf()?)?;
        Ok((&x + y)?)
    }
}
pub struct Model {
    pub config: ModelConfig,
    token_emb: Tensor,
    pos_emb: Tensor,
    blocks: Vec<Block>,
    final_ln: LayerNorm,
}
impl Model {
    fn load(config_bytes: &[u8], weights: &[u8], vocab: &Vocab, quantized: bool) -> Result<Self> {
        let config: ModelConfig = serde_json::from_slice(config_bytes)?;
        config.validate()?;
        ensure!(
            config.vocab_size == vocab.tokens.len(),
            "model vocabulary size mismatch"
        );
        let mut tensors = candle_core::safetensors::load_buffer(weights, &Device::Cpu)?;
        if quantized {
            let names: Vec<_> = tensors
                .keys()
                .filter(|n| !n.ends_with(".scale"))
                .cloned()
                .collect();
            for name in names {
                if tensors[&name].rank() == 2 {
                    let q = tensors.remove(&name).context("missing quantized matrix")?;
                    ensure!(q.dtype() == DType::U8, "expected INT8 matrix: {name}");
                    let scale = tensors
                        .remove(&format!("{name}.scale"))
                        .context("missing row scale")?;
                    ensure!(
                        scale.dtype() == DType::F32 && scale.dims() == [q.dim(0)?, 1],
                        "invalid row scale shape"
                    );
                    ensure!(
                        scale
                            .flatten_all()?
                            .to_vec1::<f32>()?
                            .iter()
                            .all(|s| s.is_finite() && *s > 0.0),
                        "invalid row scale"
                    );
                    tensors.insert(
                        name,
                        (q.to_dtype(DType::F32)? - 128.0)?.broadcast_mul(&scale)?,
                    );
                }
            }
        }
        ensure!(
            tensors.len() == 4 + 12 * config.n_layers,
            "unexpected/missing weight tensors"
        );
        ensure!(
            tensors.values().all(|t| t.dtype() == DType::F32),
            "only FP32 weights supported"
        );
        for (name, t) in &tensors {
            ensure!(
                t.flatten_all()?
                    .to_vec1::<f32>()?
                    .iter()
                    .all(|x| x.is_finite()),
                "nonfinite weights in {name}"
            );
        }
        let vb = VarBuilder::from_tensors(tensors, DType::F32, &Device::Cpu);
        let token_emb = vb.get((config.vocab_size, config.d_model), "token_emb.weight")?;
        let pos_emb = vb.get((config.max_seq_len, config.d_model), "pos_emb.weight")?;
        let blocks = (0..config.n_layers)
            .map(|i| Block::load(&config, vb.pp(format!("blocks.{i}"))))
            .collect::<Result<_>>()?;
        let final_ln = candle_nn::layer_norm(config.d_model, 1e-5, vb.pp("final_ln"))?;
        Ok(Self {
            config,
            token_emb,
            pos_emb,
            blocks,
            final_ln,
        })
    }
    pub fn logits(&self, ids: &[usize]) -> Result<Vec<Vec<f32>>> {
        ensure!(
            !ids.is_empty() && ids.len() <= self.config.max_seq_len,
            "invalid sequence length"
        );
        ensure!(
            ids.iter().all(|i| *i < self.config.vocab_size),
            "unknown input ID"
        );
        let device = &Device::Cpu;
        let input = Tensor::new(ids.iter().map(|&x| x as u32).collect::<Vec<_>>(), device)?;
        let pos = Tensor::new((0..ids.len() as u32).collect::<Vec<_>>(), device)?;
        let mut x = (&self.token_emb.index_select(&input, 0)?
            + &self.pos_emb.index_select(&pos, 0)?)?
            .unsqueeze(0)?;
        for block in &self.blocks {
            x = block.forward(&x, &self.config)?;
        }
        let logits = self
            .final_ln
            .forward(&x)?
            .squeeze(0)?
            .matmul(&self.token_emb.t()?)?;
        Ok(logits.to_vec2::<f32>()?)
    }
}
#[derive(Debug, Default, Clone, serde::Serialize)]
pub struct GenerationStats {
    pub attempts: usize,
    pub rejected: usize,
    pub model_tokens: usize,
    pub fallback_reason: Option<String>,
}
#[derive(Default)]
struct FrameNode {
    children: BTreeMap<usize, usize>,
    completions: usize,
    relation: Option<usize>,
    kind: Option<usize>,
    prior_mass: f32,
}
struct FrameConstraints {
    nodes: Vec<FrameNode>,
    completion_prior: f32,
    relation_history_size: usize,
    kind_weights: Vec<f32>,
    probability_mix: f32,
    top_h: Option<f64>,
}
impl FrameConstraints {
    fn load(value: &Value, vocab: &Vocab, max_len: usize) -> Result<Self> {
        ensure!(
            value["version"] == "semantic_frames_v1",
            "unsupported frame constraints"
        );
        let sequences: Vec<Vec<usize>> = serde_json::from_value(value["sequences"].clone())?;
        ensure!(
            !sequences.is_empty() && sequences.len() <= 50000,
            "invalid frame count"
        );
        let relation_ids: Option<Vec<usize>> = value
            .get("relation_ids")
            .map(|ids| serde_json::from_value(ids.clone()))
            .transpose()?;
        if let Some(ids) = &relation_ids {
            ensure!(
                ids.len() == sequences.len() && ids.iter().all(|id| *id < 50000),
                "invalid relation IDs"
            );
        }
        let relation_history_size = value
            .get("relation_history_size")
            .map(|n| n.as_u64().context("invalid relation history"))
            .transpose()?
            .unwrap_or(0) as usize;
        ensure!(relation_history_size <= 64, "relation history too large");
        ensure!(
            relation_history_size == 0 || relation_ids.is_some(),
            "relation history lacks IDs"
        );
        let kind_weights: Vec<f32> = value
            .get("kind_weights")
            .map(|weights| serde_json::from_value(weights.clone()))
            .transpose()?
            .unwrap_or_default();
        let kind_ids: Option<Vec<usize>> = value
            .get("kind_ids")
            .map(|ids| serde_json::from_value(ids.clone()))
            .transpose()?;
        ensure!(
            kind_weights.len() <= 16 && kind_weights.iter().all(|w| w.is_finite() && *w > 0.0),
            "invalid kind weights"
        );
        if !kind_weights.is_empty() {
            let ids = kind_ids.as_ref().context("kind weights lack IDs")?;
            ensure!(
                ids.len() == sequences.len() && ids.iter().all(|id| *id < kind_weights.len()),
                "invalid kind IDs"
            );
        }
        let probability_mix = value
            .get("probability_mix")
            .map(|n| n.as_f64().context("invalid probability mix"))
            .transpose()?
            .unwrap_or(0.0) as f32;
        ensure!(
            probability_mix.is_finite() && (0.0..=0.5).contains(&probability_mix),
            "invalid probability mix"
        );
        let mut tree = Self {
            top_h: value
                .get("top_h")
                .map(|n| n.as_f64().context("invalid top_h"))
                .transpose()?,
            probability_mix,
            relation_history_size,
            kind_weights,
            nodes: (0..3).map(|_| FrameNode::default()).collect(),
            completion_prior: match value.get("completion_prior") {
                Some(prior) => prior.as_f64().context("invalid completion prior")? as f32,
                None => 0.0,
            },
        };
        ensure!(
            tree.top_h
                .is_none_or(|h| h.is_finite() && h > 0.0 && h <= 1.0),
            "invalid top_h"
        );
        let balance_relations = value
            .get("balance_relations")
            .map(|v| v.as_bool().context("invalid relation balance"))
            .transpose()?
            .unwrap_or(false);
        ensure!(
            !balance_relations || relation_ids.is_some(),
            "relation balance lacks IDs"
        );
        let balance_power = value
            .get("relation_balance_power")
            .map(|n| n.as_f64().context("invalid relation balance power"))
            .transpose()?
            .unwrap_or(1.0) as f32;
        ensure!(
            balance_power.is_finite() && (0.0..=1.0).contains(&balance_power),
            "invalid relation balance power"
        );
        let mut relation_sizes = BTreeMap::new();
        if balance_relations {
            for (seq, relation) in sequences.iter().zip(relation_ids.as_ref().unwrap()) {
                let difficulty = *seq.first().context("empty frame sequence")?;
                *relation_sizes
                    .entry((difficulty, *relation))
                    .or_insert(0usize) += 1;
            }
        }
        ensure!(
            tree.completion_prior.is_finite() && (0.0..=2.0).contains(&tree.completion_prior),
            "invalid completion prior"
        );
        for (sequence_index, sequence) in sequences.into_iter().enumerate() {
            ensure!(
                (3..=max_len - 1).contains(&sequence.len()),
                "invalid frame length"
            );
            ensure!(
                (3..=5).contains(&sequence[0]) && sequence.last() == Some(&1),
                "invalid frame control IDs"
            );
            let segments = vocab.decode(&sequence[1..sequence.len() - 1])?;
            ensure!(
                vocab.tokens[sequence[sequence.len() - 2]].surface == "。",
                "frame lacks internal sentence mark"
            );
            let difficulty =
                [Difficulty::Easy, Difficulty::Normal, Difficulty::Hard][sequence[0] - 3];
            TypingPrompt::from_segments(segments, PromptSource::ModelConstrained)
                .validate(&GenerateOptions {
                    difficulty,
                    ..Default::default()
                })
                .context("invalid authored frame")?;
            let mut node = sequence[0] - 3;
            for &id in &sequence[1..] {
                let next = if let Some(&next) = tree.nodes[node].children.get(&id) {
                    next
                } else {
                    let next = tree.nodes.len();
                    ensure!(next < 200000, "too many frame nodes");
                    tree.nodes.push(FrameNode::default());
                    tree.nodes[node].children.insert(id, next);
                    next
                };
                node = next;
            }
            tree.nodes[node].relation = relation_ids.as_ref().map(|ids| ids[sequence_index]);
            tree.nodes[node].prior_mass = if balance_relations {
                (relation_sizes[&(sequence[0], relation_ids.as_ref().unwrap()[sequence_index])]
                    as f32)
                    .powf(-balance_power)
            } else {
                1.0
            };
            tree.nodes[node].kind = kind_ids
                .as_ref()
                .and_then(|ids| ids.get(sequence_index))
                .copied();
        }
        for node in (0..tree.nodes.len()).rev() {
            tree.nodes[node].completions = if tree.nodes[node].children.is_empty() {
                usize::from(node >= 3)
            } else {
                tree.nodes[node]
                    .children
                    .values()
                    .map(|&next| tree.nodes[next].completions)
                    .sum()
            };
        }
        Ok(tree)
    }
    // With power=1, equal total prior per relation; smaller powers soften the correction.
    // Keep the availability mask produced by history/genre selection.
    fn completion_masses(&self, available: Option<&[usize]>) -> Vec<f32> {
        let mut masses = vec![0.0; self.nodes.len()];
        for node in (0..self.nodes.len()).rev() {
            masses[node] = if available.is_some_and(|counts| counts[node] == 0) {
                0.0
            } else if self.nodes[node].children.is_empty() {
                self.nodes[node].prior_mass
            } else {
                self.nodes[node]
                    .children
                    .values()
                    .map(|&next| masses[next])
                    .sum()
            };
        }
        masses
    }
    fn available_counts(&self, recent: &VecDeque<usize>, kind: Option<usize>) -> Vec<usize> {
        let mut counts = vec![0; self.nodes.len()];
        for node in (0..self.nodes.len()).rev() {
            counts[node] = if self.nodes[node].children.is_empty() {
                usize::from(
                    node >= 3
                        && (kind.is_none() || self.nodes[node].kind == kind)
                        && !self.nodes[node]
                            .relation
                            .is_some_and(|id| recent.contains(&id)),
                )
            } else {
                self.nodes[node]
                    .children
                    .values()
                    .map(|&next| counts[next])
                    .sum()
            };
        }
        counts
    }
}
pub struct TypingGenerator {
    pub vocab: Option<Vocab>,
    model: Option<Model>,
    constraints: Option<FrameConstraints>,
    constraints_enabled: bool,
    static_fallback_enabled: bool,
    lexical_boundary_guard: bool,
    polite_boundary_guard: bool,
    normalize_morph_readings: bool,
    default_options: GenerateOptions,
    fallback: Vec<TypingPrompt>,
    history: VecDeque<String>,
    recent_relations: VecDeque<usize>,
    history_size: usize,
    rng: StdRng,
    seed: Option<u64>,
    pub last_stats: GenerationStats,
}
impl TypingGenerator {
    pub fn fallback_only() -> Result<Self> {
        let fallback = include_str!("../../../data/examples/sentences.jsonl")
            .lines()
            .map(|line| -> Result<TypingPrompt> {
                let r: Value = serde_json::from_str(line)?;
                let segments = serde_json::from_value(r["segments"].clone())?;
                let p = TypingPrompt::from_segments(segments, PromptSource::Fallback);
                validate_pair(&p.text, &p.kana, &p.segments)?;
                Ok(p)
            })
            .collect::<Result<_>>()?;
        Ok(Self {
            vocab: None,
            model: None,
            constraints: None,
            constraints_enabled: true,
            static_fallback_enabled: true,
            lexical_boundary_guard: false,
            polite_boundary_guard: false,
            normalize_morph_readings: false,
            default_options: GenerateOptions::default(),
            fallback,
            history: VecDeque::new(),
            recent_relations: VecDeque::new(),
            history_size: 16,
            rng: StdRng::from_entropy(),
            seed: None,
            last_stats: GenerationStats::default(),
        })
    }
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let dir = path.as_ref();
        for (file, max) in [
            ("manifest.json", 1_048_576),
            ("model-config.json", 65536),
            ("vocab.json", 16_000_000),
            ("model.safetensors", 128_000_000),
        ] {
            ensure!(
                fs::metadata(dir.join(file))
                    .with_context(|| format!("missing model file: {file}"))?
                    .len()
                    <= max,
                "oversized {file}"
            );
        }
        Self::from_bytes(
            &fs::read(dir.join("manifest.json"))
                .context("missing model manifest; export first or use --fallback-only")?,
            &fs::read(dir.join("model-config.json"))?,
            &fs::read(dir.join("vocab.json"))?,
            &fs::read(dir.join("model.safetensors"))?,
        )
    }
    /// Load a checked bundle from memory, including browser-fetched bytes.
    pub fn from_bytes(
        manifest: &[u8],
        config: &[u8],
        vocab_bytes: &[u8],
        weights: &[u8],
    ) -> Result<Self> {
        ensure!(manifest.len() <= 1_048_576, "oversized manifest");
        let manifest: Value = serde_json::from_slice(manifest)?;
        ensure!(
            manifest["version"] == VERSION,
            "unsupported manifest version"
        );
        let quantized = manifest["dtype"] == "I8_ROW_F32";
        ensure!(
            quantized || manifest["dtype"] == "F32",
            "unsupported manifest dtype"
        );
        for (file, key, bytes, max) in [
            ("vocab.json", "vocab_sha256", vocab_bytes, 16_000_000),
            ("model.safetensors", "weights_sha256", weights, 128_000_000),
            ("model-config.json", "config_sha256", config, 65536),
        ] {
            ensure!(bytes.len() <= max, "oversized {file}");
            ensure!(
                manifest[key].as_str() == Some(&sha256(bytes)),
                "checksum mismatch: {file}"
            );
        }
        let vocab: Vocab = serde_json::from_slice(vocab_bytes)?;
        vocab.validate()?;
        let model = Model::load(config, weights, &vocab, quantized)?;
        let constraints = manifest
            .get("generation_constraints")
            .map(|value| FrameConstraints::load(value, &vocab, model.config.max_seq_len))
            .transpose()?;
        let mut g = Self::fallback_only()?;
        g.vocab = Some(vocab);
        g.model = Some(model);
        g.constraints = constraints;
        g.lexical_boundary_guard = manifest
            .get("lexical_boundary_guard")
            .map(|v| v.as_bool().context("invalid lexical boundary guard"))
            .transpose()?
            .unwrap_or(false);
        g.polite_boundary_guard = manifest
            .get("polite_boundary_guard")
            .map(|v| v.as_bool().context("invalid polite boundary guard"))
            .transpose()?
            .unwrap_or(false);
        if let Some(version) = manifest.get("reading_normalization") {
            ensure!(
                version == "japanese_morph_v1",
                "unknown reading normalization"
            );
            g.normalize_morph_readings = true;
        }
        if let Some(defaults) = manifest.get("generation_defaults") {
            g.default_options.temperature = defaults["temperature"]
                .as_f64()
                .context("invalid default temperature")?
                as f32;
            g.default_options.top_p = defaults["top_p"]
                .as_f64()
                .context("invalid default top_p")? as f32;
            g.default_options.bounds()?;
        }
        g.static_fallback_enabled = manifest
            .get("allow_static_fallback")
            .map(|value| value.as_bool().context("invalid static fallback policy"))
            .transpose()?
            .unwrap_or(true);
        if manifest["generation_mode"] == "autoregressive" {
            ensure!(
                g.constraints.is_none(),
                "autoregressive model must not contain sentence constraints"
            );
            ensure!(
                !g.static_fallback_enabled,
                "autoregressive model must not use static fallback"
            );
        }
        Ok(g)
    }
    /// Model-recommended sampling; explicit options passed to generate still take precedence.
    pub fn default_options(&self) -> GenerateOptions {
        self.default_options.clone()
    }
    pub fn logits(&self, ids: &[usize]) -> Result<Vec<Vec<f32>>> {
        self.model.as_ref().context("no model loaded")?.logits(ids)
    }
    pub fn set_history_size(&mut self, n: usize) {
        self.history_size = n;
        while self.recent_relations.len() > n {
            self.recent_relations.pop_front();
        }
        while self.history.len() > n {
            self.history.pop_front();
        }
    }
    pub fn clear_history(&mut self) {
        self.history.clear();
        self.recent_relations.clear();
    }
    pub fn set_constraints_enabled(&mut self, enabled: bool) {
        self.constraints_enabled = enabled;
    }
    fn remember(&mut self, text: &str) {
        if self.history_size > 0 {
            self.history.push_back(text.into());
            while self.history.len() > self.history_size {
                self.history.pop_front();
            }
        }
    }
    pub fn generate(&mut self, o: &GenerateOptions) -> Result<TypingPrompt> {
        let (min, max) = o.bounds()?;
        self.last_stats = GenerationStats::default();
        if self.seed != o.seed {
            self.rng = match o.seed {
                Some(s) => StdRng::seed_from_u64(s),
                None => StdRng::from_entropy(),
            };
            self.seed = o.seed;
        }
        if let (Some(model), Some(vocab)) = (&self.model, &self.vocab) {
            let constraints = self
                .constraints
                .as_ref()
                .filter(|_| self.constraints_enabled);
            let mut available = constraints
                .filter(|tree| tree.relation_history_size > 0 && self.history_size > 0)
                .map(|tree| tree.available_counts(&self.recent_relations, None));
            if let Some(tree) = constraints.filter(|tree| !tree.kind_weights.is_empty()) {
                let empty = VecDeque::new();
                let recent = if self.history_size > 0 {
                    &self.recent_relations
                } else {
                    &empty
                };
                let choices: Vec<_> = tree
                    .kind_weights
                    .iter()
                    .enumerate()
                    .filter_map(|(kind, weight)| {
                        let counts = tree.available_counts(recent, Some(kind));
                        (counts[o.difficulty.id() - 3] > 0).then_some((counts, *weight))
                    })
                    .collect();
                let total: f32 = choices.iter().map(|(_, weight)| weight).sum();
                if total > 0.0 {
                    let mut draw = self.rng.gen::<f32>() * total;
                    for (counts, weight) in choices {
                        draw -= weight;
                        if draw <= 0.0 {
                            available = Some(counts);
                            break;
                        }
                    }
                }
            }
            // A very narrow request can exhaust relation groups; back off the
            // relation cooldown while keeping exact-text history enforced.
            let available = available.filter(|counts| counts[o.difficulty.id() - 3] > 0);
            let masses = constraints.map(|tree| tree.completion_masses(available.as_deref()));
            for _ in 0..o.max_attempts {
                self.last_stats.attempts += 1;
                let mut ids = vec![0, o.difficulty.id()];
                let mut ended = false;
                let mut kana_len = 0;
                let mut frame_node = o.difficulty.id() - 3;
                while ids.len() < model.config.max_seq_len {
                    let logits = model.logits(&ids)?;
                    let mut last = logits.last().unwrap().clone();
                    mask_candidates(&mut last, vocab, &ids, kana_len, min, max);
                    if self.lexical_boundary_guard || self.polite_boundary_guard {
                        if let Some(previous) = ids
                            .last()
                            .filter(|&&id| id >= 7)
                            .map(|&id| &vocab.tokens[id])
                        {
                            for (id, logit) in last.iter_mut().enumerate().skip(7) {
                                if (self.lexical_boundary_guard
                                    && lexical_collision(previous, &vocab.tokens[id]))
                                    || (self.polite_boundary_guard
                                        && polite_collision(previous, &vocab.tokens[id]))
                                {
                                    *logit = f32::NEG_INFINITY;
                                }
                            }
                        }
                    }
                    if let Some(tree) = constraints {
                        for (id, value) in last.iter_mut().enumerate() {
                            if let Some(&next) = tree.nodes[frame_node].children.get(&id) {
                                // A prior over valid continuations compensates for
                                // narrow branches without consulting training text.
                                let count = masses.as_ref().unwrap()[next];
                                if count == 0.0 {
                                    *value = f32::NEG_INFINITY;
                                    continue;
                                }
                                *value += o.temperature * tree.completion_prior * count.ln();
                            } else {
                                *value = f32::NEG_INFINITY;
                            }
                        }
                    }
                    if let Some(tree) = constraints.filter(|tree| tree.probability_mix > 0.0) {
                        let counts: Vec<_> = (0..last.len())
                            .map(|id| {
                                tree.nodes[frame_node]
                                    .children
                                    .get(&id)
                                    .map_or(0.0, |&next| masses.as_ref().unwrap()[next])
                            })
                            .collect();
                        mix_frame_probabilities(
                            &mut last,
                            &counts,
                            o.temperature,
                            tree.probability_mix,
                        );
                    }
                    if !last.iter().any(|x| x.is_finite()) {
                        break;
                    }
                    let id = sample_with_entropy(
                        &last,
                        o.temperature,
                        o.top_p,
                        constraints.and_then(|tree| tree.top_h),
                        &mut self.rng,
                    )?;
                    self.last_stats.model_tokens += 1;
                    if id == 1 {
                        if let Some(tree) = constraints {
                            frame_node = tree.nodes[frame_node].children[&id];
                        }
                        ended = true;
                        break;
                    }
                    kana_len += typing_token_len(&vocab.tokens[id]);
                    if kana_len > max {
                        break;
                    }
                    ids.push(id);
                    if let Some(tree) = constraints {
                        frame_node = tree.nodes[frame_node].children[&id];
                    }
                }
                let source = if constraints.is_some() {
                    PromptSource::ModelConstrained
                } else {
                    PromptSource::Model
                };
                let mut segments = vocab.decode(&ids[2..])?;
                if self.normalize_morph_readings {
                    normalize_morph_readings(&mut segments);
                }
                let mut candidate = TypingPrompt::from_segments(segments, source);
                if self.normalize_morph_readings {
                    candidate.reading_validation =
                        "lexical_morphology_normalization; contextual_reading_not_fully_verified"
                            .into();
                }
                if ended && candidate.validate(o).is_ok() && !self.history.contains(&candidate.text)
                {
                    if let Some(tree) = constraints {
                        if let Some(relation) = tree.nodes[frame_node].relation {
                            let limit = if self.history_size > 0 {
                                tree.relation_history_size
                            } else {
                                0
                            };
                            if limit > 0 {
                                self.recent_relations.push_back(relation);
                                while self.recent_relations.len() > limit {
                                    self.recent_relations.pop_front();
                                }
                            }
                        }
                    }
                    self.remember(&candidate.text);
                    return Ok(candidate.without_terminal_period());
                }
                self.last_stats.rejected += 1;
            }
            self.last_stats.fallback_reason = Some("model attempts exhausted".into());
        } else {
            self.last_stats.fallback_reason = Some("explicit fallback-only mode".into());
        }
        ensure!(
            self.static_fallback_enabled,
            "autoregressive generation exhausted attempts; static fallback is disabled"
        );
        let candidates: Vec<_> = self
            .fallback
            .iter()
            .filter(|p| p.validate(o).is_ok() && !self.history.contains(&p.text))
            .cloned()
            .collect();
        ensure!(
            !candidates.is_empty(),
            "no fallback satisfies constraints/history; reduce history size or change constraints"
        );
        let p = candidates[self.rng.gen_range(0..candidates.len())].clone();
        self.remember(&p.text);
        Ok(p.without_terminal_period())
    }
}
// Small structural guard, not a semantic or contextual-reading validator.
fn lexical_collision(previous: &Token, token: &Token) -> bool {
    previous.pos == "名詞"
        && previous.surface.chars().last().is_some_and(is_kanji)
        && ((token.pos == "名詞" && token.surface.chars().next().is_some_and(is_kanji))
            || token.pos == "文字")
}
// A lexical ambiguity, independent of sentence/training membership: the noun
// 思い出 is おもいで, but the stem in 思い出しました is おもいだ.
fn normalize_morph_readings(segments: &mut [Segment]) {
    for i in 0..segments.len().saturating_sub(1) {
        if segments[i].surface != "思い出" || segments[i].kana != "おもいで" {
            continue;
        }
        let following: String = segments[i + 1..]
            .iter()
            .take(5)
            .map(|s| s.surface.as_str())
            .collect();
        if [
            "しました",
            "します",
            "した",
            "して",
            "す",
            "せる",
            "せば",
            "さない",
        ]
        .iter()
        .any(|suffix| following.starts_with(suffix))
        {
            segments[i].kana = "おもいだ".into();
            segments[i].pos = "動詞".into();
        }
    }
}
fn polite_collision(previous: &Token, token: &Token) -> bool {
    matches!(previous.pos.as_str(), "動詞" | "助動詞")
        && ["ます", "ました", "です", "でした", "ません", "ませんでした"]
            .iter()
            .any(|suffix| previous.surface.ends_with(suffix))
        && !matches!(
            token.surface.as_str(),
            "。" | "、" | "が" | "ので" | "から" | "と" | "し" | "けど" | "けれど" | "けれども"
        )
}
fn can_end(token: &Token) -> bool {
    match token.pos.as_str() {
        "動詞" => !["て", "で", "に", "ず"]
            .iter()
            .any(|s| token.surface.ends_with(s)),
        "形容詞" => !["く", "な"].iter().any(|s| token.surface.ends_with(s)),
        "名詞" | "助動詞" | "成句" => true,
        _ => false,
    }
}
fn typing_token_len(token: &Token) -> usize {
    if token.surface == "、" {
        1
    } else {
        token.kana.chars().count()
    }
}
fn mask_candidates(
    logits: &mut [f32],
    vocab: &Vocab,
    ids: &[usize],
    kana_len: usize,
    min: usize,
    max: usize,
) {
    let previous = ids
        .last()
        .filter(|&&id| id >= 7)
        .map(|&id| &vocab.tokens[id]);
    let ended = previous.is_some_and(|t| matches!(t.surface.as_str(), "。" | "！" | "？"));
    for (id, logit) in logits.iter_mut().enumerate() {
        let allowed = if id == 1 {
            ended && kana_len >= min
        } else if id < 7 || ended {
            false
        } else {
            let token = &vocab.tokens[id];
            let terminal = matches!(token.surface.as_str(), "。" | "！" | "？");
            // Literal characters let the story model spell arbitrary names
            // without a dictionary entry. Typing prompts use lexical tokens.
            token.pos != "文字"
                && !token
                    .pos
                    .split('-')
                    .any(|part| matches!(part, "人名" | "固有名詞"))
                && kana_len + typing_token_len(token) <= max
                && (!terminal || (kana_len >= min && previous.is_some_and(can_end)))
        };
        if !allowed {
            *logit = f32::NEG_INFINITY;
        }
    }
}
// Blend with the authored completion distribution, never with training-text
// membership. Model probabilities retain at least half the mass.
fn mix_frame_probabilities(logits: &mut [f32], counts: &[f32], temperature: f32, alpha: f32) {
    let max = logits
        .iter()
        .filter(|x| x.is_finite())
        .map(|x| *x as f64 / temperature as f64)
        .fold(f64::NEG_INFINITY, f64::max);
    if !max.is_finite() {
        return;
    }
    let z: f64 = logits
        .iter()
        .filter(|x| x.is_finite())
        .map(|x| (*x as f64 / temperature as f64 - max).exp())
        .sum();
    let total: f32 = logits
        .iter()
        .enumerate()
        .filter(|(_, x)| x.is_finite())
        .map(|(id, _)| counts[id])
        .sum();
    if total == 0.0 {
        return;
    }
    for (id, value) in logits.iter_mut().enumerate().filter(|(_, x)| x.is_finite()) {
        let p = (1.0 - alpha as f64) * (*value as f64 / temperature as f64 - max).exp() / z
            + alpha as f64 * counts[id] as f64 / total as f64;
        *value = (temperature as f64 * p.ln()) as f32;
    }
}
#[cfg(test)]
fn sample(logits: &[f32], temperature: f32, top_p: f32, rng: &mut StdRng) -> Result<usize> {
    sample_with_entropy(logits, temperature, top_p, None, rng)
}
fn sample_with_entropy(
    logits: &[f32],
    temperature: f32,
    top_p: f32,
    top_h: Option<f64>,
    rng: &mut StdRng,
) -> Result<usize> {
    let mut sorted: Vec<_> = logits
        .iter()
        .enumerate()
        .filter(|(_, x)| x.is_finite())
        .map(|(i, x)| (i, *x as f64 / temperature as f64))
        .collect();
    ensure!(!sorted.is_empty(), "no finite sampling candidates");
    sorted.sort_by(|a, b| b.1.total_cmp(&a.1));
    let max = sorted[0].1;
    let denom: f64 = sorted.iter().map(|(_, x)| (x - max).exp()).sum();
    let entropy: f64 = sorted
        .iter()
        .map(|(_, x)| {
            let p = (x - max).exp() / denom;
            if p > 0.0 {
                -p * p.ln()
            } else {
                0.0
            }
        })
        .sum();
    let mut retained = Vec::new();
    let mut mass = 0.0;
    let mut partial_entropy = 0.0;
    for (i, x) in sorted {
        let p = (x - max).exp() / denom;
        let next_mass = mass + p;
        let next_entropy = partial_entropy - if p > 0.0 { p * p.ln() } else { 0.0 };
        // Entropy of the renormalized prefix. Always retain the most likely token.
        if !retained.is_empty()
            && top_h
                .is_some_and(|h| next_entropy / next_mass + next_mass.ln() > h * entropy + 1e-12)
        {
            break;
        }
        partial_entropy = next_entropy;
        mass += p;
        retained.push((i, p));
        if top_h.is_none() && mass >= top_p as f64 {
            break;
        }
    }
    let draw = rng.gen::<f64>() * mass;
    let mut cumulative = 0.0;
    for (i, p) in &retained {
        cumulative += p;
        if draw < cumulative {
            return Ok(*i);
        }
    }
    Ok(retained.last().unwrap().0)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn reading_boundary_guard_does_not_create_unseen_compound_readings() {
        let token = |s: &str, p: &str| Token {
            id: 7,
            surface: s.into(),
            kana: "".into(),
            pos: p.into(),
            metadata: Value::Null,
        };
        assert!(lexical_collision(
            &token("風", "名詞"),
            &token("船", "名詞")
        ));
        assert!(lexical_collision(
            &token("光", "名詞"),
            &token("る", "文字")
        ));
        assert!(!lexical_collision(
            &token("風船", "名詞"),
            &token("が", "助詞")
        ));
        assert!(!lexical_collision(
            &token("毎日", "名詞"),
            &token("遊ぶ", "動詞")
        ));
        assert!(polite_collision(
            &token("遊びました", "動詞"),
            &token("いる", "動詞")
        ));
        assert!(!polite_collision(
            &token("遊びました", "動詞"),
            &token("。", "記号")
        ));
        assert!(!polite_collision(
            &token("遊びました", "動詞"),
            &token("が", "助詞")
        ));
        assert!(!polite_collision(
            &token("読む", "動詞"),
            &token("本", "名詞")
        ));
    }
    #[test]
    fn noun_memory_and_verb_recall_have_different_readings() {
        let segment = |s: &str, k: &str, p: &str| Segment {
            surface: s.into(),
            kana: k.into(),
            pos: p.into(),
        };
        let mut noun = vec![
            segment("思い出", "おもいで", "名詞"),
            segment("を", "を", "助詞"),
            segment("残す", "のこす", "動詞"),
        ];
        normalize_morph_readings(&mut noun);
        assert_eq!(noun[0].kana, "おもいで");
        let mut verb = vec![
            segment("思い出", "おもいで", "名詞"),
            segment("しました", "しました", "動詞"),
        ];
        normalize_morph_readings(&mut verb);
        assert_eq!(
            verb.iter().map(|s| s.kana.as_str()).collect::<String>(),
            "おもいだしました"
        );
    }
    #[test]
    fn authored_probability_mix_preserves_masks_and_model_mass() {
        let mut logits = vec![0.99_f32.ln(), 0.01_f32.ln(), f32::NEG_INFINITY];
        mix_frame_probabilities(&mut logits, &[1.0, 1.0, 100.0], 1.0, 0.35);
        assert!((logits[0].exp() - 0.8185).abs() < 1e-6);
        assert!((logits[1].exp() - 0.1815).abs() < 1e-6);
        assert!(!logits[2].is_finite());
        let mut masked = vec![f32::NEG_INFINITY; 2];
        mix_frame_probabilities(&mut masked, &[1.0, 1.0], 0.8, 0.35);
        assert!(masked.iter().all(|x| !x.is_finite()));
    }
    #[test]
    fn frame_completion_counts_and_internal_end_marker_are_checked() -> Result<()> {
        let mut tokens: Vec<_> = SPECIALS
            .iter()
            .enumerate()
            .map(|(id, surface)| Token {
                id,
                surface: (*surface).into(),
                kana: "".into(),
                pos: "special".into(),
                metadata: Value::Null,
            })
            .collect();
        for (surface, kana, pos) in [
            ("猫", "ねこ", "名詞"),
            ("が", "が", "助詞"),
            ("眠る", "ねむる", "動詞"),
            ("。", "", "記号"),
            ("犬", "いぬ", "名詞"),
            ("走る", "はしる", "動詞"),
        ] {
            tokens.push(Token {
                id: tokens.len(),
                surface: surface.into(),
                kana: kana.into(),
                pos: pos.into(),
                metadata: Value::Null,
            });
        }
        let vocab = Vocab {
            version: VERSION.into(),
            encoding: "UTF-8".into(),
            tokens,
        };
        let value = serde_json::json!({"version":"semantic_frames_v1","completion_prior":1.0,
            "sequences":[[3,7,8,9,10,1],[3,7,8,12,10,1],[3,11,8,9,10,1]]});
        let tree = FrameConstraints::load(&value, &vocab, 16)?;
        assert_eq!(tree.nodes[0].completions, 3);
        assert_eq!(tree.nodes[tree.nodes[0].children[&7]].completions, 2);
        assert_eq!(tree.nodes[tree.nodes[0].children[&11]].completions, 1);
        assert_eq!(tree.nodes[1].completions, 0);
        let mut grouped = value.clone();
        grouped["relation_ids"] = serde_json::json!([0, 0, 1]);
        grouped["relation_history_size"] = serde_json::json!(12);
        let grouped_tree = FrameConstraints::load(&grouped, &vocab, 16)?;
        let mut balanced = grouped.clone();
        balanced["balance_relations"] = serde_json::json!(true);
        let balanced_tree = FrameConstraints::load(&balanced, &vocab, 16)?;
        let mass = balanced_tree.completion_masses(None);
        assert!((mass[0] - 2.0).abs() < 1e-6);
        assert!((mass[balanced_tree.nodes[0].children[&7]] - 1.0).abs() < 1e-6);
        let masked_mass = balanced_tree.completion_masses(Some(
            &balanced_tree.available_counts(&VecDeque::from([0]), None),
        ));
        assert!((masked_mass[0] - 1.0).abs() < 1e-6);
        for bad_h in [
            serde_json::json!(0),
            serde_json::json!(1.1),
            serde_json::json!("wrong"),
        ] {
            let mut invalid = balanced.clone();
            invalid["top_h"] = bad_h;
            assert!(FrameConstraints::load(&invalid, &vocab, 16).is_err());
        }
        let mut invalid = balanced.clone();
        invalid["sequences"] = serde_json::json!([[]]);
        invalid["relation_ids"] = serde_json::json!([0]);
        assert!(FrameConstraints::load(&invalid, &vocab, 16).is_err());
        let counts = grouped_tree.available_counts(&VecDeque::from([0]), None);
        assert_eq!(counts[0], 1);
        assert_eq!(counts[grouped_tree.nodes[0].children[&7]], 0);
        assert_eq!(counts[grouped_tree.nodes[0].children[&11]], 1);
        assert_eq!(
            grouped_tree.available_counts(&VecDeque::from([1]), None)[0],
            2
        );
        grouped["kind_ids"] = serde_json::json!([0, 0, 1]);
        grouped["kind_weights"] = serde_json::json!([0.75, 0.25]);
        let mixed_tree = FrameConstraints::load(&grouped, &vocab, 16)?;
        assert_eq!(mixed_tree.available_counts(&VecDeque::new(), Some(0))[0], 2);
        assert_eq!(mixed_tree.available_counts(&VecDeque::new(), Some(1))[0], 1);
        assert_eq!(
            mixed_tree.available_counts(&VecDeque::from([0]), Some(0))[0],
            0
        );
        grouped["relation_ids"] = serde_json::json!([0]);
        assert!(FrameConstraints::load(&grouped, &vocab, 16).is_err());
        let bad = serde_json::json!({"version":"semantic_frames_v1","sequences":[[3,7,8,9,1]]});
        assert!(FrameConstraints::load(&bad, &vocab, 16).is_err());
        let mut bad = value;
        bad["completion_prior"] = serde_json::json!("wrong");
        assert!(FrameConstraints::load(&bad, &vocab, 16).is_err());
        Ok(())
    }
    #[test]
    fn entropy_budget_preserves_masks_and_can_keep_a_flat_distribution() -> Result<()> {
        let mut rng = StdRng::seed_from_u64(42);
        for _ in 0..100 {
            assert_eq!(
                sample_with_entropy(
                    &[0.9_f32.ln(), 0.1_f32.ln(), f32::NEG_INFINITY],
                    1.0,
                    1.0,
                    Some(0.5),
                    &mut rng
                )?,
                0
            );
        }
        let mut seen = std::collections::BTreeSet::new();
        for _ in 0..100 {
            seen.insert(sample_with_entropy(
                &[0.0, 0.0, 0.0, f32::NEG_INFINITY],
                1.0,
                0.1,
                Some(1.0),
                &mut rng,
            )?);
        }
        assert_eq!(seen, std::collections::BTreeSet::from([0, 1, 2]));
        assert_eq!(
            sample_with_entropy(&[f32::NEG_INFINITY, 0.0], 1.0, 1.0, Some(0.5), &mut rng)?,
            1
        );
        Ok(())
    }
    #[test]
    fn sentence_end_guard() {
        let token = |surface: &str, pos: &str| Token {
            id: 7,
            surface: surface.into(),
            kana: "".into(),
            pos: pos.into(),
            metadata: Value::Null,
        };
        for (surface, pos) in [
            ("を", "助詞"),
            ("読んで", "動詞"),
            ("取りに", "動詞"),
            ("薄く", "形容詞"),
            ("静かな", "形容詞"),
            ("ゆっくり", "副詞"),
        ] {
            assert!(!can_end(&token(surface, pos)));
        }
        for (surface, pos) in [
            ("読んだ", "動詞"),
            ("きた", "動詞"),
            ("いる", "動詞"),
            ("温かい", "形容詞"),
            ("休み", "名詞"),
            ("一期一会", "成句"),
        ] {
            assert!(can_end(&token(surface, pos)));
        }
        let vocab = Vocab {
            version: VERSION.into(),
            encoding: "UTF-8".into(),
            tokens: (0..10)
                .map(|id| {
                    let mut t = token(if id == 8 { "。" } else { "本" }, "名詞");
                    t.id = id;
                    t.kana = if id == 8 { "" } else { "ほん" }.into();
                    t
                })
                .collect(),
        };
        let mut logits = vec![0.; 10];
        mask_candidates(&mut logits, &vocab, &[0, 3], 0, 5, 12);
        assert!(!logits[1].is_finite() && !logits[8].is_finite());
        let mut logits = vec![0.; 10];
        mask_candidates(&mut logits, &vocab, &[0, 3, 7], 12, 5, 12);
        assert!(logits[8].is_finite() && !logits[7].is_finite());
        let mut logits = vec![0.; 10];
        mask_candidates(&mut logits, &vocab, &[0, 3, 7, 8], 12, 5, 12);
        assert_eq!(logits.iter().filter(|x| x.is_finite()).count(), 1);
        assert!(logits[1].is_finite());
    }
    #[test]
    fn comma_counts_as_one_input_character_at_generation_boundary() {
        let comma = Token {
            id: 7,
            surface: "、".into(),
            kana: String::new(),
            pos: "記号".into(),
            metadata: Value::Null,
        };
        assert_eq!(typing_token_len(&comma), 1);
        let vocab = Vocab {
            version: VERSION.into(),
            encoding: "UTF-8".into(),
            tokens: vec![comma; 8],
        };
        for (length, allowed) in [(11, true), (12, false)] {
            let mut logits = vec![0.0; 8];
            mask_candidates(&mut logits, &vocab, &[0, 3], length, 5, 12);
            assert_eq!(logits[7].is_finite(), allowed);
        }
    }
    #[test]
    fn typing_prompts_exclude_character_spelling_and_named_entities() {
        let mut tokens: Vec<_> = fumiori_core::SPECIALS
            .iter()
            .enumerate()
            .map(|(id, surface)| Token {
                id,
                surface: (*surface).into(),
                kana: String::new(),
                pos: "special".into(),
                metadata: Value::Null,
            })
            .collect();
        for (surface, kana, pos) in [
            ("ま", "ま", "文字"),
            ("ト", "と", "文字"),
            ("太郎", "たろう", "名詞-固有名詞-人名"),
            ("東京", "とうきょう", "固有名詞"),
            ("花子", "はなこ", "人名"),
            ("猫", "ねこ", "名詞"),
            ("みかん", "みかん", "名詞"),
            ("が", "が", "助詞"),
            ("眠る", "ねむる", "動詞"),
        ] {
            tokens.push(Token {
                id: tokens.len(),
                surface: surface.into(),
                kana: kana.into(),
                pos: pos.into(),
                metadata: Value::Null,
            });
        }
        let vocab = Vocab {
            version: VERSION.into(),
            encoding: "UTF-8".into(),
            tokens,
        };
        vocab.validate().unwrap();
        for (ids, kana_len) in [(vec![0, 3], 0), (vec![0, 3, 12, 14], 3)] {
            let mut logits = vec![0.0; vocab.tokens.len()];
            mask_candidates(&mut logits, &vocab, &ids, kana_len, 5, 12);
            assert!(logits[7..12].iter().all(|x| !x.is_finite()));
            assert!(logits[12..].iter().all(|x| x.is_finite()));
        }
    }
    #[test]
    fn reproducible_sampling() {
        let mut a = StdRng::seed_from_u64(42);
        let mut b = StdRng::seed_from_u64(42);
        for _ in 0..100 {
            assert_eq!(
                sample(&[0.0, 1.0, f32::NEG_INFINITY], 0.8, 0.9, &mut a).unwrap(),
                sample(&[0.0, 1.0, f32::NEG_INFINITY], 0.8, 0.9, &mut b).unwrap()
            );
        }
        assert!(sample(&[f32::NAN], 1.0, 1.0, &mut a).is_err());
    }
    #[test]
    fn fallback_constraints_history() -> Result<()> {
        let mut a = TypingGenerator::fallback_only()?;
        let mut b = TypingGenerator::fallback_only()?;
        for d in [Difficulty::Easy, Difficulty::Normal, Difficulty::Hard] {
            let o = GenerateOptions {
                difficulty: d,
                require_kanji: true,
                seed: Some(42),
                ..Default::default()
            };
            for _ in 0..3 {
                let p = a.generate(&o)?;
                p.validate(&o)?;
                assert_eq!(p.source, PromptSource::Fallback);
                assert_eq!(p.text, b.generate(&o)?.text);
            }
        }
        Ok(())
    }
    #[test]
    fn bounded_exhaustion() -> Result<()> {
        let mut g = TypingGenerator::fallback_only()?;
        let o = GenerateOptions {
            difficulty: Difficulty::Easy,
            min_kana_chars: 5,
            max_kana_chars: 5,
            ..Default::default()
        };
        for _ in 0..30 {
            if g.generate(&o).is_err() {
                return Ok(());
            }
        }
        anyhow::bail!("history should exhaust finite catalog")
    }
    #[test]
    fn missing_model_fails() {
        assert!(TypingGenerator::load("missing-model-path").is_err());
    }
}
