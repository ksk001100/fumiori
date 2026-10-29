use crate::ModelConfig;
use anyhow::{ensure, Result};
use candle_core::{DType, Device, Module, Tensor};
use candle_nn::{Linear, RmsNorm, VarBuilder};
use std::collections::HashMap;

struct Block {
    input_norm: RmsNorm,
    post_norm: RmsNorm,
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    gate: Linear,
    up: Linear,
    down: Linear,
}
impl Block {
    fn load(c: &ModelConfig, vb: VarBuilder<'_>) -> Result<Self> {
        Ok(Self {
            input_norm: candle_nn::rms_norm(c.d_model, c.norm_eps, vb.pp("input_layernorm"))?,
            post_norm: candle_nn::rms_norm(
                c.d_model,
                c.norm_eps,
                vb.pp("post_attention_layernorm"),
            )?,
            q: candle_nn::linear_no_bias(c.d_model, c.d_model, vb.pp("self_attn.q_proj"))?,
            k: candle_nn::linear_no_bias(c.d_model, c.d_model, vb.pp("self_attn.k_proj"))?,
            v: candle_nn::linear_no_bias(c.d_model, c.d_model, vb.pp("self_attn.v_proj"))?,
            o: candle_nn::linear_no_bias(c.d_model, c.d_model, vb.pp("self_attn.o_proj"))?,
            gate: candle_nn::linear_no_bias(c.d_model, c.d_ff, vb.pp("mlp.gate_proj"))?,
            up: candle_nn::linear_no_bias(c.d_model, c.d_ff, vb.pp("mlp.up_proj"))?,
            down: candle_nn::linear_no_bias(c.d_ff, c.d_model, vb.pp("mlp.down_proj"))?,
        })
    }
    fn forward(&self, x: &Tensor, c: &ModelConfig, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let (b, t, d) = x.dims3()?;
        let dh = d / c.n_heads;
        let normalized = self.input_norm.forward(x)?;
        let heads = |linear: &Linear| -> candle_core::Result<Tensor> {
            linear
                .forward(&normalized)?
                .reshape((b, t, c.n_heads, dh))?
                .transpose(1, 2)?
                .contiguous()
        };
        let rotate = |x: Tensor| -> candle_core::Result<Tensor> {
            let rotated = Tensor::cat(
                &[
                    &x.narrow(3, dh / 2, dh / 2)?.neg()?,
                    &x.narrow(3, 0, dh / 2)?,
                ],
                3,
            )?;
            x.broadcast_mul(cos)? + rotated.broadcast_mul(sin)?
        };
        let (q, k, v) = (
            rotate(heads(&self.q)?)?,
            rotate(heads(&self.k)?)?,
            heads(&self.v)?,
        );
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
        let x = (x + self.o.forward(&z)?)?;
        let n = self.post_norm.forward(&x)?;
        let activated = (candle_nn::ops::silu(&self.gate.forward(&n)?)? * self.up.forward(&n)?)?;
        Ok((&x + self.down.forward(&activated)?)?)
    }
}
pub(crate) struct LlamaModel {
    embedding: Tensor,
    layers: Vec<Block>,
    norm: RmsNorm,
    head: Tensor,
}
impl LlamaModel {
    pub fn load(c: &ModelConfig, tensors: HashMap<String, Tensor>) -> Result<Self> {
        ensure!(
            tensors.len() == 3 + 9 * c.n_layers,
            "unexpected/missing Llama tensors"
        );
        let vb = VarBuilder::from_tensors(tensors, DType::F32, &Device::Cpu);
        Ok(Self {
            embedding: vb.get((c.vocab_size, c.d_model), "embed_tokens.weight")?,
            layers: (0..c.n_layers)
                .map(|i| Block::load(c, vb.pp(format!("layers.{i}"))))
                .collect::<Result<_>>()?,
            norm: candle_nn::rms_norm(c.d_model, c.norm_eps, vb.pp("norm"))?,
            head: vb.get((c.vocab_size, c.d_model), "lm_head.weight")?,
        })
    }
    pub fn compute_logits(
        &self,
        ids: &[usize],
        c: &ModelConfig,
        last_only: bool,
    ) -> Result<Vec<Vec<f32>>> {
        let t = ids.len();
        let dh = c.d_model / c.n_heads;
        let mut cos = Vec::with_capacity(t * dh);
        let mut sin = Vec::with_capacity(t * dh);
        for i in 0..t {
            for j in 0..dh {
                let inv = 1.0 / (c.rope_theta as f32).powf((2 * (j % (dh / 2))) as f32 / dh as f32);
                let angle = i as f32 * inv;
                cos.push(angle.cos());
                sin.push(angle.sin());
            }
        }
        let cos = Tensor::from_vec(cos, (1, 1, t, dh), &Device::Cpu)?;
        let sin = Tensor::from_vec(sin, (1, 1, t, dh), &Device::Cpu)?;
        let input = Tensor::new(
            ids.iter().map(|&i| i as u32).collect::<Vec<_>>(),
            &Device::Cpu,
        )?;
        let mut x = self.embedding.index_select(&input, 0)?.unsqueeze(0)?;
        for layer in &self.layers {
            x = layer.forward(&x, c, &cos, &sin)?;
        }
        let hidden = self.norm.forward(&x)?.squeeze(0)?;
        let hidden = if last_only {
            hidden.narrow(0, ids.len() - 1, 1)?
        } else {
            hidden
        };
        Ok(hidden.matmul(&self.head.t()?)?.to_vec2::<f32>()?)
    }
}
