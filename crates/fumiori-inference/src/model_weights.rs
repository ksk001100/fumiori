use anyhow::{ensure, Context, Result};
use candle_core::{DType, Device, Tensor};
use std::collections::HashMap;

pub(crate) fn load(weights: &[u8], storage: &str) -> Result<HashMap<String, Tensor>> {
    let mut tensors = candle_core::safetensors::load_buffer(weights, &Device::Cpu)?;
    let mut restored_elements = 0usize;
    if storage == "I8_ROW_F32" {
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
    } else if storage == "I4_BLOCK128_F32" {
        let names: Vec<_> = tensors
            .keys()
            .filter(|n| n.ends_with(".shape"))
            .map(|n| n.trim_end_matches(".shape").to_string())
            .collect();
        ensure!(!names.is_empty(), "missing packed INT4 tensors");
        for name in names {
            let shape = tensors
                .remove(&format!("{name}.shape"))
                .context("missing INT4 shape")?;
            ensure!(
                shape.dtype() == DType::U32 && shape.dims() == [2],
                "invalid INT4 shape"
            );
            let dims = shape.to_vec1::<u32>()?;
            let rows = dims[0] as usize;
            let cols = dims[1] as usize;
            let count = rows.checked_mul(cols).context("INT4 dimension overflow")?;
            restored_elements = restored_elements
                .checked_add(count)
                .context("INT4 total overflow")?;
            ensure!(
                restored_elements <= 64_000_000,
                "oversized restored INT4 weights"
            );
            ensure!(
                rows > 0 && cols > 0 && count <= 16_000_000 && count.is_multiple_of(128),
                "invalid INT4 dimensions"
            );
            let q = tensors.remove(&name).context("missing packed INT4 data")?;
            let scale = tensors
                .remove(&format!("{name}.scale"))
                .context("missing INT4 scale")?;
            ensure!(
                q.dtype() == DType::U8 && q.dims() == [count / 2],
                "invalid INT4 packing"
            );
            ensure!(
                scale.dtype() == DType::F32 && scale.dims() == [count / 128],
                "invalid INT4 scale shape"
            );
            let q = q.to_vec1::<u8>()?;
            let scales = scale.to_vec1::<f32>()?;
            ensure!(
                scales.iter().all(|s| s.is_finite() && *s > 0.0),
                "invalid INT4 scale"
            );
            let values = (0..count)
                .map(|i| {
                    let packed = q[i / 2];
                    let nibble = if i % 2 == 0 { packed & 15 } else { packed >> 4 };
                    (nibble as f32 - 8.0) * scales[i / 128]
                })
                .collect::<Vec<_>>();
            tensors.insert(name, Tensor::from_vec(values, (rows, cols), &Device::Cpu)?);
        }
    } else {
        ensure!(storage == "F32", "unsupported weight storage");
    }
    ensure!(
        tensors.values().all(|t| t.dtype() == DType::F32),
        "only FP32 weights supported"
    );
    for (name, tensor) in &tensors {
        ensure!(
            tensor
                .flatten_all()?
                .to_vec1::<f32>()?
                .iter()
                .all(|x| x.is_finite()),
            "nonfinite weights in {name}"
        );
    }
    Ok(tensors)
}
