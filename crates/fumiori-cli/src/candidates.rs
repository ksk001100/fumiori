//! Offline research only: no training-reference lookup or language-quality oracle.
use super::{env_info, percentile, write_report, Level};
use anyhow::{ensure, Context, Result};
use clap::Args;
use fumiori_core::{Difficulty, TypingPrompt, Vocab};
use fumiori_inference::TypingGenerator;
use serde_json::{json, Value};
use std::{collections::BTreeMap, path::PathBuf, time::Instant};

#[derive(Args)]
pub(super) struct CandidateArgs {
    #[arg(long)]
    model: PathBuf,
    #[arg(long, default_value = "600")]
    samples: usize,
    #[arg(long, default_value = "870000")]
    seed: u64,
    #[arg(long, value_enum, default_value = "mixed")]
    difficulty: Level,
    #[arg(long, default_value = "0.8")]
    temperature: f32,
    #[arg(long, default_value = "0.95")]
    top_p: f32,
    #[arg(long, default_value = "4")]
    max_candidates: usize,
    #[arg(long)]
    report: PathBuf,
}

type TokenKey = (String, String, String);

fn token_index(vocab: &Vocab) -> BTreeMap<TokenKey, usize> {
    vocab
        .tokens
        .iter()
        .skip(7)
        .map(|t| {
            let kana = if t.surface == "、" && t.kana.is_empty() {
                "、"
            } else {
                &t.kana
            };
            ((t.surface.clone(), kana.to_owned(), t.pos.clone()), t.id)
        })
        .collect()
}

fn encode(
    prompt: &TypingPrompt,
    difficulty: Difficulty,
    index: &BTreeMap<TokenKey, usize>,
) -> Result<Vec<usize>> {
    let level = match difficulty {
        Difficulty::Easy => 3,
        Difficulty::Normal => 4,
        Difficulty::Hard => 5,
    };
    let mut ids = vec![0, level];
    for segment in &prompt.segments {
        let key = (
            segment.surface.clone(),
            segment.kana.clone(),
            segment.pos.clone(),
        );
        ids.push(
            *index
                .get(&key)
                .context("public segment has no exact paired token; score unavailable")?,
        );
    }
    // The public API removes the period; the trained sequence includes it.
    ids.push(
        *index
            .get(&("。".into(), "".into(), "記号".into()))
            .context("missing terminal period token")?,
    );
    Ok(ids)
}

fn mean_log_probability(logits: &[Vec<f32>], targets: &[usize]) -> Result<f64> {
    ensure!(
        logits.len() == targets.len() && !targets.is_empty(),
        "score shape mismatch"
    );
    let mut sum = 0.0;
    for (row, &target) in logits.iter().zip(targets) {
        ensure!(
            target < row.len() && row.iter().all(|v| v.is_finite()),
            "invalid score logits"
        );
        let maximum = row
            .iter()
            .copied()
            .map(f64::from)
            .fold(f64::NEG_INFINITY, f64::max);
        let partition = row
            .iter()
            .map(|&v| (f64::from(v) - maximum).exp())
            .sum::<f64>()
            .ln();
        sum += f64::from(row[target]) - maximum - partition;
    }
    Ok(sum / targets.len() as f64)
}

fn best_index(scores: &[Option<f64>]) -> Option<usize> {
    let mut best: Option<usize> = None;
    for (i, score) in scores.iter().enumerate() {
        if let Some(score) = score {
            if best.is_none_or(|j| *score > scores[j].unwrap()) {
                best = Some(i);
            }
        }
    }
    best
}

pub(super) fn run(args: CandidateArgs) -> Result<()> {
    ensure!(
        matches!(args.max_candidates, 1 | 2 | 4),
        "max-candidates must be 1, 2 or 4"
    );
    ensure!(
        (1..=10000).contains(&args.samples),
        "samples must be 1..10000"
    );
    ensure!(
        args.seed.checked_add(args.samples as u64 * 4 - 1).is_some(),
        "seed range overflows"
    );
    ensure!(!args.report.exists(), "report already exists");
    let mut generator = TypingGenerator::load(&args.model)?;
    generator.set_history_size(0);
    let vocab = generator
        .vocab
        .as_ref()
        .context("paired-token model required")?;
    ensure!(vocab.encoding == "UTF-8", "paired-token model required");
    let index = token_index(vocab);
    let mut options = generator.default_options();
    options.temperature = args.temperature;
    options.top_p = args.top_p;
    let mut pools = Vec::new();
    let mut variants: Vec<Value> = [1,2,4].iter().filter(|&&n| n <= args.max_candidates).map(|n| json!({"candidate_count":n,"requested":args.samples,"samples":[],"failures":[],"durations_ms":[]})).collect();
    for request in 0..args.samples {
        options.difficulty =
            args.difficulty.difficulties()[request % args.difficulty.difficulties().len()];
        let mut pool = Vec::new();
        let mut scores = Vec::new();
        for candidate in 0..args.max_candidates {
            let seed = args.seed + request as u64 * 4 + candidate as u64;
            options.seed = Some(seed);
            generator.clear_history();
            let start = Instant::now();
            let prompt = generator.generate(&options);
            let generation_ms = start.elapsed().as_secs_f64() * 1000.0;
            let start = Instant::now();
            let score = prompt
                .as_ref()
                .map_err(|e| anyhow::anyhow!(e.to_string()))
                .and_then(|p| {
                    let ids = encode(p, options.difficulty, &index)?;
                    let logits = generator.logits(&ids)?;
                    let mut targets = ids[2..].to_vec();
                    targets.push(1); // EOS in the paired-token format.
                    mean_log_probability(&logits[1..], &targets)
                });
            let scoring_ms = start.elapsed().as_secs_f64() * 1000.0;
            scores.push(score.as_ref().ok().copied());
            pool.push(json!({"seed":seed,"prompt":prompt.as_ref().ok(),"generation_error":prompt.err().map(|e| e.to_string()),
                "mean_log_probability":score.as_ref().ok(),"scoring_error":score.err().map(|e|e.to_string()),
                "generation_ms":generation_ms,"scoring_ms":scoring_ms}));
        }
        for variant in &mut variants {
            let n = variant["candidate_count"].as_u64().unwrap() as usize;
            // Candidate 1 is the unreranked control, even if it cannot be scored.
            let winner = if n == 1 {
                (!pool[0]["prompt"].is_null()).then_some(0)
            } else {
                best_index(&scores[..n])
            };
            let elapsed: f64 = pool[..n]
                .iter()
                .map(|r| {
                    r["generation_ms"].as_f64().unwrap()
                        + if n == 1 {
                            0.0
                        } else {
                            r["scoring_ms"].as_f64().unwrap()
                        }
                })
                .sum();
            variant["durations_ms"]
                .as_array_mut()
                .unwrap()
                .push(json!(elapsed));
            if let Some(winner) = winner {
                let mut prompt = pool[winner]["prompt"].clone();
                prompt["request_id"] = json!(request);
                prompt["candidate_seed"] = pool[winner]["seed"].clone();
                variant["samples"].as_array_mut().unwrap().push(prompt);
            } else {
                variant["failures"]
                    .as_array_mut()
                    .unwrap()
                    .push(json!({"request_id":request,"error":"no eligible candidate"}));
            }
        }
        pools.push(json!({"request_id":request,"difficulty":format!("{:?}",options.difficulty),"candidates":pool}));
    }
    for variant in &mut variants {
        let durations: Vec<_> = variant["durations_ms"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();
        variant["p50_ms"] = json!(percentile(&durations, 0.5));
        variant["p95_ms"] = json!(percentile(&durations, 0.95));
        variant["success"] = json!(variant["samples"].as_array().unwrap().len());
    }
    write_report(
        &args.report,
        &json!({"model":args.model,"requested":args.samples,"seed_start":args.seed,
        "temperature":args.temperature,"top_p":args.top_p,"history_size":0,"environment":env_info(),
        "protocol":"Independent request pools, seed = start + request_id*4 + candidate_index; shared prefixes for N=1/2/4. Mean unmasked student log probability of content + period + EOS; excludes difficulty prediction. No training-reference lookup. Ties prefer first candidate. Scores are not naturalness or reading accuracy.",
        "latency_scope":"Sum of actual per-candidate generation and full-logit scoring; N=1 excludes unused scoring. Includes retries/failures, excludes load, JSON report serialization and ranking bookkeeping. Research CPU path, not WASM.",
        "variants":variants,"pools":pools}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn likelihood_is_stable_normalized_and_includes_all_targets() {
        let score = mean_log_probability(&[vec![1000.0, 1000.0], vec![0.0, 0.0]], &[0, 1]).unwrap();
        assert!((score + 2.0f64.ln()).abs() < 1e-10);
        assert!(mean_log_probability(&[vec![f32::NAN]], &[0]).is_err());
        assert!(mean_log_probability(&[], &[]).is_err());
    }
    #[test]
    fn selection_keeps_ties_and_missing_scores_explicit() {
        assert_eq!(
            best_index(&[None, Some(-2.0), Some(-2.0), Some(-3.0)]),
            Some(1)
        );
        assert_eq!(best_index(&[None, None]), None);
    }
}
