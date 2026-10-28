use anyhow::{ensure, Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use fumiori_core::*;
use fumiori_inference::TypingGenerator;
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Path, PathBuf},
    time::Instant,
};
#[derive(Parser)]
#[command(version, about = "Offline Japanese paired-token generator")]
struct Args {
    #[command(subcommand)]
    command: Commands,
}
#[derive(Clone, Copy, ValueEnum)]
enum Level {
    Easy,
    Normal,
    Hard,
    Mixed,
}
impl Level {
    fn difficulties(self) -> &'static [Difficulty] {
        match self {
            Self::Easy => &[Difficulty::Easy],
            Self::Normal => &[Difficulty::Normal],
            Self::Hard => &[Difficulty::Hard],
            Self::Mixed => &[Difficulty::Easy, Difficulty::Normal, Difficulty::Hard],
        }
    }
}
#[derive(Subcommand)]
enum Commands {
    Generate {
        #[arg(
            long,
            required_unless_present = "fallback_only",
            conflicts_with = "fallback_only"
        )]
        model: Option<PathBuf>,
        #[arg(long)]
        fallback_only: bool,
        #[arg(long, value_enum, default_value = "normal")]
        difficulty: Level,
        #[arg(long, default_value = "5")]
        min_kana: usize,
        #[arg(long, default_value = "40")]
        max_kana: usize,
        #[arg(long)]
        require_kanji: bool,
        #[arg(long)]
        temperature: Option<f32>,
        #[arg(long)]
        top_p: Option<f32>,
        #[arg(long, default_value = "20")]
        max_attempts: usize,
        #[arg(long, default_value = "1")]
        count: usize,
        #[arg(long)]
        seed: Option<u64>,
        #[arg(long, default_value = "16")]
        history_size: usize,
        #[arg(long)]
        unconstrained: bool,
        #[arg(long,default_value="jsonl",value_parser=["jsonl"])]
        format: String,
    },
    Logits {
        #[arg(long)]
        model: PathBuf,
        #[arg(long, value_delimiter = ',')]
        ids: Vec<usize>,
    },
    Parity {
        #[arg(long)]
        model: PathBuf,
        #[arg(long, default_value = "0.0002")]
        tolerance: f32,
    },
    Evaluate {
        #[arg(long)]
        model: PathBuf,
        #[arg(long, default_value = "200")]
        samples: usize,
        #[arg(long, default_value = "build/eval.json")]
        report: PathBuf,
        #[arg(long, default_value = "build/stories-v1-data/train.jsonl")]
        training_data: PathBuf,
        #[arg(long, default_value = "42")]
        seed: u64,
        #[arg(long)]
        unconstrained: bool,
        #[arg(long, default_value_t = 0)]
        history_size: usize,
        #[arg(long, default_value = "0.8")]
        temperature: f32,
        #[arg(long, default_value = "0.9")]
        top_p: f32,
        /// Skip the quadratic near-training-string diagnostic; exact matches remain measured.
        #[arg(long)]
        skip_near_statistics: bool,
    },
    Bench {
        #[arg(long)]
        model: PathBuf,
        #[arg(long, default_value = "100")]
        samples: usize,
        #[arg(long, default_value = "build/bench.json")]
        report: PathBuf,
    },
}
fn write_report(path: &Path, v: &Value) -> Result<()> {
    if let Some(p) = path.parent() {
        fs::create_dir_all(p)?;
    }
    fs::write(path, serde_json::to_vec_pretty(v)?)?;
    println!("{}", serde_json::to_string(v)?);
    Ok(())
}
fn percentile(v: &[f64], p: f64) -> Option<f64> {
    if v.is_empty() {
        None
    } else {
        let mut x = v.to_vec();
        x.sort_by(f64::total_cmp);
        Some(x[((x.len() - 1) as f64 * p).ceil() as usize])
    }
}
fn env_info() -> Value {
    json!({"os":std::env::consts::OS,"arch":std::env::consts::ARCH,"cpu":std::env::var("PROCESSOR_IDENTIFIER").ok(),"build_mode":if cfg!(debug_assertions){"debug"}else{"release"},"rustc":std::process::Command::new("rustc").arg("--version").output().ok().map(|o|String::from_utf8_lossy(&o.stdout).trim().to_owned()),"peak_rss_bytes":null,"peak_rss_note":"not measured"})
}
fn dice(a: &str, b: &str) -> f64 {
    let grams = |s: &str| {
        s.chars()
            .collect::<Vec<_>>()
            .windows(2)
            .map(|w| w.iter().collect::<String>())
            .collect::<BTreeSet<_>>()
    };
    let x = grams(a);
    let y = grams(b);
    if x.len() + y.len() == 0 {
        0.0
    } else {
        2.0 * x.intersection(&y).count() as f64 / (x.len() + y.len()) as f64
    }
}
fn main() -> Result<()> {
    match Args::parse().command {
        Commands::Generate {
            model,
            fallback_only,
            difficulty,
            min_kana,
            max_kana,
            require_kanji,
            temperature,
            top_p,
            max_attempts,
            count,
            seed,
            history_size,
            unconstrained,
            format: _,
        } => {
            ensure!((1..=10000).contains(&count), "count must be 1..10000");
            let mut g = if fallback_only {
                TypingGenerator::fallback_only()?
            } else {
                TypingGenerator::load(model.context("model required")?)?
            };
            g.set_history_size(history_size);
            g.set_constraints_enabled(!unconstrained);
            let levels: Vec<_> = difficulty
                .difficulties()
                .iter()
                .copied()
                .filter(|d| {
                    let (lo, hi) = d.bounds();
                    lo.max(min_kana) <= hi.min(max_kana)
                })
                .collect();
            ensure!(
                !levels.is_empty(),
                "requested kana range does not intersect difficulty range"
            );
            let mut o = GenerateOptions {
                difficulty: levels[0],
                min_kana_chars: min_kana,
                max_kana_chars: max_kana,
                require_kanji,
                temperature: temperature.unwrap_or(g.default_options().temperature),
                top_p: top_p.unwrap_or(g.default_options().top_p),
                max_attempts,
                seed,
            };
            for i in 0..count {
                o.difficulty = levels[i % levels.len()];
                let p = g.generate(&o)?;
                println!("{}", serde_json::to_string(&p)?);
            }
            Ok(())
        }
        Commands::Logits { model, ids } => {
            let g = TypingGenerator::load(model)?;
            println!("{}", serde_json::to_string(&g.logits(&ids)?)?);
            Ok(())
        }
        Commands::Parity { model, tolerance } => {
            ensure!(
                tolerance.is_finite() && tolerance > 0.0,
                "invalid tolerance"
            );
            let cases: Vec<Value> = serde_json::from_slice(&fs::read(model.join("parity.json"))?)?;
            ensure!(!cases.is_empty(), "empty parity fixtures");
            let g = TypingGenerator::load(&model)?;
            let mut max_error = 0.0f32;
            let mut count = 0;
            for case in &cases {
                let ids: Vec<usize> = serde_json::from_value(case["ids"].clone())?;
                let expected: Vec<Vec<f32>> = serde_json::from_value(case["logits"].clone())?;
                let actual = g.logits(&ids)?;
                ensure!(
                    expected.len() == actual.len(),
                    "parity sequence dimension mismatch"
                );
                for (a, b) in actual.iter().zip(&expected) {
                    ensure!(a.len() == b.len(), "parity vocabulary dimension mismatch");
                    for (x, y) in a.iter().zip(b) {
                        ensure!(x.is_finite() && y.is_finite(), "nonfinite parity logits");
                        max_error = max_error.max((x - y).abs());
                        count += 1;
                    }
                }
            }
            println!(
                "{}",
                json!({"cases":cases.len(),"logits_compared":count,"max_absolute_error":max_error,"tolerance":tolerance,"passed":max_error<=tolerance})
            );
            ensure!(max_error <= tolerance, "Python/Rust logits parity failed");
            Ok(())
        }
        Commands::Evaluate {
            model,
            samples,
            report,
            training_data,
            seed,
            unconstrained,
            history_size,
            temperature,
            top_p,
            skip_near_statistics,
        } => {
            ensure!((1..=100000).contains(&samples), "samples must be 1..100000");
            let mut g = TypingGenerator::load(model)?;
            g.set_history_size(history_size);
            g.set_constraints_enabled(!unconstrained);
            let training = fs::read_to_string(training_data).ok().map(|s| {
                s.lines()
                    .filter_map(|l| serde_json::from_str::<Value>(l).ok())
                    .filter_map(|r| {
                        r["text"]
                            .as_str()
                            .map(|s| s.trim_end_matches('。').to_owned())
                    })
                    .collect::<BTreeSet<_>>()
            });
            let (mut outputs, mut failures) = (Vec::new(), Vec::new());
            let (mut attempts, mut rejected, mut model_n, mut fallback_n, mut exact, mut near) =
                (0usize, 0usize, 0usize, 0usize, 0usize, 0usize);
            let mut unique = BTreeSet::new();
            let mut frequency = BTreeMap::new();
            let mut distributions = BTreeMap::new();
            for i in 0..samples {
                let d = [Difficulty::Easy, Difficulty::Normal, Difficulty::Hard][i % 3];
                let o = GenerateOptions {
                    difficulty: d,
                    seed: Some(seed),
                    temperature,
                    top_p,
                    ..Default::default()
                };
                let result = g.generate(&o);
                attempts += g.last_stats.attempts;
                rejected += g.last_stats.rejected;
                match result {
                    Ok(p) => {
                        p.validate(&o)?;
                        if p.source.is_model() {
                            model_n += 1;
                            if let Some(t) = &training {
                                exact += usize::from(t.contains(&p.text));
                                if !skip_near_statistics {
                                    near += usize::from(t.iter().any(|x| dice(x, &p.text) >= 0.8));
                                }
                            }
                            for s in &p.segments {
                                *frequency
                                    .entry(format!("{}|{}|{}", s.surface, s.kana, s.pos))
                                    .or_insert(0usize) += 1;
                            }
                        } else {
                            fallback_n += 1;
                        }
                        unique.insert(p.text.clone());
                        let key = format!("{:?}/{:?}", d, p.source);
                        let values = distributions.entry(key).or_insert(Vec::new());
                        values.push(p.kana.chars().count());
                        outputs.push(p);
                    }
                    Err(e) => failures.push(json!({"index":i,"error":e.to_string()})),
                }
            }
            let success = outputs.len();
            let r = json!({"history_size":history_size,"requested":samples,"success":success,"model_count":model_n,"fallback_count":fallback_n,"failures":failures,"attempts":attempts,"rejected":rejected,"rejection_rate":if attempts>0{Some(rejected as f64/attempts as f64)}else{None},"fallback_rate":if success>0{Some(fallback_n as f64/success as f64)}else{None},"unique_texts":unique.len(),"duplicate_rate":if success>0{Some(1.0-unique.len() as f64/success as f64)}else{None},"model_training_exact_rate":training.as_ref().and_then(|_|if model_n>0{Some(exact as f64/model_n as f64)}else{None}),"model_training_near_rate":training.as_ref().and_then(|_|if !skip_near_statistics && model_n>0{Some(near as f64/model_n as f64)}else{None}),"model_token_frequency":frequency,"kana_length_distributions":distributions,"samples":outputs,"human_quality":"not evaluated; structural checks do not guarantee naturalness/contextual reading","environment":env_info()});
            write_report(&report, &r)
        }
        Commands::Bench {
            model,
            samples,
            report,
        } => {
            ensure!((1..=100000).contains(&samples), "samples must be 1..100000");
            let start = Instant::now();
            let mut g = TypingGenerator::load(&model)?;
            let load_ms = start.elapsed().as_secs_f64() * 1000.0;
            g.set_history_size(0);
            let mut durations = Vec::new();
            let (mut tokens, mut retries, mut models, mut fallback, mut failures) =
                (0usize, 0usize, 0usize, 0usize, 0usize);
            for _ in 0..samples {
                let start = Instant::now();
                let result = g.generate(&GenerateOptions {
                    seed: Some(42),
                    ..g.default_options()
                });
                durations.push(start.elapsed().as_secs_f64() * 1000.0);
                tokens += g.last_stats.model_tokens;
                retries += g.last_stats.rejected;
                match result {
                    Ok(p) => {
                        if p.source.is_model() {
                            models += 1
                        } else {
                            fallback += 1
                        }
                    }
                    Err(_) => failures += 1,
                }
            }
            let files = fs::read_dir(model)?
                .filter_map(|e| e.ok())
                .filter_map(|e| {
                    e.metadata()
                        .ok()
                        .filter(|m| m.is_file())
                        .map(|m| (e.file_name().to_string_lossy().into_owned(), m.len()))
                })
                .collect::<BTreeMap<_, _>>();
            write_report(
                &report,
                &json!({"samples":samples,"model_count":models,"fallback_count":fallback,"failures":failures,"cold_load_ms":load_ms,"first_generation_ms":durations.first(),"warm_p50_ms":percentile(&durations[1..],0.5),"warm_p95_ms":percentile(&durations[1..],0.95),"forward_sampled_tokens":tokens,"forward_sampled_tokens_per_second":tokens as f64/(durations.iter().sum::<f64>()/1000.0),"rejected_attempts":retries,"files_bytes":files,"binary_bytes":std::env::current_exe().ok().and_then(|p|fs::metadata(p).ok()).map(|m|m.len()),"environment":env_info()}),
            )
        }
    }
}
