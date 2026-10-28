use anyhow::{ensure, Context, Result};
use clap::{Parser, Subcommand};
use fumiori_core::*;
use lindera::{dictionary::load_dictionary_from_path, segmenter::Segmenter};
use lindera_dictionary::{
    dictionary_builder::{
        CharacterDefinitionBuilderOptions, ConnectionCostMatrixBuilderOptions,
        PrefixDictionaryBuilderOptions, UnknownDictionaryBuilderOptions,
    },
    mode::Mode,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    io::{BufRead, BufReader, Read},
    path::{Path, PathBuf},
};
#[derive(Parser)]
struct Args {
    #[command(subcommand)]
    command: Commands,
}
#[derive(Subcommand)]
enum Commands {
    /// Build only from existing UTF-8 dictionary sources. No network access.
    BuildDictionary {
        #[arg(long)]
        input_dir: PathBuf,
        #[arg(long)]
        out_dir: PathBuf,
    },
    Prepare {
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        out_dir: PathBuf,
        #[arg(long)]
        allow_auto_aligned: bool,
        #[arg(long)]
        allow_curated_examples: bool,
        #[arg(long)]
        dictionary: Option<PathBuf>,
        #[arg(long)]
        dictionary_version: Option<String>,
        #[arg(long, default_value = "16000")]
        max_vocab: usize,
    },
}
#[derive(Clone, Deserialize, Serialize)]
struct Row {
    id: String,
    text: String,
    kana: String,
    source: String,
    license: String,
    #[serde(default)]
    segments: Option<Vec<Segment>>,
    #[serde(default)]
    review: Value,
    #[serde(default)]
    generation: Value,
}
#[derive(Clone)]
struct Accepted {
    row: Row,
    trust: String,
}
fn lindera_segments(value: &Value) -> Result<Vec<Segment>> {
    let tokens = value
        .as_array()
        .context("Lindera JSON must be IPADIC token array")?;
    tokens
        .iter()
        .map(|t| {
            let surface = t["text"]
                .as_str()
                .context("Lindera text missing")?
                .to_owned();
            let d = t["details"].as_array().context("Lindera details missing")?;
            let pos = d
                .first()
                .and_then(Value::as_str)
                .context("Lindera POS missing")?
                .to_owned();
            let kana = if punctuation(&surface) {
                String::new()
            } else if pos == "助詞" && ["は", "へ", "を"].contains(&surface.as_str()) {
                surface.clone()
            } else {
                let reading = d
                    .get(7)
                    .and_then(Value::as_str)
                    .context("IPADIC reading missing")?;
                ensure!(reading != "*", "unknown reading");
                reading
                    .chars()
                    .map(|c| {
                        if ('ァ'..='ヶ').contains(&c) {
                            char::from_u32(c as u32 - 0x60).unwrap()
                        } else {
                            c
                        }
                    })
                    .collect()
            };
            let s = Segment { surface, kana, pos };
            validate_segment(&s)?;
            Ok(s)
        })
        .collect()
}
fn build_dictionary(input: &Path, out: &Path) -> Result<()> {
    fs::create_dir_all(out)?;
    let chars = CharacterDefinitionBuilderOptions::default()
        .builder()?
        .build(input, out)?;
    UnknownDictionaryBuilderOptions::default()
        .builder()?
        .build(input, &chars, out)?;
    PrefixDictionaryBuilderOptions::default()
        .builder()?
        .build(input, out)?;
    ConnectionCostMatrixBuilderOptions::default()
        .builder()?
        .build(input, out)?;
    Ok(())
}
fn local_segmenter(path: &Path) -> Result<Segmenter> {
    Ok(Segmenter::new(
        Mode::Normal,
        load_dictionary_from_path(path)?,
        None,
    ))
}
fn check_row(row: &mut Row, analyzer: Option<&Segmenter>) -> Result<String> {
    ensure!(
        !row.id.trim().is_empty()
            && !row.source.trim().is_empty()
            && row.source.trim().to_lowercase() != "unknown",
        "missing provenance"
    );
    ensure!(
        row.license == "CC0-1.0"
            || (row.license == "public-domain"
                && matches!(
                    row.source.as_str(),
                    "traditional-expression" | "public-domain-classic"
                )),
        "license/source pair not allowlisted"
    );
    row.text = nfc(&row.text);
    row.kana = nfc(&row.kana);
    let supplied = row.segments.is_some();
    if !supplied {
        let analyzer =
            analyzer.context("segments missing; local Lindera dictionary not configured")?;
        let mut tokens = analyzer.segment(row.text.as_str().into())?;
        let values: Vec<_> = tokens
            .iter_mut()
            .map(|t| {
                let text = t.text.to_string();
                json!({"text": text, "details": t.details()})
            })
            .collect();
        row.segments = Some(lindera_segments(&json!(values))?);
    }
    let seg = row.segments.as_mut().unwrap();
    for s in seg.iter_mut() {
        s.surface = nfc(&s.surface);
        s.kana = nfc(&s.kana);
    }
    validate_pair(&row.text, &row.kana, seg)?;
    ensure!(seg.len() + 3 <= 64, "sequence exceeds 64 tokens");
    ensure!(row.kana.chars().count() >= 5, "too short");
    let human = row.review["status"] == "verified"
        && row.review["kind"] == "human"
        && row.review["reviewer"]
            .as_str()
            .is_some_and(|s| !s.trim().is_empty());
    Ok(if supplied && human {
        "verified"
    } else {
        "auto_aligned"
    }
    .into())
}
fn grams(s: &str) -> BTreeSet<String> {
    s.chars()
        .collect::<Vec<_>>()
        .windows(2)
        .map(|w| w.iter().collect())
        .collect()
}
#[cfg(test)]
fn similar(a: &Row, b: &Row) -> bool {
    if a.kana == b.kana {
        return true;
    }
    let x = grams(&a.text);
    let y = grams(&b.text);
    let n = x.len() + y.len();
    n > 0 && 2.0 * x.intersection(&y).count() as f64 / n as f64 >= 0.8
}
fn root(p: &mut [usize], i: usize) -> usize {
    if p[i] != i {
        p[i] = root(p, p[i]);
    }
    p[i]
}
fn write_json(path: impl AsRef<Path>, v: &impl Serialize) -> Result<()> {
    fs::write(path, serde_json::to_vec_pretty(v)?)?;
    Ok(())
}
#[allow(clippy::too_many_arguments)]
fn prepare(
    input: &Path,
    out: &Path,
    allow_auto: bool,
    allow_curated: bool,
    dictionary: Option<&Path>,
    dict: Option<&str>,
    max_vocab: usize,
) -> Result<()> {
    ensure!(
        (8..=16000).contains(&max_vocab),
        "max_vocab must be 8..16000"
    );
    ensure!(
        dictionary.is_none() || dict.is_some(),
        "--dictionary-version required for Lindera provenance"
    );
    let analyzer = dictionary.map(local_segmenter).transpose()?;
    ensure!(
        fs::metadata(input)?.len() <= 100 * 1024 * 1024,
        "input exceeds 100 MiB"
    );
    let bytes = fs::read(input)?;
    ensure!(bytes.len() <= 100 * 1024 * 1024, "input exceeds 100 MiB");
    let mut reader = BufReader::new(bytes.as_slice());
    let mut line = Vec::new();
    let mut accepted = Vec::new();
    let mut report = Vec::new();
    let mut seen = BTreeSet::new();
    let mut seen_ids = BTreeSet::new();
    let mut number = 0;
    loop {
        line.clear();
        let n = reader.by_ref().take(65537).read_until(b'\n', &mut line)?;
        if n == 0 {
            break;
        }
        number += 1;
        ensure!(n <= 65536, "line {number} exceeds 64 KiB");
        ensure!(
            number <= 8000,
            "PoC limit 8000 total rows (train and held-out) (quadratic similarity detection)"
        );
        let parsed = (|| -> Result<(Row, String)> {
            let mut r: Row = serde_json::from_slice(&line)?;
            ensure!(seen_ids.insert(r.id.clone()), "duplicate id");
            let t = check_row(&mut r, analyzer.as_ref())?;
            Ok((r, t))
        })();
        match parsed {
            Err(e) => {
                report.push(json!({"line":number,"status":"rejected","reason":e.to_string()}))
            }
            Ok((r, t)) => {
                let curated = allow_curated
                    && matches!(
                        r.source.as_str(),
                        "original-agent-authored"
                            | "traditional-expression"
                            | "public-domain-classic"
                    )
                    && r.review["status"] == "agent_curated";
                if t != "verified" && !allow_auto && !curated {
                    report.push(json!({"line":number,"id":r.id,"status":t,"included":false,"reason":"human review required or explicit experiment flag"}));
                    continue;
                }
                if !seen.insert((r.text.clone(), r.kana.clone())) {
                    report.push(json!({"line":number,"id":r.id,"status":"rejected","reason":"exact duplicate"}));
                    continue;
                }
                accepted.push(Accepted { row: r, trust: t });
            }
        }
    }
    accepted.sort_by(|a, b| {
        (&a.row.text, &a.row.kana, &a.row.id).cmp(&(&b.row.text, &b.row.kana, &b.row.id))
    });
    let triples: BTreeSet<_> = accepted
        .iter()
        .flat_map(|a| {
            a.row
                .segments
                .as_ref()
                .unwrap()
                .iter()
                .map(|s| (s.surface.clone(), s.kana.clone(), s.pos.clone()))
        })
        .collect();
    let mut tokens: Vec<Token> = SPECIALS
        .iter()
        .enumerate()
        .map(|(id, s)| Token {
            id,
            surface: s.to_string(),
            kana: String::new(),
            pos: "special".into(),
            metadata: json!({}),
        })
        .collect();
    let mut map = BTreeMap::new();
    for key in triples.into_iter().take(max_vocab - 7) {
        let id = tokens.len();
        map.insert(key.clone(), id);
        tokens.push(Token {
            id,
            surface: key.0,
            kana: key.1,
            pos: key.2,
            metadata: json!({}),
        });
    }
    accepted.retain(|a|{let ok=a.row.segments.as_ref().unwrap().iter().all(|s|map.contains_key(&(s.surface.clone(),s.kana.clone(),s.pos.clone())));if !ok{report.push(json!({"id":a.row.id,"status":"rejected","reason":"vocabulary pruning would require UNK"}));}ok});
    fs::create_dir_all(out)?;
    write_json(out.join("rejections.json"), &report)?;
    ensure!(!accepted.is_empty(),"no accepted rows; see rejections.json; examples require --allow-curated-examples (not human-reviewed)");
    let vocab = Vocab {
        version: VERSION.into(),
        encoding: "UTF-8".into(),
        tokens,
    };
    vocab.validate()?;
    write_json(out.join("vocab.json"), &vocab)?;
    let vocab_hash = sha256(&fs::read(out.join("vocab.json"))?);
    let mut parent: Vec<_> = (0..accepted.len()).collect();
    // All expansions of a declared template stay together, even if lexical
    // substitutions put their character similarity below the threshold.
    let mut family_roots = BTreeMap::new();
    for (i, a) in accepted.iter().enumerate() {
        if let Some(key) = a.row.generation["group_key"].as_str() {
            ensure!(!key.is_empty(), "empty generation group key");
            if let Some(&first) = family_roots.get(key) {
                parent[i] = first;
            } else {
                family_roots.insert(key.to_owned(), i);
            }
        }
    }
    // Cache bigrams once instead of rebuilding two sets for every pair.
    let cached: Vec<_> = accepted.iter().map(|a| grams(&a.row.text)).collect();
    for i in 0..accepted.len() {
        for j in 0..i {
            let a = root(&mut parent, i);
            let b = root(&mut parent, j);
            if a == b {
                continue;
            }
            let (x, y) = (&cached[i], &cached[j]);
            let similar = accepted[i].row.kana == accepted[j].row.kana
                || (!x.is_empty()
                    && !y.is_empty()
                    && 2 * x.len().min(y.len()) * 5 >= (x.len() + y.len()) * 4
                    && 2 * x.intersection(y).count() * 5 >= (x.len() + y.len()) * 4);
            if similar {
                parent[a] = b;
            }
        }
    }
    let mut splits: BTreeMap<&str, Vec<Value>> =
        BTreeMap::from([("train", vec![]), ("validation", vec![]), ("test", vec![])]);
    let mut groups = BTreeMap::new();
    let mut counts = BTreeMap::new();
    for (i, a) in accepted.iter().enumerate() {
        let g = root(&mut parent, i);
        let next = groups.len();
        let group = *groups.entry(g).or_insert(next);
        let split = match group % 10 {
            0 => "validation",
            1 => "test",
            _ => "train",
        };
        *counts.entry(a.trust.clone()).or_insert(0usize) += 1;
        let r = &a.row;
        let difficulty = Difficulty::from_len(r.kana.chars().count());
        let mut ids = vec![0, difficulty.id()];
        ids.extend(
            r.segments
                .as_ref()
                .unwrap()
                .iter()
                .map(|s| map[&(s.surface.clone(), s.kana.clone(), s.pos.clone())]),
        );
        ids.push(1);
        ensure!(ids.len() <= 64, "sequence exceeds 64 tokens");
        let restored = vocab.decode(&ids[2..ids.len() - 1])?;
        validate_pair(&r.text, &r.kana, &restored)?;
        splits.get_mut(split).unwrap().push(json!({"id":r.id,"text":r.text,"kana":r.kana,"segments":r.segments,"ids":ids,"trust":a.trust,"review":r.review,"generation":r.generation,"group":group,"difficulty":difficulty,"source":r.source,"license":r.license}));
    }
    for (split, rows) in &splits {
        fs::write(
            out.join(format!("{split}.jsonl")),
            rows.iter()
                .map(|r| format!("{}\n", serde_json::to_string(r).unwrap()))
                .collect::<String>(),
        )?;
    }
    let split_hashes: BTreeMap<_, _> = ["train", "validation", "test"]
        .iter()
        .map(|s| {
            Ok((
                format!("{s}.jsonl"),
                sha256(&fs::read(out.join(format!("{s}.jsonl")))?),
            ))
        })
        .collect::<Result<_>>()?;
    let dictionary_hashes: Option<BTreeMap<_, _>> = dictionary
        .map(|d| -> Result<_> {
            fs::read_dir(d)?
                .map(|e| -> Result<_> {
                    let e = e?;
                    Ok((
                        e.file_name().to_string_lossy().into_owned(),
                        sha256(&fs::read(e.path())?),
                    ))
                })
                .collect()
        })
        .transpose()?;
    write_json(
        out.join("manifest.json"),
        &json!({"version":VERSION,"input":input,"input_sha256":sha256(&bytes),"split_sha256":split_hashes,"dictionary_sha256":dictionary_hashes,"input_ids":accepted.iter().map(|a|&a.row.id).collect::<Vec<_>>(),"licenses":accepted.iter().map(|a|a.row.license.clone()).collect::<BTreeSet<_>>(),"dictionary_version":dict.unwrap_or("none: explicit segments"),"code_version":VERSION,"converted_unix_seconds":std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_secs(),"vocab_sha256":vocab_hash,"trust_counts":counts,"auto_aligned_fraction":*counts.get("auto_aligned").unwrap_or(&0) as f64/accepted.len() as f64,"allow_auto_aligned":allow_auto,"allow_curated_examples":allow_curated,"splits":splits.iter().map(|(s,r)|(*s,r.len())).collect::<BTreeMap<_,_>>(),"near_duplicate_threshold":0.8,"vocab_scope":"all accepted splits; closed-vocabulary PoC"}),
    )?;
    println!(
        "prepared {} rows, {} tokens; trust counts {:?}",
        accepted.len(),
        vocab.tokens.len(),
        counts
    );
    Ok(())
}
fn main() -> Result<()> {
    match Args::parse().command {
        Commands::BuildDictionary { input_dir, out_dir } => build_dictionary(&input_dir, &out_dir),
        Commands::Prepare {
            input,
            out_dir,
            allow_auto_aligned,
            allow_curated_examples,
            dictionary,
            dictionary_version,
            max_vocab,
        } => prepare(
            &input,
            &out_dir,
            allow_auto_aligned,
            allow_curated_examples,
            dictionary.as_deref(),
            dictionary_version.as_deref(),
            max_vocab,
        ),
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    fn example() -> Row {
        serde_json::from_str(
            include_str!("../../../data/examples/sentences.jsonl")
                .lines()
                .next()
                .unwrap(),
        )
        .unwrap()
    }
    #[test]
    fn trust_is_not_inferred() {
        let mut r = example();
        assert_eq!(check_row(&mut r, None).unwrap(), "auto_aligned");
        r.review = json!({"status":"verified","kind":"human","reviewer":"fixture-only"});
        assert_eq!(check_row(&mut r, None).unwrap(), "verified");
        r.segments = None;
        assert!(check_row(&mut r, None).is_err());
    }
    #[test]
    fn ambiguous_ids() {
        let mut x = example();
        let y = x.clone();
        assert!(similar(&x, &y));
        x.license = "unknown".into();
        assert!(check_row(&mut x, None).is_err());
    }
    #[test]
    fn public_domain_requires_explicit_supported_provenance() {
        let mut row = example();
        row.license = "public-domain".into();
        assert!(check_row(&mut row, None).is_err());
        row.source = "public-domain-classic".into();
        assert!(check_row(&mut row, None).is_ok());
        row.source = "traditional-expression".into();
        assert!(check_row(&mut row, None).is_ok());
    }
    #[test]
    fn ipadic_adapter() {
        let tokens = json!([{"text":"は","details":["助詞","*","*","*","*","*","は","ワ"]},{"text":"猫","details":["名詞","*","*","*","*","*","猫","ネコ"]}]);
        let s = lindera_segments(&tokens).unwrap();
        assert_eq!(s[0].kana, "は");
        assert_eq!(s[1].kana, "ねこ");
        assert!(lindera_segments(&json!([{"text":"謎","details":["未知語"]}])).is_err());
    }
    #[test]
    fn real_lindera_local_dictionary_stays_unverified() -> Result<()> {
        let dir = std::env::temp_dir().join(format!("fumiori-lindera-test-{}", std::process::id()));
        let input = Path::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../data/examples/lindera-src"
        ));
        build_dictionary(input, &dir)?;
        let analyzer = local_segmenter(&dir)?;
        let mut r = example();
        r.segments = None;
        // Even a claimed human review does not promote newly inferred boundaries.
        r.review = json!({"status":"verified","kind":"human","reviewer":"test fixture"});
        assert_eq!(check_row(&mut r, Some(&analyzer))?, "auto_aligned");
        assert_eq!(r.segments.as_ref().unwrap()[0].kana, "ねこ");
        r.segments = None;
        r.kana = "いぬがねむる".into();
        assert!(check_row(&mut r, Some(&analyzer)).is_err());
        r.text = "未知の固有名詞".into();
        r.kana = "みちのこゆうめいし".into();
        r.segments = None;
        assert!(check_row(&mut r, Some(&analyzer)).is_err());
        Ok(())
    }
    #[test]
    fn unreviewed_default_and_pruning_reject() -> Result<()> {
        let base = std::env::temp_dir().join(format!("fumiori-reject-test-{}", std::process::id()));
        let input = Path::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../data/examples/sentences.jsonl"
        ));
        assert!(prepare(
            input,
            &base.join("default"),
            false,
            false,
            None,
            None,
            16000
        )
        .is_err());
        assert!(prepare(input, &base.join("pruned"), false, true, None, None, 8).is_err());
        let reasons = fs::read_to_string(base.join("pruned/rejections.json"))?;
        assert!(reasons.contains("UNK"));
        Ok(())
    }
    #[test]
    fn deterministic_roundtrip_split() -> Result<()> {
        let dir = std::env::temp_dir().join(format!("fumiori-data-test-{}", std::process::id()));
        let input = Path::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../data/examples/sentences.jsonl"
        ));
        prepare(input, &dir, false, true, None, None, 16000)?;
        let first = fs::read(dir.join("vocab.json"))?;
        prepare(input, &dir, false, true, None, None, 16000)?;
        assert_eq!(first, fs::read(dir.join("vocab.json"))?);
        let mut groups = BTreeMap::new();
        for split in ["train", "validation", "test"] {
            for line in fs::read_to_string(dir.join(format!("{split}.jsonl")))?.lines() {
                let row: Value = serde_json::from_str(line)?;
                let g = row["group"].as_u64().unwrap();
                if let Some(old) = groups.insert(g, split) {
                    assert_eq!(old, split);
                }
            }
        }
        Ok(())
    }
}
