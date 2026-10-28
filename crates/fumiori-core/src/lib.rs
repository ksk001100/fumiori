use anyhow::{bail, ensure, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use unicode_normalization::UnicodeNormalization;

pub const SPECIALS: [&str; 7] = ["BOS", "EOS", "PAD", "EASY", "NORMAL", "HARD", "UNK"];
pub const VERSION: &str = "0.1.0";
pub fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
pub fn nfc(s: &str) -> String {
    s.nfc().collect()
}
pub fn is_kana(c: char) -> bool {
    ('ぁ'..='ゖ').contains(&c) || c == 'ー'
}
pub fn is_typing_kana(c: char) -> bool {
    is_kana(c) || c == '、'
}
pub fn is_kanji(c: char) -> bool {
    ('一'..='鿿').contains(&c) || c == '々'
}
pub fn punctuation(s: &str) -> bool {
    !s.is_empty() && s.chars().all(|c| "。、！？「」『』（）　 ".contains(c))
}
pub fn allowed_text(c: char) -> bool {
    is_kana(c) || is_kanji(c) || ('ァ'..='ヺ').contains(&c) || "。、！？「」『』（）　 ".contains(c)
}
pub const BANNED: [&str; 4] = ["死ね", "殺す", "馬鹿", "ばか"];

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Segment {
    pub surface: String,
    pub kana: String,
    #[serde(default = "default_pos")]
    pub pos: String,
}
fn default_pos() -> String {
    "unspecified".into()
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Token {
    pub id: usize,
    pub surface: String,
    pub kana: String,
    pub pos: String,
    #[serde(default)]
    pub metadata: serde_json::Value,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Vocab {
    pub version: String,
    pub encoding: String,
    pub tokens: Vec<Token>,
}
impl Vocab {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.version == VERSION && self.encoding == "UTF-8",
            "unsupported vocabulary version/encoding"
        );
        ensure!(
            self.tokens.len() >= 8 && self.tokens.len() <= 16000,
            "invalid vocabulary size"
        );
        let mut unique = HashSet::new();
        for (i, t) in self.tokens.iter().enumerate() {
            ensure!(t.id == i, "non-contiguous token IDs");
            if i < 7 {
                ensure!(
                    t.surface == SPECIALS[i] && t.kana.is_empty() && t.pos == "special",
                    "invalid special token"
                );
            } else {
                validate_segment(&Segment {
                    surface: t.surface.clone(),
                    kana: t.kana.clone(),
                    pos: t.pos.clone(),
                })?;
                ensure!(
                    unique.insert((&t.surface, &t.kana, &t.pos)),
                    "duplicate token"
                );
            }
        }
        Ok(())
    }
    pub fn decode(&self, ids: &[usize]) -> Result<Vec<Segment>> {
        ids.iter()
            .map(|&id| {
                ensure!(id >= 7, "reserved token in output");
                let t = self
                    .tokens
                    .get(id)
                    .ok_or_else(|| anyhow::anyhow!("unknown token ID {id}"))?;
                Ok(Segment {
                    surface: t.surface.clone(),
                    kana: t.kana.clone(),
                    pos: t.pos.clone(),
                })
            })
            .collect()
    }
}
pub fn validate_segment(s: &Segment) -> Result<()> {
    ensure!(
        !s.surface.is_empty() && s.surface.chars().count() <= 64,
        "empty/long surface"
    );
    ensure!(
        s.surface.chars().all(allowed_text) && s.surface == nfc(&s.surface),
        "invalid text/NFC"
    );
    ensure!(
        s.kana == nfc(&s.kana) && s.kana.chars().all(is_typing_kana),
        "invalid kana/NFC"
    );
    ensure!(
        !s.kana.is_empty() || punctuation(&s.surface),
        "missing reading"
    );
    ensure!(
        !punctuation(&s.surface) || s.kana.is_empty() || (s.surface == "、" && s.kana == "、"),
        "punctuation must have empty reading except a typed comma"
    );
    ensure!(
        s.surface.matches('、').count() == s.kana.matches('、').count()
            || (s.surface == "、" && s.kana.is_empty()),
        "comma mismatch"
    );
    ensure!(!s.pos.is_empty(), "missing pos");
    if ["は", "へ", "を"].contains(&s.surface.as_str()) && s.pos == "助詞" {
        ensure!(s.surface == s.kana, "particle orthography mismatch");
    }
    Ok(())
}
pub fn validate_pair(text: &str, kana: &str, segments: &[Segment]) -> Result<()> {
    ensure!(
        !text.is_empty() && text.chars().count() <= 256,
        "empty/long text"
    );
    ensure!(
        !kana.is_empty() && kana.chars().count() <= 40 && kana.chars().all(is_typing_kana),
        "invalid kana"
    );
    ensure!(!segments.is_empty(), "empty segments");
    for s in segments {
        validate_segment(s)?;
    }
    ensure!(
        segments
            .iter()
            .map(|s| s.surface.as_str())
            .collect::<String>()
            == text,
        "surface concatenation mismatch"
    );
    ensure!(
        segments.iter().map(|s| s.kana.as_str()).collect::<String>() == kana,
        "kana concatenation mismatch"
    );
    ensure!(
        !BANNED.iter().any(|w| text.contains(w)
            || (if *w == "ばか" {
                segments.iter().any(|s| s.kana.starts_with(w))
            } else {
                kana.contains(w)
            })),
        "banned phrase"
    );
    ensure!(
        !text.contains("。。") && !text.contains("、、"),
        "repeated punctuation"
    );
    let mut brackets = Vec::new();
    for c in text.chars() {
        match c {
            '「' => brackets.push('」'),
            '『' => brackets.push('』'),
            '（' => brackets.push('）'),
            '」' | '』' | '）' => ensure!(brackets.pop() == Some(c), "mismatched brackets"),
            _ => {}
        }
    }
    ensure!(brackets.is_empty(), "unclosed brackets");
    ensure!(
        !text.starts_with('、') && !text.contains("、。"),
        "misplaced comma"
    );
    for w in segments.windows(3) {
        ensure!(!(w[0] == w[1] && w[1] == w[2]), "repeated segment");
    }
    Ok(())
}
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Difficulty {
    Easy,
    Normal,
    Hard,
}
impl Difficulty {
    pub fn bounds(self) -> (usize, usize) {
        match self {
            Self::Easy => (5, 12),
            Self::Normal => (13, 22),
            Self::Hard => (23, 40),
        }
    }
    pub fn id(self) -> usize {
        match self {
            Self::Easy => 3,
            Self::Normal => 4,
            Self::Hard => 5,
        }
    }
    pub fn from_len(n: usize) -> Self {
        if n <= 12 {
            Self::Easy
        } else if n <= 22 {
            Self::Normal
        } else {
            Self::Hard
        }
    }
}
#[derive(Debug, Clone)]
pub struct GenerateOptions {
    pub difficulty: Difficulty,
    pub min_kana_chars: usize,
    pub max_kana_chars: usize,
    pub require_kanji: bool,
    pub temperature: f32,
    pub top_p: f32,
    pub max_attempts: usize,
    pub seed: Option<u64>,
}
impl Default for GenerateOptions {
    fn default() -> Self {
        Self {
            difficulty: Difficulty::Normal,
            min_kana_chars: 5,
            max_kana_chars: 40,
            require_kanji: false,
            temperature: 0.8,
            top_p: 0.9,
            max_attempts: 20,
            seed: None,
        }
    }
}
impl GenerateOptions {
    pub fn bounds(&self) -> Result<(usize, usize)> {
        ensure!(
            self.temperature.is_finite()
                && self.temperature > 0.0
                && self.top_p.is_finite()
                && self.top_p > 0.0
                && self.top_p <= 1.0,
            "invalid sampling parameters"
        );
        ensure!(self.max_attempts <= 1000, "max_attempts exceeds 1000");
        let (lo, hi) = self.difficulty.bounds();
        let range = (lo.max(self.min_kana_chars), hi.min(self.max_kana_chars));
        ensure!(
            range.0 <= range.1,
            "requested kana range does not intersect difficulty range"
        );
        Ok(range)
    }
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum PromptSource {
    Model,
    #[serde(rename = "model_constrained")]
    ModelConstrained,
    Fallback,
}
impl PromptSource {
    pub fn is_model(&self) -> bool {
        matches!(self, Self::Model | Self::ModelConstrained)
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypingPrompt {
    pub text: String,
    pub kana: String,
    pub segments: Vec<Segment>,
    pub source: PromptSource,
    pub model_version: String,
    pub reading_validation: String,
}
impl TypingPrompt {
    /// Omit the display-only sentence mark while preserving segment roundtrips.
    pub fn without_terminal_period(mut self) -> Self {
        if let Some(last) = self.segments.last_mut() {
            if last.surface.ends_with('。') {
                last.surface.pop();
                if last.surface.is_empty() && last.kana.is_empty() {
                    self.segments.pop();
                }
                self.text = self.segments.iter().map(|s| s.surface.as_str()).collect();
            }
        }
        self
    }
    pub fn from_segments(mut segments: Vec<Segment>, source: PromptSource) -> Self {
        // Older model vocabularies store commas with an empty reading.
        // Public prompts include them as typing input without changing token IDs.
        for segment in &mut segments {
            if segment.surface == "、" && segment.kana.is_empty() {
                segment.kana = "、".into();
            }
        }
        let reading_validation = match source {
            PromptSource::Model => "structural_only; contextual_reading_not_verified",
            PromptSource::ModelConstrained => {
                "authored_frame_constraints; contextual_reading_not_verified"
            }
            PromptSource::Fallback => "agent_curated; independent_human_review_pending",
        }
        .to_string();
        Self {
            text: segments.iter().map(|s| s.surface.as_str()).collect(),
            kana: segments.iter().map(|s| s.kana.as_str()).collect(),
            segments,
            source,
            model_version: VERSION.into(),
            reading_validation,
        }
    }
    pub fn validate(&self, o: &GenerateOptions) -> Result<()> {
        validate_pair(&self.text, &self.kana, &self.segments)?;
        let (lo, hi) = o.bounds()?;
        ensure!(
            (lo..=hi).contains(&self.kana.chars().count()),
            "kana length out of range"
        );
        ensure!(
            !o.require_kanji || self.text.chars().any(is_kanji),
            "kanji required"
        );
        let has_period = self.text.ends_with('。');
        ensure!(
            self.text.matches('。').count() == usize::from(has_period),
            "incomplete/multiple sentence"
        );
        if self.segments.len() <= usize::from(has_period) {
            bail!("incomplete sentence");
        }
        let end = &self.segments[self.segments.len() - 1 - usize::from(has_period)];
        ensure!(
            !["助詞", "名詞"].contains(&end.pos.as_str()),
            "incomplete ending"
        );
        for w in self.segments.windows(2) {
            ensure!(
                !(w[0].pos == "助詞" && w[1].pos == "助詞"),
                "consecutive particles"
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn s(surface: &str, kana: &str, pos: &str) -> Segment {
        Segment {
            surface: surface.into(),
            kana: kana.into(),
            pos: pos.into(),
        }
    }
    #[test]
    fn legitimate_kana_crossings_do_not_trigger_insult_filter() {
        for segments in [
            vec![s("乗り場", "のりば", "名詞"), s("から", "から", "助詞")],
            vec![s(
                "待てば海路の日和あり",
                "まてばかいろのひよりあり",
                "成句",
            )],
        ] {
            let prompt = TypingPrompt::from_segments(segments, PromptSource::Model);
            assert!(validate_pair(&prompt.text, &prompt.kana, &prompt.segments).is_ok());
        }
        for segment in [s("ばか", "ばか", "名詞"), s("馬鹿", "ばか", "名詞")] {
            assert!(validate_pair(
                &segment.surface,
                &segment.kana,
                std::slice::from_ref(&segment)
            )
            .is_err());
        }
    }
    #[test]
    fn brackets_and_commas() {
        for text in ["「猫。", "猫」。", "（猫」。", "、猫。", "猫、。"] {
            assert!(validate_pair(text, "ねこ", &[s(text, "ねこ", "語")]).is_err());
        }
        assert!(validate_pair("「猫」。", "ねこ", &[s("「猫」。", "ねこ", "語")]).is_ok());
    }
    #[test]
    fn typing_commas_preserve_roundtrips_and_count_toward_difficulty() {
        let prompt = TypingPrompt::from_segments(
            vec![
                s("猫", "ねこ", "名詞"),
                s("、", "", "記号"),
                s("眠る", "ねむる", "動詞"),
                s("。", "", "記号"),
            ],
            PromptSource::Model,
        )
        .without_terminal_period();
        assert_eq!(prompt.text, "猫、眠る");
        assert_eq!(prompt.kana, "ねこ、ねむる");
        assert_eq!(prompt.segments[1].kana, "、");
        let mut options = GenerateOptions {
            difficulty: Difficulty::Easy,
            ..Default::default()
        };
        prompt.validate(&options).unwrap();
        options.max_kana_chars = 5;
        assert!(prompt.validate(&options).is_err());
        assert!(validate_segment(&s("猫", "ね、こ", "名詞")).is_err());
        assert!(validate_segment(&s("、", "ねこ", "記号")).is_err());
        // Existing model vocabulary remains readable.
        assert!(validate_segment(&s("、", "", "記号")).is_ok());
        assert!(validate_segment(&s("、", "、", "記号")).is_ok());
    }
    #[test]
    fn particles_and_readings() {
        for p in ["は", "へ", "を"] {
            assert!(validate_segment(&s(p, p, "助詞")).is_ok());
        }
        assert!(validate_segment(&s("は", "わ", "助詞")).is_err());
        for (a, b) in [
            ("今日", "きょう"),
            ("今日", "こんにち"),
            ("人気", "にんき"),
            ("人気", "ひとけ"),
            ("走った", "はしった"),
            ("良い", "いい"),
        ] {
            assert!(validate_segment(&s(a, b, "語")).is_ok());
        }
    }
    #[test]
    fn invalid_and_concat() {
        for (a, b) in [
            ("猫", "ネコ"),
            ("猫", ""),
            ("", "ねこ"),
            ("猫😀", "ねこ"),
            ("。", "まる"),
        ] {
            assert!(validate_segment(&s(a, b, "語")).is_err());
        }
        let seg = vec![s("猫", "ねこ", "名詞"), s("。", "", "記号")];
        assert!(validate_pair("猫。", "ねこ", &seg).is_ok());
        assert!(validate_pair("犬。", "ねこ", &seg).is_err());
        assert!(validate_pair("猫。", "いぬ", &seg).is_err());
    }
    #[test]
    fn bounds() {
        let mut o = GenerateOptions {
            min_kana_chars: 30,
            ..Default::default()
        };
        assert!(o.bounds().is_err());
        o.temperature = f32::NAN;
        assert!(o.bounds().is_err());
    }
    #[test]
    fn display_period_removal_preserves_reading_and_checks_endings() {
        let original = TypingPrompt::from_segments(
            vec![
                s("猫", "ねこ", "名詞"),
                s("が", "が", "助詞"),
                s("眠る", "ねむる", "動詞"),
                s("。", "", "記号"),
            ],
            PromptSource::Model,
        );
        let options = GenerateOptions {
            difficulty: Difficulty::Easy,
            ..Default::default()
        };
        original.validate(&options).unwrap();
        let plain = original.without_terminal_period();
        plain.validate(&options).unwrap();
        assert_eq!(plain.text, "猫が眠る");
        assert_eq!(plain.kana, "ねこがねむる");
        assert_eq!(
            plain.text,
            plain
                .segments
                .iter()
                .map(|s| s.surface.as_str())
                .collect::<String>()
        );
        assert_eq!(plain.clone().without_terminal_period().text, plain.text);
        let fragment = TypingPrompt::from_segments(
            vec![s("友達", "ともだち", "名詞"), s("と", "と", "助詞")],
            PromptSource::Model,
        );
        assert!(fragment.validate(&options).is_err());
        let multiple = TypingPrompt::from_segments(
            vec![
                s("猫", "ねこ", "名詞"),
                s("。", "", "記号"),
                s("眠る", "ねむる", "動詞"),
            ],
            PromptSource::Model,
        );
        assert!(multiple.validate(&options).is_err());
    }
}
