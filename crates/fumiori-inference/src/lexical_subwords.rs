//! A word lexicon constrains subword spelling, without listing sentences.
use super::{can_end, lexical_collision, polite_collision};
use anyhow::{ensure, Context, Result};
use fumiori_core::{validate_segment, Segment, Token};
use serde::Deserialize;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Deserialize)]
pub(crate) struct Definition {
    pub encoding: String,
    pub tokens: Vec<Piece>,
    pub lexicon: Vec<Token>,
    #[serde(default)]
    pub reading_contexts: Vec<ReadingContext>,
}
#[derive(Deserialize)]
pub(crate) struct ReadingContext {
    previous: [String; 2],
    word: usize,
    count: u64,
}
#[derive(Deserialize)]
pub(crate) struct Piece {
    pub id: usize,
    pub surface: String,
}
#[derive(Default)]
struct Node {
    children: BTreeMap<char, usize>,
    words: Vec<usize>,
}
#[derive(Default)]
struct PieceNode {
    children: BTreeMap<char, usize>,
    ids: Vec<usize>,
    bare_any: bool,
}
pub(crate) struct Lexicon {
    pub pieces: Vec<String>,
    pub words: Vec<Token>,
    nodes: Vec<Node>,
    piece_nodes: Vec<PieceNode>,
    readings: BTreeMap<(String, String, String, String), BTreeMap<usize, u64>>,
}
#[derive(Clone)]
pub(crate) struct State {
    node: usize,
    path: Vec<usize>,
    kana: usize,
    score: f64,
    pub ended: bool,
}
impl State {
    pub fn initial() -> Self {
        Self {
            node: 0,
            path: vec![],
            kana: 0,
            score: 0.,
            ended: false,
        }
    }
}
impl Lexicon {
    pub fn load(definition: Definition) -> Result<Self> {
        ensure!(
            definition.encoding == "sentencepiece_lexicon_v1",
            "unknown lexical subword encoding"
        );
        ensure!(
            (8..=16000).contains(&definition.tokens.len()),
            "invalid subword vocabulary size"
        );
        ensure!(
            !definition.lexicon.is_empty() && definition.lexicon.len() <= 16000,
            "invalid word lexicon size"
        );
        let mut pieces = Vec::new();
        for (i, piece) in definition.tokens.into_iter().enumerate() {
            ensure!(
                piece.id == i && piece.surface.len() <= 512,
                "invalid piece ID/length"
            );
            pieces.push(piece.surface);
        }
        let mut piece_nodes = vec![PieceNode::default()];
        for (id, piece) in pieces.iter().enumerate().skip(7) {
            let bare = !piece.starts_with('▁');
            let mut node = 0;
            piece_nodes[node].bare_any |= bare;
            for c in piece.trim_start_matches('▁').chars() {
                node = if let Some(&next) = piece_nodes[node].children.get(&c) {
                    next
                } else {
                    let next = piece_nodes.len();
                    piece_nodes.push(PieceNode::default());
                    piece_nodes[node].children.insert(c, next);
                    next
                };
                piece_nodes[node].bare_any |= bare;
            }
            piece_nodes[node].ids.push(id);
        }
        let mut nodes = vec![Node::default()];
        let mut unique = BTreeSet::new();
        for (i, word) in definition.lexicon.iter().enumerate() {
            validate_segment(&Segment {
                surface: word.surface.clone(),
                kana: word.kana.clone(),
                pos: word.pos.clone(),
            })?;
            ensure!(
                word.pos != "文字"
                    && !word
                        .pos
                        .split('-')
                        .any(|p| matches!(p, "人名" | "固有名詞")),
                "forbidden lexical entry"
            );
            ensure!(
                unique.insert((&word.surface, &word.kana, &word.pos)),
                "duplicate lexical entry"
            );
            let mut node = 0;
            for c in word.surface.chars() {
                node = if let Some(&next) = nodes[node].children.get(&c) {
                    next
                } else {
                    let next = nodes.len();
                    nodes.push(Node::default());
                    nodes[node].children.insert(c, next);
                    next
                };
            }
            nodes[node].words.push(i);
        }
        let mut readings: BTreeMap<_, BTreeMap<_, _>> = BTreeMap::new();
        for context in definition.reading_contexts {
            let word = definition
                .lexicon
                .get(context.word)
                .context("reading context word outside lexicon")?;
            ensure!(context.count > 0, "invalid reading context count");
            let key = (
                context.previous[0].clone(),
                context.previous[1].clone(),
                word.surface.clone(),
                word.pos.clone(),
            );
            readings
                .entry(key)
                .or_default()
                .insert(context.word, context.count);
        }
        Ok(Self {
            pieces,
            words: definition.lexicon,
            nodes,
            piece_nodes,
            readings,
        })
    }
    fn compact(states: Vec<State>) -> Vec<State> {
        let mut best = BTreeMap::new();
        for state in states {
            let tail: Vec<_> = state.path.iter().rev().take(2).copied().collect();
            let key = (state.node, state.kana, state.ended, tail);
            let old = best.entry(key).or_insert_with(|| state.clone());
            if state.score > old.score {
                *old = state;
            }
        }
        let mut values: Vec<_> = best.into_values().collect();
        values.sort_by(|a, b| b.score.total_cmp(&a.score));
        values.truncate(64);
        values
    }
    pub fn advance(
        &self,
        states: &[State],
        piece: usize,
        initial: bool,
        min: usize,
        max: usize,
    ) -> Vec<State> {
        let Some(raw) = self.pieces.get(piece) else {
            return vec![];
        };
        if piece < 7 || (raw.starts_with('▁') && !initial) {
            return vec![];
        }
        let text = raw.trim_start_matches('▁');
        if text.is_empty() {
            return if initial { states.to_vec() } else { vec![] };
        }
        let mut current = states.to_vec();
        for c in text.chars() {
            current = self.advance_char(&current, c, min, max);
            if current.is_empty() {
                break;
            }
        }
        current
    }
    fn advance_char(&self, current: &[State], c: char, min: usize, max: usize) -> Vec<State> {
        let mut next_states = vec![];
        for state in current {
            if state.ended {
                continue;
            }
            let Some(&next) = self.nodes[state.node].children.get(&c) else {
                continue;
            };
            if !self.nodes[next].children.is_empty() {
                let mut partial = state.clone();
                partial.node = next;
                next_states.push(partial);
            }
            for &word_id in &self.nodes[next].words {
                let word = &self.words[word_id];
                let previous = state.path.last().map(|&i| &self.words[i]);
                if word.pos == "接尾辞"
                    && previous.is_none_or(|p| {
                        !p.pos.starts_with("名詞")
                            && !p.pos.starts_with("動詞")
                            && !p.pos.starts_with("形容詞")
                    })
                {
                    continue;
                }
                let prior2 = state
                    .path
                    .iter()
                    .rev()
                    .nth(1)
                    .map(|&i| self.words[i].surface.clone())
                    .unwrap_or_default();
                let key = (
                    prior2,
                    previous.map(|p| p.surface.clone()).unwrap_or_default(),
                    word.surface.clone(),
                    word.pos.clone(),
                );
                let local = self.readings.get(&key);
                // Resolve readings before length checks. A shorter, wrong
                // reading must never be selected to fit a difficulty bound.
                let preferred = self.nodes[next]
                    .words
                    .iter()
                    .copied()
                    .filter(|&i| self.words[i].pos == word.pos)
                    .max_by_key(|&i| {
                        (
                            local.and_then(|m| m.get(&i)).copied().unwrap_or(0),
                            self.words[i]
                                .metadata
                                .get("count")
                                .and_then(|n| n.as_u64())
                                .unwrap_or(1),
                            std::cmp::Reverse(i),
                        )
                    });
                if preferred != Some(word_id) {
                    continue;
                }
                if let Some(prior) = previous {
                    if lexical_collision(prior, word)
                        || polite_collision(prior, word)
                        || (prior.pos == "助詞" && word.pos == "助詞")
                    {
                        continue;
                    }
                }
                if word.surface == "。" && (state.kana < min || !previous.is_some_and(can_end)) {
                    continue;
                }
                if word.surface == "、" && previous.is_none_or(|w| w.surface == "、") {
                    continue;
                }
                if state.path.iter().rev().take(2).count() == 2
                    && state.path.iter().rev().take(2).all(|&i| i == word_id)
                {
                    continue;
                }
                let length = if word.surface == "、" {
                    1
                } else {
                    word.kana.chars().count()
                };
                if state.kana + length > max {
                    continue;
                }
                let mut complete = state.clone();
                complete.node = 0;
                complete.kana += length;
                complete.path.push(word_id);
                complete.ended = word.surface == "。";
                let count = word
                    .metadata
                    .get("count")
                    .and_then(|n| n.as_u64())
                    .unwrap_or(1);
                // Prefer complete dictionary compounds to arbitrary finer splits.
                complete.score += (count as f64 + 1.).ln() * 0.1 - 10.;
                next_states.push(complete);
            }
        }
        Self::compact(next_states)
    }
    pub fn valid_pieces(
        &self,
        states: &[State],
        initial: bool,
        min: usize,
        max: usize,
    ) -> Vec<bool> {
        let mut valid = vec![false; self.pieces.len()];
        self.walk_pieces(0, states, initial, (min, max), &mut valid);
        valid
    }
    fn walk_pieces(
        &self,
        node: usize,
        states: &[State],
        initial: bool,
        bounds: (usize, usize),
        valid: &mut [bool],
    ) {
        for &id in &self.piece_nodes[node].ids {
            if initial || (node != 0 && !self.pieces[id].starts_with('▁')) {
                valid[id] = true;
            }
        }
        for (&c, &child) in &self.piece_nodes[node].children {
            if !initial && !self.piece_nodes[child].bare_any {
                continue;
            }
            let next = self.advance_char(states, c, bounds.0, bounds.1);
            if !next.is_empty() {
                self.walk_pieces(child, &next, initial, bounds, valid);
            }
        }
    }
    pub fn finished(&self, states: &[State]) -> bool {
        states.iter().any(|s| s.ended)
    }
    pub fn decode(&self, states: &[State]) -> Result<Vec<Segment>> {
        let best = states
            .iter()
            .filter(|s| s.ended)
            .max_by(|a, b| a.score.total_cmp(&b.score))
            .context("no completed lexical path")?;
        Ok(best
            .path
            .iter()
            .map(|&i| {
                let w = &self.words[i];
                Segment {
                    surface: w.surface.clone(),
                    kana: w.kana.clone(),
                    pos: w.pos.clone(),
                }
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    fn fixture() -> Result<Lexicon> {
        let mut pieces = vec!["BOS", "EOS", "PAD", "EASY", "NORMAL", "HARD", "UNK"];
        pieces.extend([
            "▁本を",
            "読",
            "んだ",
            "。",
            "▁トム",
            "▁本棚",
            "▁本を取り",
            "▁本",
            "本",
            "を",
            "本を",
            "▁",
        ]);
        let words = [
            ("本", "ほん", "名詞"),
            ("を", "を", "助詞"),
            ("読んだ", "よんだ", "動詞"),
            ("。", "", "記号"),
            ("棚", "たな", "名詞"),
            ("本棚", "ほんだな", "名詞"),
            ("取り", "とり", "動詞-非終止"),
        ];
        let value = json!({"encoding":"sentencepiece_lexicon_v1",
            "tokens":pieces.iter().enumerate().map(|(id,surface)|json!({"id":id,"surface":surface})).collect::<Vec<_>>(),
            "lexicon":words.iter().enumerate().map(|(id,(surface,kana,pos))|json!({"id":id,"surface":surface,"kana":kana,"pos":pos})).collect::<Vec<_>>()});
        Lexicon::load(serde_json::from_value(value)?)
    }
    #[test]
    fn subword_fragments_only_complete_registered_words() -> Result<()> {
        let lexicon = fixture()?;
        let states = lexicon.advance(&[State::initial()], 7, true, 5, 40);
        let states = lexicon.advance(&states, 8, false, 5, 40);
        assert!(!lexicon.finished(&states));
        assert!(lexicon.advance(&states, 10, false, 5, 40).is_empty());
        let states = lexicon.advance(&states, 9, false, 5, 40);
        let states = lexicon.advance(&states, 10, false, 5, 40);
        assert!(lexicon.finished(&states));
        let segments = lexicon.decode(&states)?;
        assert_eq!(
            segments
                .iter()
                .map(|s| s.surface.as_str())
                .collect::<String>(),
            "本を読んだ。"
        );
        assert_eq!(
            segments.iter().map(|s| s.kana.as_str()).collect::<String>(),
            "ほんをよんだ"
        );
        assert!(lexicon
            .advance(&[State::initial()], 11, true, 5, 40)
            .is_empty());
        Ok(())
    }
    #[test]
    fn shared_piece_prefix_masks_equal_individual_transitions() -> Result<()> {
        let lexicon = fixture()?;
        for bounds in [(1, 3), (5, 12), (13, 22), (1, 40)] {
            let mut frontier = vec![(vec![State::initial()], true)];
            for _ in 0..5 {
                let mut next_frontier = Vec::new();
                for (states, initial) in frontier {
                    let valid = lexicon.valid_pieces(&states, initial, bounds.0, bounds.1);
                    for (id, &allowed) in valid.iter().enumerate() {
                        let next = lexicon.advance(&states, id, initial, bounds.0, bounds.1);
                        assert_eq!(allowed, !next.is_empty(), "piece {id}, initial {initial}");
                        if allowed {
                            next_frontier.push((next, false));
                        }
                    }
                }
                frontier = next_frontier;
            }
        }
        Ok(())
    }
    #[test]
    fn dictionary_compounds_keep_their_contextual_compound_reading() -> Result<()> {
        let lexicon = fixture()?;
        let states = lexicon.advance(&[State::initial()], 12, true, 1, 40);
        let states = lexicon.advance(&states, 10, false, 1, 40);
        assert_eq!(lexicon.decode(&states)?[0].kana, "ほんだな");
        Ok(())
    }
    #[test]
    fn nonterminal_predicate_cannot_finish_and_kana_bounds_are_enforced() -> Result<()> {
        let lexicon = fixture()?;
        let states = lexicon.advance(&[State::initial()], 13, true, 5, 40);
        assert!(lexicon.advance(&states, 10, false, 5, 40).is_empty());
        assert!(lexicon
            .advance(&[State::initial()], 12, true, 1, 3)
            .is_empty());
        Ok(())
    }
    #[test]
    fn bounds_cannot_force_a_less_likely_homograph_reading() -> Result<()> {
        let mut pieces = vec!["BOS", "EOS", "PAD", "EASY", "NORMAL", "HARD", "UNK"];
        pieces.extend(["▁車", "。"]);
        let value = json!({"encoding":"sentencepiece_lexicon_v1",
            "tokens":pieces.iter().enumerate().map(|(id,surface)|json!({"id":id,"surface":surface})).collect::<Vec<_>>(),
            "lexicon":[{"id":0,"surface":"車","kana":"くるま","pos":"名詞","metadata":{"count":500}},
                       {"id":1,"surface":"車","kana":"しゃ","pos":"名詞","metadata":{"count":5}},
                       {"id":2,"surface":"。","kana":"","pos":"記号"}]});
        let lexicon = Lexicon::load(serde_json::from_value(value)?)?;
        assert!(lexicon
            .advance(&[State::initial()], 7, true, 1, 2)
            .is_empty());
        let states = lexicon.advance(&[State::initial()], 7, true, 1, 3);
        let states = lexicon.advance(&states, 8, false, 1, 3);
        assert_eq!(lexicon.decode(&states)?[0].kana, "くるま");
        Ok(())
    }
    #[test]
    fn attested_local_context_resolves_ambiguous_verb_readings() -> Result<()> {
        let pieces = [
            "BOS",
            "EOS",
            "PAD",
            "EASY",
            "NORMAL",
            "HARD",
            "UNK",
            "▁実験を行った",
            "。",
        ];
        let value = json!({"encoding":"sentencepiece_lexicon_v1",
            "tokens":pieces.iter().enumerate().map(|(id,surface)|json!({"id":id,"surface":surface})).collect::<Vec<_>>(),
            "lexicon":[{"id":0,"surface":"行った","kana":"いった","pos":"動詞","metadata":{"count":500}},
                       {"id":1,"surface":"行った","kana":"おこなった","pos":"動詞","metadata":{"count":50}},
                       {"id":2,"surface":"実験","kana":"じっけん","pos":"名詞"},
                       {"id":3,"surface":"を","kana":"を","pos":"助詞"},
                       {"id":4,"surface":"。","kana":"","pos":"記号"}],
            "reading_contexts":[{"previous":["実験","を"],"word":1,"count":3}]});
        let lexicon = Lexicon::load(serde_json::from_value(value)?)?;
        let states = lexicon.advance(&[State::initial()], 7, true, 5, 40);
        let states = lexicon.advance(&states, 8, false, 5, 40);
        let reading = lexicon
            .decode(&states)?
            .iter()
            .map(|s| s.kana.as_str())
            .collect::<String>();
        assert_eq!(reading, "じっけんをおこなった");
        Ok(())
    }
}
