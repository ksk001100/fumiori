"""Offline surface-copy diagnostics; no language-quality or reading oracle."""
from collections import Counter, defaultdict
import math
import unicodedata


def normalized(text):
    """Keep wording, spaces and commas; normalize width and terminal periods only."""
    value = unicodedata.normalize('NFKC', text).rstrip('。')
    if not value or '\0' in value:
        raise ValueError('empty text or reserved NUL separator')
    return value


def words(row):
    segments = row['segments']
    if normalized(''.join(s['surface'] for s in segments)) != normalized(row['text']):
        raise ValueError('segments do not reconstruct text')
    # These are the existing lexical segments, not an independent morphological analysis.
    if any(s['pos'] == '文字' for s in segments):
        raise ValueError('literal character tokens cannot be treated as words')
    return tuple(unicodedata.normalize('NFKC', s['surface']) for s in segments
                 if s['pos'] != '記号' and s['surface'])


def grams(sequence, n):
    return Counter(tuple(sequence[i:i+n]) for i in range(len(sequence)-n+1))


class SubstringIndex:
    """Suffix automaton over sentence-delimited training text, queried in O(length)."""
    def __init__(self, texts):
        self.edges, self.links, self.lengths = [{}], [-1], [0]
        last = 0
        for text in texts:
            for char in text + '\0':
                current = len(self.edges)
                self.edges.append({})
                self.links.append(0)
                self.lengths.append(self.lengths[last]+1)
                p = last
                while p >= 0 and char not in self.edges[p]:
                    self.edges[p][char] = current
                    p = self.links[p]
                if p >= 0:
                    q = self.edges[p][char]
                    if self.lengths[p]+1 == self.lengths[q]:
                        self.links[current] = q
                    else:
                        clone = len(self.edges)
                        self.edges.append(self.edges[q].copy())
                        self.lengths.append(self.lengths[p]+1)
                        self.links.append(self.links[q])
                        while p >= 0 and self.edges[p].get(char) == q:
                            self.edges[p][char] = clone
                            p = self.links[p]
                        self.links[q] = self.links[current] = clone
                last = current

    def longest(self, text):
        state = length = best = end = 0
        for i, char in enumerate(text):
            while state and char not in self.edges[state]:
                state = self.links[state]
                length = self.lengths[state]
            if char in self.edges[state]:
                state = self.edges[state][char]
                length += 1
            else:
                length = 0
            if length > best:
                best, end = length, i+1
        return text[end-best:end]


def self_bleu(sequences, order=4):
    """Mean leave-one-out sentence BLEU, effective order, add-one for n>=2.

    References exclude only the current observation, so duplicate outputs remain
    visible. Scores use [0,1]. A singleton has no reference and returns None.
    """
    if len(sequences) < 2:
        return None
    counters = [[grams(s, n) for n in range(1, order+1)] for s in sequences]
    maxima = [defaultdict(list) for _ in range(order)]
    for counts in counters:
        for n, counter in enumerate(counts):
            for gram, count in counter.items():
                top = maxima[n][gram]
                top.append(count)
                top.sort(reverse=True)
                del top[2:]
    scores = []
    for i, sequence in enumerate(sequences):
        if not sequence:
            scores.append(0.0)
            continue
        precisions = []
        for n, counter in enumerate(counters[i]):
            total = counter.total()
            if not total:
                break
            matches = 0
            for gram, count in counter.items():
                top = maxima[n][gram]
                maximum = (top[1] if len(top) > 1 else 0) if top[0] == count else top[0]
                matches += min(count, maximum)
            precisions.append(matches/total if n == 0 else (matches+1)/(total+1))
        reference_length = min((len(s) for j, s in enumerate(sequences) if j != i),
                               key=lambda length: (abs(length-len(sequence)), length))
        bp = math.exp(min(0, 1-reference_length/len(sequence)))
        scores.append(bp*math.exp(sum(math.log(p) for p in precisions)/len(precisions))
                      if all(precisions) else 0.0)
    return sum(scores)/len(scores)


def split_audit(splits):
    texts = {name: Counter(normalized(r['text']) for r in rows) for name, rows in splits.items()}
    result = dict(within={name: dict(rows=len(splits[name]), unique_texts=len(counts),
                                   duplicate_rows=counts.total()-len(counts)) for name, counts in texts.items()},
                  between=[])
    names = sorted(texts)
    for i, left in enumerate(names):
        for right in names[i+1:]:
            shared = texts[left].keys() & texts[right].keys()
            result['between'].append(dict(left=left, right=right, shared_texts=len(shared),
                                          examples=sorted(shared)[:5]))
    result['exact_disjoint'] = not any(r['shared_texts'] for r in result['between'])
    result['scope'] = 'NFKC + terminal period only; not semantic, paraphrase or combination disjointness'
    return result


class TrainingIndex:
    def __init__(self, rows):
        if not rows:
            raise ValueError('training reference must not be empty')
        self.texts = list(dict.fromkeys(normalized(r['text']) for r in rows))
        self.exact = set(self.texts)
        self.substrings = SubstringIndex(self.texts)
        self.bigrams = [set(grams(t, 2)) for t in self.texts]
        self.postings = defaultdict(list)
        for i, bigrams in enumerate(self.bigrams):
            for gram in bigrams:
                self.postings[gram].append(i)
        self.known = {}
        for unit, sequences in [('character', self.texts), ('word', [words(r) for r in rows])]:
            self.known[unit] = {n: set().union(*(set(grams(s, n)) for s in sequences)) for n in range(1, 5)}

    def near(self, text):
        query = set(grams(text, 2))
        candidates = set().union(*(self.postings[g] for g in query))
        best, index = 0.0, None
        for i in sorted(candidates):
            score = 2*len(query & self.bigrams[i])/(len(query)+len(self.bigrams[i]))
            if score > best:
                best, index = score, i
        if text in self.exact:  # Single-character exact matches have no bigrams.
            return 1.0, text
        return best, self.texts[index] if index is not None else None

    def evaluate(self, rows, near_threshold=.8):
        if not 0 < near_threshold <= 1:
            raise ValueError('near threshold must be in (0,1]')
        sequences = dict(character=[normalized(r['text']) for r in rows], word=[words(r) for r in rows])
        details = []
        for row, text in zip(rows, sequences['character']):
            score, reference = self.near(text)
            common = self.substrings.longest(text)
            details.append(dict(text=row['text'], exact=text in self.exact,
                                near=score >= near_threshold, nearest_dice=score, nearest_text=reference,
                                longest_common_substring=common, lcs_characters=len(common),
                                lcs_fraction=len(common)/len(text)))
        novelty = {}
        for unit, values in sequences.items():
            novelty[unit] = {}
            for n in range(1, 5):
                counts = sum((grams(s, n) for s in values), Counter())
                unseen = counts.keys()-self.known[unit][n]
                novelty[unit][str(n)] = dict(total=counts.total(), unseen=sum(counts[g] for g in unseen),
                    occurrence_rate=sum(counts[g] for g in unseen)/counts.total() if counts else None,
                    types=len(counts), unseen_types=len(unseen), type_rate=len(unseen)/len(counts) if counts else None)
        # Separate new words from novel combinations of words observed in training.
        known_words = self.known['word'][1]
        known_component_total = known_component_novel = 0
        for sequence in sequences['word']:
            for gram, count in grams(sequence, 2).items():
                if all((word,) in known_words for word in gram):
                    known_component_total += count
                    known_component_novel += count * (gram not in self.known['word'][2])
        def mean(key):
            return sum(r[key] for r in details)/len(details) if details else None
        unique = len(set(sequences['character']))
        return dict(count=len(rows), unique_texts=unique, unique_ratio=unique/len(rows) if rows else None,
                    exact_match_rate=mean('exact'), near_duplicate_rate=mean('near'),
                    nonexact_near_count=sum(r['near'] and not r['exact'] for r in details),
                    mean_lcs_characters=mean('lcs_characters'), mean_lcs_fraction=mean('lcs_fraction'),
                    ngram_novelty=novelty,
                    self_bleu4={unit: self_bleu(values) for unit, values in sequences.items()},
                    known_word_novel_bigrams=dict(total=known_component_total, novel=known_component_novel,
                        rate=known_component_novel/known_component_total if known_component_total else None),
                    details=details)
