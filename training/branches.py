"""Training-only branch balancing; no prefix table is shipped at inference.

Each *exact* prefix defines a categorical next-word distribution in train.
Flatten its observed probabilities by alpha, then weight hard-label CE by
q(token|prefix)/p(token|prefix). Deterministic continuations stay unchanged.
This is a corpus reweighting hypothesis, not neural-teacher OPD.
"""
from collections import Counter, defaultdict
import hashlib
import json
import math
import torch


def branch_weights(rows, length, strength=0.5, max_ratio=4.0, row_weights=None):
    if not math.isfinite(strength) or not 0 <= strength <= 1 or not math.isfinite(max_ratio) or max_ratio < 1:
        raise ValueError('invalid branch balancing parameters')
    if row_weights is not None and (len(row_weights) != len(rows)
            or any(not math.isfinite(w) or w <= 0 for w in row_weights)):
        raise ValueError('branch row weights must be positive finite masses for every training row')
    counts = defaultdict(Counter)
    for i, row in enumerate(rows):
        seq = row['ids']
        if len(seq) > length or len(seq) < 3 or 2 in seq:
            raise ValueError('invalid unpadded training sequence')
        for j in range(1, len(seq)):
            counts[tuple(seq[:j])][seq[j]] += 1 if row_weights is None else row_weights[i]
    factors = {}
    for prefix, choices in counts.items():
        total = sum(choices.values())
        normalizer = sum((n / total) ** (1-strength) for n in choices.values())
        factors[prefix] = {token: min(max_ratio, max(1/max_ratio, (n/total) ** (-strength) / normalizer))
                           for token, n in choices.items()}
    weights = torch.zeros((len(rows), length-1), dtype=torch.float32)
    for i, row in enumerate(rows):
        seq = row['ids']
        for j in range(1, len(seq)):
            weights[i, j-1] = factors[tuple(seq[:j])][seq[j]]
    return weights, dict(prefixes=len(counts), branching_prefixes=sum(len(c)>1 for c in counts.values()),
                         strength=strength, max_ratio=max_ratio,
                         frequency_source='uniform_rows' if row_weights is None else 'sampling_mass',
                         row_mass_sha256=None if row_weights is None else hashlib.sha256(json.dumps(row_weights,separators=(',',':')).encode()).hexdigest(),
                         provenance='train split only; exact full prefixes; not deployed')
