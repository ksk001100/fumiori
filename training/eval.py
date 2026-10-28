"""Measure held-out token loss; does not assess Japanese naturalness."""
import argparse
import json
import math
from pathlib import Path
import torch
from .model import greedy, loss_for
from .train import load_checkpoint, read_rows, tensor_batch

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data", default="build/semantic-data/test.jsonl")
    p.add_argument("--report", default="build/python-eval.json")
    a = p.parse_args()
    torch.set_num_threads(2)
    model, ck = load_checkpoint(a.checkpoint)
    with torch.no_grad():
        loss = float(loss_for(model, tensor_batch(read_rows(a.data), "cpu")))
    report = dict(test_loss=loss, test_perplexity=math.exp(min(loss, 80)),
                  samples=[greedy(model, d, ck["vocab"]) for d in (3, 4, 5)],
                  parameter_count=model.parameter_count(), human_quality="not assessed")
    Path(a.report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True))
