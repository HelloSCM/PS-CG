"""Evaluate logistic regression and MinDV on existing 2-D simulation data.

Example:
    python simu_mindv.py --data samples.npy --method both --output simulation.csv
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from torch import nn
from torch.nn import functional as F


class MinDVClassifier(nn.Module):
    """Binary linear classifier with a sample-to-boundary variance penalty."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)

    def forward(self, x):
        return self.linear(x).squeeze(-1)

    def distance_variance(self, x):
        distances = self(x).abs() / self.linear.weight.norm().clamp_min(1e-8)
        return distances.var(unbiased=False)

    def fit(self, x, y, epochs=1000, lr=0.1, reg=0.5):
        x, y = [torch.as_tensor(v, dtype=torch.float32) for v in (x, y)]
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.train()
        for _ in range(epochs):
            optimizer.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(self(x), y)
            loss = loss + reg * self.distance_variance(x)
            loss.backward()
            optimizer.step()
        return self

    @torch.no_grad()
    def predict(self, x):
        self.eval()
        return (self(torch.as_tensor(x, dtype=torch.float32)) > 0).numpy().astype(np.int64)


def make_task(groups, n):
    x = np.concatenate(groups[:3, :n])
    labels_12 = np.repeat([0, 1, 0], n)
    labels_ab = np.repeat([0, 0, 1], n)
    return x, (labels_12, labels_ab), groups[3, :n]


def parallelism(groups, n):
    """Compute centroid PS using the first n points from each group."""
    a, b, c, d = groups[:, :n].mean(axis=1)

    def cosine(u, v):
        denominator = np.linalg.norm(u) * np.linalg.norm(v)
        value = np.dot(u, v) / max(denominator, 1e-12)
        return float(np.clip(value, -1, 1))

    return cosine(a-b, c-d), cosine(a-c, b-d)


def run(args):
    samples = np.load(args.data, allow_pickle=False)
    if samples.ndim == 3:
        samples = samples[None]
    if samples.ndim != 4 or samples.shape[1] != 4 or samples.shape[-1] != 2:
        raise ValueError("Input must have shape [trials, 4, samples, 2] or [4, samples, 2]")
    if len(samples) == 0 or samples.shape[2] < args.samples_per_group or not np.isfinite(samples).all():
        raise ValueError("Input must be finite, nonempty, and provide at least n points per group")
    methods = ["logistic", "mindv"] if args.method == "both" else [args.method]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["trial", "method", "PS_mean", "PS_12", "PS_AB", "Acc_both", "Acc_12", "Acc_AB"]
    scores = {method: [] for method in methods}
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for trial, groups in enumerate(samples):
            ps_12, ps_ab = parallelism(groups, args.samples_per_group)
            x, labels, test = make_task(groups, args.samples_per_group)
            for method in methods:
                predictions = []
                for index, y in enumerate(labels):
                    if method == "logistic":
                        model = LogisticRegression(max_iter=args.max_iter)
                        model.fit(x, y)
                    else:
                        torch.manual_seed(args.seed + 2 * trial + index)
                        model = MinDVClassifier().fit(x, y, args.epochs, args.lr, args.reg)
                    predictions.append(model.predict(test) == 1)
                correct_12, correct_ab = predictions
                both = float(np.mean(correct_12 & correct_ab))
                writer.writerow(dict(trial=trial, method=method, PS_mean=(ps_12+ps_ab)/2,
                                     PS_12=ps_12, PS_AB=ps_ab, Acc_both=both,
                                     Acc_12=float(correct_12.mean()), Acc_AB=float(correct_ab.mean())))
                scores[method].append(both)
            stream.flush()
            print(f"Trial {trial+1}/{len(samples)} complete", flush=True)
    for method, values in scores.items():
        print(f"{method}: mean both accuracy={np.mean(values):.4f}; saved {args.output}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, required=True, help="Existing numeric NPY simulation data.")
    p.add_argument("--output", type=Path, default=Path("simulation_metrics.csv"))
    p.add_argument("--method", choices=("both", "logistic", "mindv"), default="both")
    p.add_argument("--reg", type=float, default=0.5, help="MinDV coefficient; zero disables the penalty.")
    p.add_argument("--epochs", type=int, default=1000, help="MinDV optimizer steps per classifier.")
    p.add_argument("--lr", type=float, default=0.1, help="MinDV Adam learning rate.")
    p.add_argument("--samples-per-group", type=int, default=100, help="Samples per concept combination.")
    p.add_argument("--max-iter", type=int, default=10000, help="Logistic regression iteration limit.")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if min(args.epochs, args.samples_per_group, args.max_iter) < 1 or args.lr <= 0 or args.reg < 0:
        p.error("Counts and lr must be positive; reg must be nonnegative")
    return args


if __name__ == "__main__":
    run(parse_args())
