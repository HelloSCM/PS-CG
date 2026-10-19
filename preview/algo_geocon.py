"""Differentiable PS, pairwise multiclass MinDV, and one GeoCon update.

The model must expose featurizer, classifier_c, and classifier_d.
Both classifier heads must be linear layers with at least two logits.

Example:
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    algorithm = GeoCon(model, optimizer, alpha=1.0, beta_c=0.1, beta_d=0.1)
    metrics = algorithm.update(images, class_labels, domain_labels)
"""

from itertools import combinations

import torch
from torch import nn
from torch.nn import functional as F


def _factor_ps(centroids, observed, eps):
    """Average context cosines within each supported target-concept pair."""
    scores = []
    for first, second in combinations(range(centroids.shape[1]), 2):
        valid = observed[:, first] & observed[:, second]
        vectors = centroids[valid, second] - centroids[valid, first]
        count = vectors.shape[0]
        if count >= 2:
            unit = F.normalize(vectors, dim=1, eps=eps)
            # Sum all off-diagonal dot products without building a Gram matrix.
            score = (unit.sum(0).square().sum() - unit.square().sum())
            scores.append((score / (count * (count - 1))).clamp(-1, 1))
    zero = centroids.sum() * 0
    return (torch.stack(scores).mean() if scores else zero), len(scores)


def parallelism_scores(features, labels_c, labels_d, eps=1e-8):
    """Return class PS, domain PS, and the pair-count-weighted dataset PS."""
    if features.ndim != 2 or len(features) == 0:
        raise ValueError("features must have nonempty shape [samples, dimensions]")
    if labels_c.shape != (len(features),) or labels_d.shape != (len(features),):
        raise ValueError("Each feature needs one class and one domain label")
    classes, c = torch.unique(labels_c, sorted=True, return_inverse=True)
    domains, d = torch.unique(labels_d, sorted=True, return_inverse=True)
    nc, nd = len(classes), len(domains)
    indices = d * nc + c
    counts = torch.bincount(indices, minlength=nd * nc)
    sums = features.new_zeros(nd * nc, features.shape[1]).index_add(0, indices, features)
    centroids = (sums / counts.clamp_min(1).unsqueeze(1)).reshape(nd, nc, -1)
    observed = counts.reshape(nd, nc) > 0
    ps_c, pairs_c = _factor_ps(centroids, observed, eps)
    ps_d, pairs_d = _factor_ps(centroids.transpose(0, 1), observed.T, eps)
    ps = (pairs_c * ps_c + pairs_d * ps_d) / max(pairs_c + pairs_d, 1)
    return ps_c, ps_d, ps


def _linear_head(classifier):
    head = getattr(classifier, "module", classifier)
    if not isinstance(head, nn.Linear) or head.out_features < 2:
        raise TypeError("MinDV requires a linear softmax head with at least two logits")
    return head


def dis_var_min(features, labels, classifier, eps=1e-8):
    """Mean pairwise distance variance."""
    head = _linear_head(classifier)
    if features.ndim != 2 or labels.shape != (len(features),) or len(features) == 0:
        raise ValueError("Expected nonempty features [N, D] and labels [N]")
    if labels.dtype != torch.long or labels.min() < 0 or labels.max() >= head.out_features:
        raise ValueError("Labels must be int64 indices within the classifier's classes")
    variances = []
    for k, l in combinations(torch.unique(labels).tolist(), 2):
        normal = head.weight[k] - head.weight[l]
        norm = normal.norm()
        if norm.detach() <= eps:
            raise ValueError(f"Classes {k} and {l} have coincident or near-coincident weights")
        offset = 0 if head.bias is None else head.bias[k] - head.bias[l]
        selected = features[(labels == k) | (labels == l)]
        distances = (selected @ normal + offset).abs() / norm
        variances.append(distances.var(unbiased=False))
    if variances:
        return torch.stack(variances).mean()
    zero = (features.sum() + head.weight.sum()) * 0
    return zero if head.bias is None else zero + head.bias.sum() * 0


class GeoCon:
    """Optimize CE_class + CE_domain + alpha*(1-PS) + beta_c*DV_c + beta_d*DV_d."""

    def __init__(self, model, optimizer, alpha=1.0, beta_c=1.0, beta_d=1.0, eps=1e-8):
        if min(alpha, beta_c, beta_d) < 0 or eps <= 0:
            raise ValueError("Regularization weights must be nonnegative and eps positive")
        self.model = model
        self.optimizer = optimizer
        self.alpha, self.beta_c, self.beta_d = alpha, beta_c, beta_d
        self.eps = eps
        _linear_head(model.classifier_c)
        _linear_head(model.classifier_d)

    def update(self, images, labels_c, labels_d):
        """Perform one training step; the caller moves inputs to the model device."""
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        features = self.model.featurizer(images)
        logits_c = self.model.classifier_c(features)
        logits_d = self.model.classifier_d(features)
        ce_c = F.cross_entropy(logits_c, labels_c)
        ce_d = F.cross_entropy(logits_d, labels_d)
        zero = features.sum() * 0
        ps_c, ps_d, ps = parallelism_scores(features, labels_c, labels_d, self.eps)
        dv_c = dis_var_min(features, labels_c, self.model.classifier_c, self.eps) if self.beta_c else zero
        dv_d = dis_var_min(features, labels_d, self.model.classifier_d, self.eps) if self.beta_d else zero
        loss = ce_c + ce_d + self.alpha * (1 - ps) + self.beta_c * dv_c + self.beta_d * dv_d
        loss.backward()
        self.optimizer.step()
        values = dict(loss=loss, ce_class=ce_c, ce_domain=ce_d, ps_class=ps_c,
                      ps_domain=ps_d, ps=ps, dv_class=dv_c, dv_domain=dv_d)
        return {name: value.detach().item() for name, value in values.items()}
