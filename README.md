# PS-CG

Code for the paper **Compositional Generalization through Neuroscience-inspired Geometric Constraints on Representation Structure.**

In this work, we investigate the relationship between the neuroscience-inspired parallelism score (PS) and compositional generalization (CG) capability in deep neural networks. We propose the Geometric Constraints (GeoCon) method, a geometric regularization approach incorporating parallelism score maximization (MaxPS) and distance variance minimization (MinDV).

The manuscript is currently under review at *npj Artificial Intelligence*. **This repository currently provides a preview release.** The full release will be made available as soon as possible after the paper is accepted.

## Preview experiments

The [preview](preview/) directory contains three experimental components:

1. **PS-CG correlation analysis on PACS, Office-Home, and NICO** — [corr_ps_cg.py](preview/corr_ps_cg.py) extracts frozen representations, computes PS, evaluates class and domain CG, and calculates Pearson and Spearman correlations.
2. **Two-dimensional MinDV simulation** — [simu_mindv.py](preview/simu_mindv.py) compares logistic regression with MinDV regularization on existing simulation data.
3. **GeoCon optimization** — [algo_geocon.py](preview/algo_geocon.py) provides differentiable PS, pairwise multiclass distance variance minimization, and the combined GeoCon parameter update.

This preview contains the core implementations. The complete experimental pipeline will accompany the full release.
