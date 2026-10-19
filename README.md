# PS-CG

Code for the paper **Compositional Generalization through Neuroscience-inspired Geometric Constraints on Representation Structure.**

In this work, we investigate the relationship between the neuroscience-inspired parallelism score (PS) and compositional generalization (CG) capability in deep neural networks. We propose the Geometric Constraints (GeoCon) method, a geometric regularization approach incorporating parallelism score maximization (MaxPS) and distance variance minimization (MinDV).

The manuscript is currently under review at *npj Artificial Intelligence*. **This repository currently provides a preview release.** The full release will be made available as soon as possible after the paper is accepted.

## Preview experiments

The [preview](preview/) directory contains three experimental components:

1. **PS-CG correlation analysis on PACS, Office-Home, and NICO** — [corr_ps_cg.py](preview/corr_ps_cg.py) extracts frozen representations, computes PS, evaluates class and domain CG using the same out-of-combination split, and calculates Pearson and Spearman correlations.
2. **Two-dimensional MinDV simulation** — [simu_mindv.py](preview/simu_mindv.py) compares logistic regression with MinDV regularization on existing simulation data. Both concept classifiers use the same training samples, and evaluation reports individual-concept and joint accuracy.
3. **GeoCon optimization** — [algo_geocon.py](preview/algo_geocon.py) provides differentiable PS, pairwise multiclass distance variance minimization, and the combined GeoCon parameter update. It is an algorithm module for use in a training pipeline.

This preview contains the core implementations. The complete experimental pipeline will accompany the full release.

## Usage

Dependencies: Python, NumPy, SciPy, scikit-learn, PyTorch, torchvision, timm, and Pillow.

### PS-CG correlation analysis

Arrange images as `datasets/PACS/<domain>/<class>/<image>`, then run:

```bash
python preview/corr_ps_cg.py --dataset PACS --data-root datasets/PACS --output-dir results
```

Use `--dataset OfficeHome` or `--dataset NICO` with the corresponding data directory for the other datasets. Use `--models` or `--model-list` to select backbones; the default selects four models. Each CG split holds out one domain-class combination per class, with class and domain probes sharing the split. Metrics use raw extracted features.

### MinDV simulation

Provide an NPY array of shape `[trials, 4, samples, 2]` or `[4, samples, 2]`, with groups ordered `A1, A2, B1, B2`:

```bash
python preview/simu_mindv.py --data samples.npy --method both --output simulation.csv
```

The first three groups form the shared training set; `B2` is held out. Data generation and plotting are not included. Run either script with `--help` to see its arguments.

### GeoCon

Import `GeoCon` from `preview.algo_geocon` and pass a model and optimizer. The model must expose `featurizer`, `classifier_c`, and `classifier_d`, with linear classifier heads. Call `update(images, class_labels, domain_labels)` for a training step. The standalone functions `parallelism_scores` and `dis_var_min` are also available.
