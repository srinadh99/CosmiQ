# CosmiQ: Uncertainty Quantification for Cosmic Ray Detection in Astronomical Images with Variational U‑Net

This repo extends **[deepCR](https://github.com/profjsb/deepCR/tree/master)**  with a **variational (Bayesian) U‑Net** that:

* Detects cosmic rays in HST/ACS images (same task as deepCR)
* Outputs **per‑pixel CR probabilities**
* Provides **per‑pixel uncertainty maps** via Monte Carlo (MC) sampling
* Evaluates **calibration** (Brier, NLL, ECE) and **risk–coverage** curves

You can switch between the deterministic baseline and VI model with a single flag:

* `bayesian=False` → standard `UNet2Sigmoid` (deepCR‑style)
* `bayesian=True`  → `UNet2SigmoidVI` with Bayesian conv layers

---

## 1. Model & Loss

### 1.1 Baseline U‑Net (deterministic)

Per‑pixel CR probability:
$p_\theta(y=1\mid x) = \sigma(f_\theta(x))$

This is an inline equation: $E=mc^2$.
This is a displayed equation:
$\sum_{i=1}^{n} i = \frac{n(n+1)}{2}$

### Binary cross‑entropy loss:
$\mathcal{L}_{\text{BCE}}(\theta)=$ 

$-\frac{1}{N}\sum_{i=1}^{N} \left[ y_i \log p_\theta(y_i=1\mid x_i) + (1-y_i)\log\big(1-p_\theta(y_i=1\mid x_i)\big) \right]$

### 1.2 Variational U‑Net (Bayesian weights)

Each conv weight has a **Gaussian posterior**:
$w \sim q_\phi(w)=\mathcal{N}(\mu,\sigma^2),
\qquad
\sigma = \text{softplus}(\rho)=\log(1+e^{\rho}).$

Gaussian prior:
$p(w)=\mathcal{N}(0,\sigma_p^2).$

### Per‑parameter KL term: 

$\mathrm{KL}\big(q_\phi(w)||p(w)\big)=$
$\log\frac{\sigma_p}{\sigma_q}+\frac{\sigma_q^2 + \mu^2}{2\sigma_p^2} \frac{1}{2}.$

The total training loss (ELBO‑style) is:
$
\mathcal{L}
===========

\mathcal{L}*{\text{BCE}}
+
\beta \cdot
\frac{1}{N*{\text{train}}}
\sum_{\ell}
\mathrm{KL}\big(q_\phi(w_\ell),|,p(w_\ell)\big),
$
where:

* ( \beta = \texttt{kl_beta} ) controls the strength of the KL term
* ( N_{\text{train}} ) is the number of training samples

---

## 2. Predictive Uncertainty (MC Sampling)

At test time we draw (T) Monte Carlo samples from the weight posterior:
$$
\hat{p}(y=1\mid x)
\approx
\frac{1}{T}\sum_{t=1}^{T}
\sigma\big(f_{W^{(t)}}(x)\big),
\qquad
W^{(t)}\sim q_\phi(W).
$$

Per‑pixel **predictive entropy** (uncertainty map):
$$
H(x)
====

## -\hat{p}\log\hat{p}

(1-\hat{p})\log(1-\hat{p}),
$$
where (\hat{p}) is the MC‑averaged probability at each pixel.

---

## 3. Calibration Metrics

Given probabilities (\hat{p}_i) and labels (y_i\in{0,1}):

**Brier score**
$$
\mathrm{Brier}
==============

\frac{1}{N}
\sum_{i=1}^{N}
(\hat{p}_i - y_i)^2.
$$

**Negative log‑likelihood (NLL)**
$$
\mathrm{NLL}
============

-\frac{1}{N}\sum_{i=1}^{N}
\left[
y_i\log\hat{p}_i
+
(1-y_i)\log(1-\hat{p}_i)
\right].
$$

**Expected calibration error (ECE)**
Bin predictions into (K) confidence bins (B_k):
$$
\mathrm{ECE}
============

\sum_{k=1}^{K}
\frac{|B_k|}{N}
,
\big|
\mathrm{acc}(B_k) - \mathrm{conf}(B_k)
\big|.
$$

---

## 4. Usage

### 4.1 Training (baseline vs VI)

```python
from training import train

# Baseline deepCR-style U-Net
trainer = train(
    image=train_dirs,
    mode='pair',
    name='ACS-WFC-baseline',
    hidden=32,
    epoch=50,
    bayesian=False      # <-- deterministic UNet2Sigmoid
)
trainer.train()

# Variational U-Net (VI)
trainer_vi = train(
    image=train_dirs,
    mode='pair',
    name='ACS-WFC-VI',
    hidden=32,
    epoch=50,
    bayesian=True,      # <-- UNet2SigmoidVI
    kl_beta=1e-6        # tune on validation
)
trainer_vi.train()
```

### 4.2 Inference + Uncertainty Map

```python
from model import deepCR
from astropy.io import fits

image = fits.getdata("jdba2sooq_flc.fits")[:512, :512]

# Load VI model
mdl = deepCR(mask="path/to/ACS-WFC-VI.pth", device="GPU")

# Probabilities + entropy via T Monte Carlo passes
prob_mask, entropy = mdl.clean_vi(image, T=16, return_entropy=True)

# Binary mask at threshold 0.5
binary_mask = (prob_mask > 0.5).astype("uint8")
```

---
