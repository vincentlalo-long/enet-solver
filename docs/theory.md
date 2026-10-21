# ElasticNet from First Principles

Theory notes for `enet-solver`. This is a **learning document**: it builds the whole chain from
ordinary least squares to a sparse ElasticNet solver, with every algebraic step written out.

Every section maps to a function in `src/enet/reference.py`; see [§9](#9-from-theory-to-code).

**Notation.** `X ∈ R^{N×D}` (rows = samples, columns = features), `y ∈ R^N`, `w ∈ R^D`,
intercept `b ∈ R`, residual `r = y − Xw − b·1`. `N` samples, `D` features, `1` is the all-ones
vector. Unless stated otherwise we assume centered data (intercept handled separately), so
`1ᵀx_j = 0` and `1ᵀy = 0`.

---

## Table of contents

1. [Linear regression and OLS](#1-linear-regression-and-ols)
2. [Regularization](#2-regularization)
3. [Why L1 produces exact zeros](#3-why-l1-produces-exact-zeros)
4. [The optimization problem and KKT](#4-the-optimization-problem-and-kkt)
5. [Soft-thresholding](#5-soft-thresholding)
6. [Coordinate descent](#6-coordinate-descent)
7. [Stopping criteria](#7-stopping-criteria)
8. [Regularization path and warm start](#8-regularization-path-and-warm-start)
9. [From theory to code](#9-from-theory-to-code)
10. [Extension: the GLM family](#10-extension-the-glm-family)
11. [References](#11-references)

---

## 1. Linear regression and OLS

### 1.1 The model

We want a linear predictor `ŷ = Xw + b` close to `y`. "Close" is measured by the residual sum of
squares (RSS):

```
RSS(w) = ||y − Xw||²  =  (y − Xw)ᵀ(y − Xw)
```

Ordinary Least Squares (OLS) minimizes RSS.

### 1.2 Closed-form solution (derivation)

Expand the objective:

```
RSS(w) = (y − Xw)ᵀ(y − Xw)
       = yᵀy − yᵀXw − (Xw)ᵀy + (Xw)ᵀ(Xw)
       = yᵀy − 2 wᵀXᵀy + wᵀXᵀXw
```

The middle two terms are equal because `yᵀXw` is a scalar, so it equals its transpose
`wᵀXᵀy`. Now differentiate using `∂/∂w (aᵀw) = a` and
`∂/∂w (wᵀAw) = (A + Aᵀ)w`:

```
∇_w RSS(w) = −2 Xᵀy + (XᵀX + (XᵀX)ᵀ) w
           = −2 Xᵀy + 2 XᵀX w          (since XᵀX is symmetric)
```

Set the gradient to zero:

```
−2 Xᵀy + 2 XᵀX w = 0
⇒ XᵀX w = Xᵀy                          (normal equations)
⇒ ŵ_OLS = (XᵀX)⁻¹ Xᵀy
```

**Step-by-step summary**

1. Expand the squared norm into four terms.
2. Merge the two cross terms (they are equal scalars).
3. Differentiate each term.
4. Set the gradient to zero.
5. Invert `XᵀX` — **this is where everything can break.**

### 1.3 Three fatal flaws of OLS

The inverse `(XᵀX)⁻¹` is the weak point. The three failure modes below are the reason
regularization exists.

#### Flaw 1 — More features than samples (`D > N`): singular matrix

**Claim.** If `D > N`, then `XᵀX` is singular and the OLS solution is not unique.

**Proof.**

1. `X` is `N × D`, so `rank(X) ≤ min(N, D) = N`.
2. For any matrix `A`, `rank(AᵀA) = rank(A)`. Hence `rank(XᵀX) = rank(X) ≤ N < D`.
3. `XᵀX` is `D × D` but has rank `< D`. Therefore it is singular: `det(XᵀX) = 0`, so
   `(XᵀX)⁻¹` does not exist.
4. The null space of `X` is non-trivial: there exists `z ≠ 0` with `Xz = 0`. If `w₀` is any
   solution, then `X(w₀ + z) = Xw₀ + Xz = Xw₀`, so `w₀ + z` is also a solution. The solution
   set is an affine subspace of dimension at least `D − N`.

**Consequence.** When `D > N` there are infinitely many `w` achieving `RSS = 0`; the algorithm has
no criterion to choose among them. This is common for text data (TF-IDF can have `D ≈ 100,000`
while `N` is a few thousand) and for genomics.

#### Flaw 2 — Multicollinearity: numerically explosive inverse

**Claim.** When columns are (nearly) linearly dependent, the OLS estimator has enormous variance.

**Proof.**

1. Assume `y = Xw* + ε` with `E[ε] = 0`, `Var(ε) = σ²I`.
2. The estimator is `ŵ = (XᵀX)⁻¹Xᵀy`. Its variance is
   ```
   Var(ŵ) = (XᵀX)⁻¹ Xᵀ · Var(y) · X (XᵀX)⁻¹
          = (XᵀX)⁻¹ Xᵀ · σ²I · X (XᵀX)⁻¹
          = σ² (XᵀX)⁻¹ XᵀX (XᵀX)⁻¹
          = σ² (XᵀX)⁻¹.
   ```
3. Diagonalize `XᵀX = VΛVᵀ` with eigenvalues `λ₁ ≥ … ≥ λ_D ≥ 0`. Then
   ```
   (XᵀX)⁻¹ = V Λ⁻¹ Vᵀ,     Var(ŵ) = σ² V Λ⁻¹ Vᵀ.
   ```
   Along the eigenvector `v_i`, the variance is `σ² / λ_i`.
4. Near-collinearity makes some `λ_i → 0⁺`, so `σ²/λ_i → ∞`. A tiny change in the training set
   can flip a coefficient from large positive to large negative.

**Consequence.** The model is unstable and uninterpretable, even though `XᵀX` is technically
invertible.

#### Flaw 3 — No feature selection: no sparsity

**Claim.** OLS has no mechanism to set coefficients exactly to zero.

**Proof / reasoning.**

1. The OLS objective is smooth and unconstrained. Its solution is a deterministic function of
   `Xᵀy`, and generically all coordinates are nonzero: the set of data for which some `ŵ_j = 0`
   exactly has measure zero.
2. There is no penalty on the magnitude or number of nonzero coefficients, so irrelevant
   features are used like any other.
3. Consequences: (a) overfitting, because noise in junk features is fitted too; (b) poor
   interpretability, because there is no notion of "which features matter".

> These three flaws motivate **regularization**: Ridge (L2) to stabilize the inverse, Lasso (L1)
> to produce exact zeros, and ElasticNet to combine both.

---

## 2. Regularization

Add a penalty on `w` to the objective:

```
min_w  (1/(2N)) ||y − Xw||²  +  penalty(w)
```

The `1/(2N)` scaling is the scikit-learn convention; it keeps the math below clean and matches
the code.

### 2.1 Ridge (L2) — fixes the singular inverse

```
Ridge:  min_w  ||y − Xw||² + λ||w||²        (λ > 0)
```

**Derivation.**

```
∇_w = −2Xᵀy + 2XᵀXw + 2λw = 0
⇒ 2(XᵀX + λI) w = 2Xᵀy
⇒ (XᵀX + λI) w = Xᵀy
⇒ ŵ_ridge = (XᵀX + λI)⁻¹ Xᵀy
```

Why it fixes Flaw 1: if `XᵀX = VΛVᵀ`, then `XᵀX + λI = V(Λ + λI)Vᵀ`, whose eigenvalues are
`λ_i + λ`. With `λ > 0` all are strictly positive, so the matrix is positive definite and
invertible even when `XᵀX` is singular.

Why it fixes Flaw 2: the variance becomes `σ²(XᵀX + λI)⁻¹`, whose largest value is `σ²/λ` —
bounded.

**Limitation.** Ridge shrinks coefficients but generically never makes them exactly zero. It does
not select features.

### 2.2 Lasso (L1) — produces sparsity

```
Lasso:  min_w  ||y − Xw||² + λ||w||₁,     ||w||₁ = Σ_j |w_j|
```

No closed-form solution. It does select features: many `ŵ_j` become exactly zero.

### 2.3 ElasticNet — both

```
ElasticNet:  f(w) = 1/(2N)||y − Xw||² + λ₁||w||₁ + (λ₂/2)||w||²
```

with `λ₁ = α·ρ` and `λ₂ = α·(1 − ρ)`, where `α ≥ 0` is the total penalty strength and
`ρ ∈ [0, 1]` is the L1 ratio (scikit-learn's `l1_ratio`).

- `ρ = 1` → Lasso.
- `ρ = 0` → Ridge.
- `λ₂ > 0` makes the objective **strongly convex**, so the minimizer is **unique** (plain Lasso
  can have multiple minimizers under collinearity). `λ₁` keeps the sparsity mechanism.

---

## 3. Why L1 produces exact zeros

This is the conceptual core. There are two equivalent views.

### 3.1 Geometric view: the L1 ball has vertices on the axes

The penalized problem is equivalent (Lagrangian duality) to a constrained one:

```
Lasso:  min_w ||y − Xw||²   s.t.  ||w||₁ ≤ t
Ridge:  min_w ||y − Xw||²   s.t.  ||w||₂² ≤ t
```

Picture the level sets of `||y − Xw||²` as ellipses growing around the unconstrained OLS
solution, and the constraint set as a region around the origin. The solution is the first point
where an ellipse touches the region.

1. The L2 ball is a **smooth sphere**: the touching point is generically *not* on an axis, so no
   coordinate is exactly zero.
2. The L1 ball is a **polytope with sharp vertices on the axes** (a diamond in 2D). The first
   contact very often happens at a vertex — and at a vertex at least one coordinate is zero.

This is why L1 selects features and L2 only shrinks them. It is geometry, not magic.

### 3.2 Analytic view: the subdifferential at zero

The function `|w|` is not differentiable at `0`. Its subdifferential is the set of slopes that
support the function at that point:

```
∂|w| = { sign(w) }   if w ≠ 0
     = [−1, 1]       if w = 0
```

At `w = 0` the derivative can be *any* value in `[−1, 1]`. That interval is exactly the "room"
that lets a coordinate stay at zero: the optimum condition at zero is an **inequality**, not an
equation.

Concretely, for one coordinate the optimality condition is

```
v_j + λ₁ s = 0,   s ∈ ∂|w_j|
```

- If `w_j ≠ 0`, then `s = sign(w_j)` and `v_j` must equal `±λ₁` exactly.
- If `w_j = 0`, then any `s ∈ [−1, 1]` works, i.e. `|v_j| ≤ λ₁`.

So a coordinate is allowed to sit at zero whenever the data's "pull" `v_j` is smaller than the
threshold `λ₁`. That is the analytic form of the vertex argument above.

---

## 4. The optimization problem and KKT

### 4.1 The objective

Write the objective with the smooth and non-smooth parts separated:

```
f(w) = [ 1/(2N)||y − Xw − b·1||² + (λ₂/2)||w||² ]  +  λ₁ Σ_j |w_j|
       \_________________ smooth, differentiable _______________/     \_ non-smooth _/
```

`f` is convex but not differentiable wherever some `w_j = 0`. A minimizer `w*` is characterized
by the KKT condition

```
0 ∈ ∇_smooth f(w*) + λ₁ ∂||w*||₁
```

We now compute the smooth gradient explicitly.

### 4.2 Expanding the smooth part

Let `e = y − Xw − b·1`. Then

```
||e||² = eᵀe
       = (y − Xw − b·1)ᵀ(y − Xw − b·1)
```

Multiply the four blocks term by term:

```
= yᵀy
  − yᵀXw − (Xw)ᵀy
  − b·yᵀ1 − b·1ᵀy
  + (Xw)ᵀ(Xw)
  + b·(Xw)ᵀ1 + b·1ᵀ(Xw)
  + b²·1ᵀ1
```

Now simplify each group:

1. `yᵀXw` and `(Xw)ᵀy` are equal scalars, so they sum to `2 wᵀXᵀy`.
2. `b·yᵀ1` and `b·1ᵀy` are equal, so they sum to `2b·1ᵀy`.
3. `b·(Xw)ᵀ1` and `b·1ᵀ(Xw)` are equal, so they sum to `2b·1ᵀXw`.
4. `1ᵀ1 = N`, so the last term is `N b²`.

Therefore

```
||e||² = yᵀy − 2 wᵀXᵀy − 2b·1ᵀy + wᵀXᵀXw + 2b·1ᵀXw + N b²
```

and the smooth objective is

```
S(w) = (1/(2N)) [ yᵀy − 2 wᵀXᵀy − 2b·1ᵀy + wᵀXᵀXw + 2b·1ᵀXw + N b² ]
       + (λ₂/2) wᵀw
```

### 4.3 The gradient, coordinate by coordinate

Differentiate `S` with respect to `w_j`. Terms that do not contain `w_j` vanish:

```
∂/∂w_j (wᵀXᵀy)   = (Xᵀy)_j
∂/∂w_j (wᵀXᵀXw)  = 2 (XᵀXw)_j
∂/∂w_j (1ᵀXw)    = (Xᵀ1)_j
∂/∂w_j (wᵀw)     = 2 w_j
```

Substitute:

```
∂S/∂w_j = (1/(2N)) [ −2(Xᵀy)_j + 2(XᵀXw)_j + 2b(Xᵀ1)_j ] + (λ₂/2)·2w_j
        = (1/N) [ −(Xᵀy)_j + (XᵀXw)_j + b(Xᵀ1)_j ] + λ₂ w_j
```

Factor the bracket:

```
−(Xᵀy)_j + (XᵀXw)_j + b(Xᵀ1)_j
= [Xᵀ(Xw + b·1 − y)]_j
= −[Xᵀr]_j                          with r = y − Xw − b·1.
```

Hence

```
v_j := ∂S/∂w_j = −(1/N)(Xᵀr)_j + λ₂ w_j
```

For **centered data** the intercept is handled separately and `1ᵀX = 0`, so the `b` term
disappears from the `w`-gradient; the formula above is still the one used with the centered
residual.

### 4.4 Subdifferential and KKT

The non-smooth part is `λ₁ Σ_j |w_j|`, which separates over coordinates:

```
∂/∂w_j (λ₁|w_j|) = λ₁ ∂|w_j|
```

with

```
∂|w_j| = { sign(w_j) }   if w_j ≠ 0
       = [−1, 1]         if w_j = 0
```

The KKT condition `0 ∈ ∇_smooth f + λ₁ ∂||w||₁` is therefore, coordinate by coordinate,

```
0 ∈ v_j + λ₁ ∂|w_j|
```

### 4.5 Reading the three cases

**Case `w_j > 0`.** Then `∂|w_j| = {+1}`, so

```
0 = v_j + λ₁ · 1   ⇒   v_j = −λ₁
```

**Case `w_j < 0`.** Then `∂|w_j| = {−1}`, so

```
0 = v_j + λ₁ · (−1)   ⇒   v_j = +λ₁
```

**Case `w_j = 0`.** Then `∂|w_j| = [−1, 1]`, so there exists `s ∈ [−1, 1]` with

```
0 = v_j + λ₁ s   ⇒   v_j = −λ₁ s   ⇒   |v_j| ≤ λ₁
```

Combining:

```
w_j > 0 :   v_j = −λ₁
w_j < 0 :   v_j = +λ₁
w_j = 0 :   |v_j| ≤ λ₁
```

**Interpretation.** The quantity `v_j` is the "pull" of the data on coordinate `j`. If that pull
is smaller in magnitude than `λ₁`, the coordinate is allowed to stay at exactly zero. This is the
selection mechanism, now derived rather than asserted.

---

## 5. Soft-thresholding

Fix all coordinates except `w_j`. We solve the one-dimensional subproblem exactly. This is the
building block of coordinate descent.

### 5.1 The coordinate subproblem

Let `r₋ⱼ` be the residual with the current `w_j` removed:

```
r₋ⱼ = y − Σ_{k≠j} x_k w_k − b·1
```

Then the full residual is `y − Xw − b·1 = r₋ⱼ − x_j w_j`, so the objective as a function of the
single variable `w_j` is

```
g(w_j) = 1/(2N) ||r₋ⱼ − x_j w_j||² + λ₁|w_j| + (λ₂/2) w_j² + const
```

### 5.2 Expanding the squared term

Expand the norm:

```
||r₋ⱼ − x_j w_j||²
= (r₋ⱼ − x_j w_j)ᵀ(r₋ⱼ − x_j w_j)
= r₋ⱼᵀr₋ⱼ − r₋ⱼᵀ(x_j w_j) − (x_j w_j)ᵀr₋ⱼ + (x_j w_j)ᵀ(x_j w_j)
= ||r₋ⱼ||² − 2 w_j (x_jᵀr₋ⱼ) + w_j² (x_jᵀx_j)
```

Define

```
z_j = ||x_j||² = x_jᵀx_j
ρ_j = x_jᵀ r₋ⱼ
```

Then

```
||r₋ⱼ − x_j w_j||² = ||r₋ⱼ||² − 2 ρ_j w_j + z_j w_j²
```

and, dropping the constant `||r₋ⱼ||²`,

```
g(w_j) = (1/(2N)) (z_j w_j² − 2 ρ_j w_j) + λ₁|w_j| + (λ₂/2) w_j² + const
```

### 5.3 Case `w_j > 0`

On this region `|w_j| = w_j`, so `g` is differentiable. Differentiate:

```
g'(w_j) = (1/(2N)) (2 z_j w_j − 2 ρ_j) + λ₁ + λ₂ w_j
        = (1/N) (z_j w_j − ρ_j) + λ₁ + λ₂ w_j
```

Set to zero:

```
(1/N)(z_j w_j − ρ_j) + λ₁ + λ₂ w_j = 0
```

Multiply the whole equation by `N`:

```
z_j w_j − ρ_j + Nλ₁ + Nλ₂ w_j = 0
```

Group the `w_j` terms:

```
(z_j + Nλ₂) w_j = ρ_j − Nλ₁
```

Therefore

```
w_j = (ρ_j − Nλ₁) / (z_j + Nλ₂)
```

This case is self-consistent only when the result is positive, i.e. `ρ_j > Nλ₁`.

### 5.4 Case `w_j < 0`

On this region `|w_j| = −w_j`, so the derivative of the penalty is `−λ₁`:

```
g'(w_j) = (1/N)(z_j w_j − ρ_j) − λ₁ + λ₂ w_j
```

Set to zero and multiply by `N`:

```
z_j w_j − ρ_j − Nλ₁ + Nλ₂ w_j = 0
⇒ (z_j + Nλ₂) w_j = ρ_j + Nλ₁
⇒ w_j = (ρ_j + Nλ₁) / (z_j + Nλ₂)
```

This case is self-consistent only when the result is negative, i.e. `ρ_j < −Nλ₁`.

### 5.5 Case `w_j = 0`

At `w_j = 0` the smooth part of `g` has derivative

```
g'_smooth(0) = (1/N)(z_j · 0 − ρ_j) = −ρ_j / N
```

and the penalty contributes the interval `λ₁[−1, 1]`. The condition for zero to be optimal is

```
0 ∈ −ρ_j/N + λ₁[−1, 1]
```

which means there exists `s ∈ [−1, 1]` with

```
0 = −ρ_j/N + λ₁ s   ⇒   ρ_j/N = λ₁ s   ⇒   |ρ_j|/N ≤ λ₁   ⇒   |ρ_j| ≤ Nλ₁
```

So the zero solution is admissible exactly when `|ρ_j| ≤ Nλ₁`.

### 5.6 Assembling the operator

Combining the three cases:

```
w_j = (ρ_j − Nλ₁)/(z_j + Nλ₂)      if ρ_j >  Nλ₁
    = (ρ_j + Nλ₁)/(z_j + Nλ₂)      if ρ_j < −Nλ₁
    = 0                            if |ρ_j| ≤ Nλ₁
```

Define the **soft-thresholding operator**

```
S(z, γ) = sign(z) · max(|z| − γ, 0)
```

Its piecewise form is

```
S(z, γ) = z − γ      if z >  γ
        = z + γ      if z < −γ
        = 0          if |z| ≤ γ
```

Comparing the two piecewise definitions gives the compact update

```
w_j* = S(ρ_j, Nλ₁) / (z_j + Nλ₂)
```

**Sanity check.** At `ρ_j = Nλ₁` the first branch gives `0`, and at `ρ_j = −Nλ₁` the second
branch gives `0`. The function is continuous at both boundaries.

### 5.7 Properties

- **Shrinkage:** every nonzero coefficient is pulled toward zero by exactly `Nλ₁` in the
  numerator.
- **Selection:** any `|ρ_j| ≤ Nλ₁` becomes exactly zero, not merely small.
- **Odd symmetry:** `S(−z, γ) = −S(z, γ)`.
- **No closed form for the whole vector:** this only solves one coordinate at a time, which is
  precisely why we iterate.

---

## 6. Coordinate descent

### 6.1 Why one coordinate at a time

The full problem has no closed-form solution because of the non-smooth `L1` term. But §5 solved
the **one-coordinate** problem exactly. Coordinate descent turns the hard vector problem into a
sequence of easy scalar problems:

> Freeze every coordinate except `w_j`; solve that one exactly; move to the next; repeat.

### 6.2 The partial-residual identity

We want to avoid recomputing `ρ_j = x_jᵀ r₋ⱼ` from scratch every time. Relate `r₋ⱼ` to the full
residual `r`:

```
r = y − Xw − b·1
  = y − Σ_k x_k w_k − b·1
  = y − Σ_{k≠j} x_k w_k − x_j w_j − b·1
  = r₋ⱼ − x_j w_j
```

Rearrange:

```
r₋ⱼ = r + x_j w_j
```

Therefore

```
ρ_j = x_jᵀ r₋ⱼ
    = x_jᵀ(r + x_j w_j)
    = x_jᵀ r + w_j (x_jᵀ x_j)
    = x_jᵀ r + w_j z_j
```

So a single sparse dot product `x_jᵀ r` plus a scalar correction gives `ρ_j`.

### 6.3 The update

Putting §5.6 and §6.2 together:

```
z_j = ||x_j||²
ρ_j = x_jᵀ r + w_j z_j
w_j ← S(ρ_j, Nλ₁) / (z_j + Nλ₂)
```

### 6.4 Residual maintenance

After updating `w_j` by `Δ = w_j^new − w_j`, update the residual instead of recomputing it.

**Proof.**

```
r_new = y − Xw_new − b·1
      = y − X(w + Δ e_j) − b·1
      = y − Xw − b·1 − Δ x_j
      = r − Δ x_j
```

∎

For a CSC matrix, `x_j` has `nnz_j` nonzeros, so this update costs `O(nnz_j)`.

### 6.5 Intercept and sparse centering

With an intercept, the model is `ŷ = Xw + b`. The intercept is not penalized. At the optimum,
for centered data,

```
b = ȳ − x̄ᵀ w
```

where `x̄ = (1/N)Xᵀ1` and `ȳ = (1/N)1ᵀy`.

Naively centering `X` destroys sparsity. Instead, keep the **uncentered residual**
`r_raw = y − Xw` and the scalar intercept `b` separately. We now derive the update used in the
code.

**Step 1 — relation between centered and raw residual.** With `X_c = X − 1x̄ᵀ` and
`y_c = y − ȳ1`,

```
r_c = y_c − X_c w
    = (y − ȳ1) − (X − 1x̄ᵀ)w
    = y − Xw − (ȳ − x̄ᵀw)1
    = r_raw − b·1
```

so the centered residual is the raw residual shifted by the scalar intercept.

**Step 2 — centered column norm.** For column `j`:

```
z_j^c = ||x_j − x̄_j 1||²
      = (x_j − x̄_j1)ᵀ(x_j − x̄_j1)
      = x_jᵀx_j − 2x̄_j(x_jᵀ1) + x̄_j²(1ᵀ1)
      = ||x_j||² − 2x̄_j(N x̄_j) + N x̄_j²
      = ||x_j||² − N x̄_j²
```

**Step 3 — centered partial residual.** We need
`ρ_j^c = x_{c,j}ᵀ r_{c,−j}`, where `r_{c,−j} = r_c + x_{c,j} w_j` (same argument as §6.2).
Thus

```
ρ_j^c = x_{c,j}ᵀ r_c + w_j ||x_{c,j}||²
      = x_{c,j}ᵀ r_c + w_j z_j^c
```

Now expand the first term:

```
x_{c,j}ᵀ r_c
= (x_j − x̄_j1)ᵀ(r_raw − b·1)
= x_jᵀr_raw − b(x_jᵀ1) − x̄_j(1ᵀr_raw) + b x̄_j(1ᵀ1)
= x_jᵀr_raw − b N x̄_j − x̄_j Σ_i (r_raw)_i + b N x̄_j
```

The last two terms cancel if `Σ_i (r_raw)_i = N b`. Check:

```
Σ_i (r_raw)_i = 1ᵀ(y − Xw) = 1ᵀy − 1ᵀXw = Nȳ − N x̄ᵀw = N(ȳ − x̄ᵀw) = N b
```

So indeed

```
x_{c,j}ᵀ r_c = x_jᵀ r_raw − b N x̄_j
```

and therefore

```
ρ_j^c = x_jᵀ r_raw − b N x̄_j + w_j z_j^c
```

**Step 4 — intercept update.** Since `b = ȳ − x̄ᵀw`, changing `w_j` by `Δ` changes `b` by
`−x̄_j Δ`:

```
b ← b − x̄_j Δ
```

**Step 5 — initialization.** Pick `w = w₀` and set

```
b = ȳ − x̄ᵀw₀
r_raw = y − Xw₀
```

so that `Σ_i (r_raw − b)_i = 0`, i.e. the centered residual really is centered.

This gives the code's sparse update, with every quantity computable in `O(nnz_j)`.

### 6.6 Convergence argument

- Each coordinate step computes the **exact** minimizer along that coordinate, so `f` never
  increases.
- `f` is convex and bounded below, so the sequence `f(w^k)` converges.
- At any limit point, no coordinate can be improved, which is exactly the KKT condition of §4.
  Hence every limit point is a global minimizer.
- If `λ₂ > 0`, `f` is strongly convex and the minimizer is unique, so the whole sequence
  converges to it.

(A rigorous statement is the classical result for coordinate descent on convex, coordinate-wise
minimizable objectives; see Tseng, 2001.)

### 6.7 Complexity

- One coordinate: `O(nnz_j)` for `x_jᵀr` plus `O(nnz_j)` for `r ← r − Δx_j`.
- One sweep: `O(nnz(X))`.
- Memory: `O(nnz(X))` for CSC plus `O(N + D)`.

---

## 7. Stopping criteria

Never stop because "`w` changed little between sweeps" — that does not measure distance to the
optimum. Use the KKT conditions.

### 7.1 KKT violation

From §4.3, the smooth gradient is

```
v_j = −(1/N)(Xᵀr)_j + λ₂ w_j
```

with `r` the current centered residual. Since the centered residual satisfies `Σ_i r_i = 0`,
we have `x_{c,j}ᵀr = x_jᵀr − x̄_j Σ_i r_i = x_jᵀr`, so the raw matrix-vector product is enough.

The per-coordinate violation follows directly from §4.5:

```
viol_j = |v_j + λ₁|                 if w_j > 0
       = |v_j − λ₁|                 if w_j < 0
       = max(|v_j| − λ₁, 0)         if w_j = 0
```

Stop when `max_j viol_j ≤ tol`. At the optimum every `viol_j` is zero.

### 7.2 Duality: Fenchel setup

scikit-learn uses a relative duality gap, so a fair comparison must be aware of the difference.
We derive the gap for the centered problem.

Write the problem as `min_w g(Xw) + h(w)` with

```
g(u) = 1/(2N)||y − u||²
h(w) = λ₁||w||₁ + (λ₂/2)||w||²
```

The Fenchel dual is

```
max_θ  −g*(θ) − h*(−Xᵀθ)
```

where `g*` and `h*` are the convex conjugates.

### 7.3 Conjugate of `g`

By definition,

```
g*(θ) = sup_u [ θᵀu − 1/(2N)||y − u||² ]
```

Substitute `u = y − v` (so `v = y − u`):

```
g*(θ) = sup_v [ θᵀ(y − v) − 1/(2N)||v||² ]
      = θᵀy + sup_v [ −θᵀv − 1/(2N)||v||² ]
```

The inner supremum is a concave quadratic in `v`. Its gradient is `−θ − v/N`; setting it to zero
gives `v = −Nθ`. The value there is

```
−θᵀ(−Nθ) − 1/(2N)||−Nθ||²
= N||θ||² − (N²/(2N))||θ||²
= N||θ||² − (N/2)||θ||²
= (N/2)||θ||²
```

Therefore

```
g*(θ) = θᵀy + (N/2)||θ||²
```

### 7.4 Conjugate of `h`

`h` separates over coordinates:

```
h*(z) = Σ_j sup_{w_j} [ z_j w_j − λ₁|w_j| − (λ₂/2)w_j² ]
```

For one coordinate, assume `λ₂ > 0`. Consider `w_j > 0`, so `|w_j| = w_j`. The inner function is

```
z_j w_j − λ₁ w_j − (λ₂/2)w_j²
```

Its derivative is `z_j − λ₁ − λ₂ w_j`; setting it to zero gives

```
w_j = (z_j − λ₁)/λ₂
```

which is positive exactly when `z_j > λ₁`. Substituting back:

```
z_j w_j − λ₁ w_j − (λ₂/2)w_j²
= (z_j − λ₁)w_j − (λ₂/2)w_j²
= (z_j − λ₁)²/λ₂ − (z_j − λ₁)²/(2λ₂)
= (z_j − λ₁)²/(2λ₂)
```

The case `w_j < 0` is symmetric and gives `(|z_j| − λ₁)²/(2λ₂)` when `z_j < −λ₁`. When
`|z_j| ≤ λ₁`, the optimum is `w_j = 0` and the value is `0`. Combining:

```
h*(z) = Σ_j (1/(2λ₂)) (|z_j| − λ₁)₊²,      (a)₊ = max(a, 0)
```

For `λ₂ = 0` (plain Lasso), the inner supremum is finite only if `|z_j| ≤ λ₁`, so

```
h*(z) = 0            if ||z||∞ ≤ λ₁
      = +∞           otherwise
```

### 7.5 Evaluating the gap at `θ = −r/N`

The natural dual point is the one induced by the primal residual. Since
`∇g(u) = (u − y)/N`, at `u = Xw` we get

```
θ = (Xw − y)/N = −r/N
```

Now compute each piece.

```
θᵀy = −rᵀy/N
(N/2)||θ||² = (N/2)||r||²/N² = ||r||²/(2N)
−Xᵀθ = Xᵀr/N =: z
```

For `λ₂ > 0`, the dual value is

```
D(θ) = −g*(θ) − h*(−Xᵀθ)
     = −θᵀy − (N/2)||θ||² − h*(z)
     = rᵀy/N − ||r||²/(2N) − h*(z)
```

and the primal value is

```
P(w) = ||r||²/(2N) + λ₁||w||₁ + (λ₂/2)||w||²
```

Subtract:

```
gap = P(w) − D(θ)
    = ||r||²/(2N) + λ₁||w||₁ + (λ₂/2)||w||²
      − rᵀy/N + ||r||²/(2N) + h*(z)
    = ||r||²/N − rᵀy/N + λ₁||w||₁ + (λ₂/2)||w||² + h*(z)
```

This is the quantity the code computes. Weak duality guarantees `gap ≥ 0`; it reaches `0` at the
optimum.

### 7.6 Lasso special case (`λ₂ = 0`)

When `λ₂ = 0`, `h*` is an indicator function, so `θ` must be dual feasible:

```
||Xᵀθ||∞ ≤ λ₁
```

At a non-optimal iterate this may fail. To keep a valid bound, rescale

```
θ ← θ · min(1, λ₁ / ||Xᵀθ||∞)
```

and evaluate `D(θ)` with the rescaled `θ`. This is what the code does.

### 7.7 Why tolerance semantics matter

`tol` is not a universal quantity:

- Our `tol` = maximum KKT violation.
- scikit-learn's `tol` = relative duality gap.

Two solvers "with `tol = 1e-9`" can therefore stop at different accuracies. Any benchmark must
either report the final KKT violation for both (as `bench/bench.py` does) or compare
time-to-the-same-criterion. Otherwise the timing comparison is meaningless.

---

## 8. Regularization path and warm start

In practice we solve not one `α` but a **sequence** `α₁ > α₂ > …`, the regularization path.

### 8.1 The largest useful alpha

At `w = 0` the residual is `r = y_c` (centered `y`). From §4.3,

```
v_j = −(1/N)(Xᵀy_c)_j
```

KKT at `w_j = 0` requires `|v_j| ≤ λ₁` for every `j`:

```
(1/N)|(Xᵀy_c)_j| ≤ λ₁        for all j
```

The smallest `λ₁` that satisfies all `D` inequalities is the largest left-hand side:

```
λ₁^max = max_j |x_jᵀ y_c| / N
```

Because the centered `y_c` sums to zero, `x_{c,j}ᵀy_c = x_jᵀy_c`, so the raw dot product is
correct. Finally, since `λ₁ = αρ`,

```
α_max = λ₁^max / ρ
```

For `ρ = 0` (Ridge) the solution is never exactly zero, so `α_max = ∞` and the path must start
from a chosen finite `α` instead.

### 8.2 The path

Choose a descending geometric grid:

```
α_k = α_max · eps^(k/(n_alphas−1)),      k = 0, …, n_alphas−1
```

with `eps` a small fraction (e.g. `1e-3`). For each `α_k`, solve with the previous solution as
the starting point.

### 8.3 Why warm start helps

As `α` decreases continuously, the minimizer moves continuously (strict convexity when
`λ₂ > 0`). Starting from the previous minimizer therefore places the solver very close to the new
one, so far fewer sweeps are needed. Warm start is often a larger speed-up than low-level
optimizations, and it is the foundation for active-set / strong-rule screening:

- **Strong rules** (Tibshirani et al., 2012) use the previous solution to predict which
  coefficients can be safely set to zero at the next `α`, shrinking the working set.
- **Active set** methods solve on a small subset of coordinates and expand it only when the KKT
  conditions are violated.

Both are natural extensions of the loop in §6; neither changes the underlying math.

---

## 9. From theory to code

| Concept | Function / component | File |
|---|---|---|
| Objective `f(w)` | `objective` | `reference.py` |
| Soft-threshold `S(z, γ)` | `soft_threshold` | `reference.py` |
| Column statistics and centering | `_column_stats` | `reference.py` |
| Coordinate descent | `coordinate_descent`, `_cd_dense`, `_cd_sparse` | `reference.py` |
| KKT violation | `kkt_violation` | `reference.py` |
| Duality gap | `duality_gap` | `reference.py` |
| `α_max` | `alpha_max` | `reference.py` |
| Warm-started path | `fit_path` | `reference.py` |
| C++ loop over CSC | `coordinate_descent_csc` | `_core.cpp` |
| Python → C++ wrapper | `coordinate_descent_cpp` | `backend.py` |

The NumPy implementation is deliberately readable and serves as the correctness oracle. The C++
implementation mirrors it exactly; the test suite asserts they agree to machine precision.

---

## 10. Extension: the GLM family

The update in §6 does not actually depend on the loss being squared error. It only needs the
gradient and curvature of the loss. We now derive the general form.

### 10.1 Second-order expansion

Let the loss be `ℓ(ŷ_i, y_i)` with `ŷ = Xw + b`. Define

```
g_i = ∂ℓ(ŷ_i, y_i)/∂ŷ_i
h_i = ∂²ℓ(ŷ_i, y_i)/∂ŷ_i²
```

Around the current prediction, the second-order Taylor expansion is

```
ℓ(ŷ_i + δ_i, y_i) ≈ ℓ(ŷ_i, y_i) + g_i δ_i + (1/2) h_i δ_i²
```

### 10.2 Coordinate subproblem

Change only `w_j` by `Δ = u − w_j`, so `δ_i = x_ij Δ`. Dropping constants:

```
(1/N) Σ_i [ g_i x_ij (u − w_j) + (1/2) h_i x_ij² (u − w_j)² ]
+ λ₁|u| + (λ₂/2)u²
```

Define the aggregated curvature and gradient:

```
A_j = (1/N) Σ_i h_i x_ij²
B_j = (1/N) Σ_i g_i x_ij
```

Expand the quadratic in `u`:

```
(1/N) Σ_i g_i x_ij (u − w_j) = B_j u − B_j w_j
(1/N) Σ_i (1/2) h_i x_ij² (u − w_j)²
  = (1/2)A_j (u² − 2w_j u + w_j²)
  = (1/2)A_j u² − A_j w_j u + const
```

Adding the `L2` term:

```
(1/2)(A_j + λ₂) u² − (A_j w_j − B_j) u + λ₁|u| + const
```

This has the same form as §5.3–5.5, with

```
c = A_j + λ₂
d = A_j w_j − B_j
```

and its minimizer is

```
u = S(d, λ₁) / c = S(A_j w_j − B_j, λ₁) / (A_j + λ₂)
```

### 10.3 Working residual form

Define `r_i = −g_i` (the negative gradient, a "working residual"). Then
`−B_j = (1/N) Σ_i r_i x_ij`, so

```
ρ_j := A_j w_j + (1/N) Σ_i r_i x_ij
u = S(ρ_j, λ₁) / (A_j + λ₂)
```

### 10.4 Recovering the Gaussian case

For squared error `ℓ(ŷ, y) = (1/2)(ŷ − y)²`:

```
g_i = ŷ_i − y_i,    h_i = 1,    r_i = y_i − ŷ_i
A_j = (1/N)||x_j||²,    ρ_j = (1/N)x_jᵀr + w_j (1/N)||x_j||²
```

Substituting and multiplying numerator and denominator by `N`:

```
u = S(x_jᵀr + w_j||x_j||², Nλ₁) / (||x_j||² + Nλ₂)
```

which is exactly §6.3. The general form therefore covers Logistic, Poisson, Gamma, and other
GLMs by changing only `g_i` and `h_i`.

---

## 11. References

1. Friedman, J., Hastie, T., & Tibshirani, R. (2010). *Regularization Paths for Generalized
   Linear Models via Coordinate Descent.* Journal of Statistical Software. — the algorithm behind
   glmnet and scikit-learn.
2. Tibshirani, R. (1996). *Regression Shrinkage and Selection via the Lasso.* JRSS-B.
3. Zou, H., & Hastie, T. (2005). *Regularization and Variable Selection via the Elastic Net.*
   JRSS-B.
4. Boyd, S., & Vandenberghe, L. (2004). *Convex Optimization.* Cambridge University Press. —
   KKT, duality, subgradients.
5. Tseng, P. (2001). *Convergence of a Block Coordinate Descent Method for Nondifferentiable
   Minimization.* Journal of Optimization Theory and Applications.
6. scikit-learn: `sklearn.linear_model.ElasticNet` and `sklearn/linear_model/_cd_fast.pyx` — the
   reference implementation used as the test oracle.
