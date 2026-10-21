# ElasticNet from First Principles

Theory notes for `enet-solver`. This is a **learning document**: it builds the whole chain from
ordinary least squares to a sparse ElasticNet solver, with every algebraic step written out.

Every section maps to a function in `src/enet/reference.py`; see [§9](#9-from-theory-to-code).

**Notation.** $X \in \mathbb{R}^{N \times D}$ (rows = samples, columns = features), $y \in \mathbb{R}^N$, $w \in \mathbb{R}^D$,
intercept $b \in \mathbb{R}$, residual $r = y - Xw - b\mathbf{1}$. $N$ samples, $D$ features, $\mathbf{1}$ is the all-ones
vector. Unless stated otherwise we assume centered data (intercept handled separately), so
$\mathbf{1}^\top x_j = 0$ and $\mathbf{1}^\top y = 0$.

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

We want a linear predictor $\hat{y} = Xw + b$ close to $y$. "Close" is measured by the residual sum of
squares (RSS):

$$
\text{RSS}(w) = \|y - Xw\|_2^2 = (y - Xw)^\top(y - Xw)
$$

Ordinary Least Squares (OLS) minimizes RSS.

### 1.2 Closed-form solution (derivation)

Expand the objective:

$$
\begin{aligned}
\text{RSS}(w) &= (y - Xw)^\top(y - Xw) \\
              &= y^\top y - y^\top Xw - (Xw)^\top y + (Xw)^\top(Xw) \\
              &= y^\top y - 2 w^\top X^\top y + w^\top X^\top Xw
\end{aligned}
$$

The middle two terms are equal because $y^\top Xw$ is a scalar, so it equals its transpose
$w^\top X^\top y$. Now differentiate using $\frac{\partial}{\partial w} (a^\top w) = a$ and
$\frac{\partial}{\partial w} (w^\top A w) = (A + A^\top)w$:

$$
\begin{aligned}
\nabla_w \text{RSS}(w) &= -2 X^\top y + (X^\top X + (X^\top X)^\top) w \\
                       &= -2 X^\top y + 2 X^\top X w \quad (\text{since } X^\top X \text{ is symmetric})
\end{aligned}
$$

Set the gradient to zero:

$$
-2 X^\top y + 2 X^\top X w = 0 \implies X^\top X w = X^\top y \implies \hat{w}_{\text{OLS}} = (X^\top X)^{-1} X^\top y
$$

**Step-by-step summary**

1. Expand the squared norm into four terms.
2. Merge the two cross terms (they are equal scalars).
3. Differentiate each term.
4. Set the gradient to zero.
5. Invert $X^\top X$ — **this is where everything can break.**

### 1.3 Three fatal flaws of OLS

The inverse $(X^\top X)^{-1}$ is the weak point. The three failure modes below are the reason
regularization exists.

#### Flaw 1 — More features than samples ($D > N$): singular matrix

**Claim.** If $D > N$, then $X^\top X$ is singular and the OLS solution is not unique.

**Proof.**

1. $X$ is $N \times D$, so $\operatorname{rank}(X) \le \min(N, D) = N$.
2. For any matrix $A$, $\operatorname{rank}(A^\top A) = \operatorname{rank}(A)$. Hence $\operatorname{rank}(X^\top X) = \operatorname{rank}(X) \le N < D$.
3. $X^\top X$ is $D \times D$ but has rank $< D$. Therefore it is singular: $\det(X^\top X) = 0$, so
   $(X^\top X)^{-1}$ does not exist.
4. The null space of $X$ is non-trivial: there exists $z \ne 0$ with $Xz = 0$. If $w_0$ is any
   solution, then $X(w_0 + z) = Xw_0 + Xz = Xw_0$, so $w_0 + z$ is also a solution. The solution
   set is an affine subspace of dimension at least $D - N$.

**Consequence.** When $D > N$ there are infinitely many $w$ achieving $\text{RSS} = 0$; the algorithm has
no criterion to choose among them. This is common for text data (TF-IDF can have $D \approx 100{,}000$
while $N$ is a few thousand) and for genomics.

#### Flaw 2 — Multicollinearity: numerically explosive inverse

**Claim.** When columns are (nearly) linearly dependent, the OLS estimator has enormous variance.

**Proof.**

1. Assume $y = Xw^* + \varepsilon$ with $\mathbb{E}[\varepsilon] = 0$, $\operatorname{Var}(\varepsilon) = \sigma^2 I$.
2. The estimator is $\hat{w} = (X^\top X)^{-1}X^\top y$. Its variance is
   $$
   \begin{aligned}
   \operatorname{Var}(\hat{w}) &= (X^\top X)^{-1} X^\top \cdot \operatorname{Var}(y) \cdot X (X^\top X)^{-1} \\
                               &= (X^\top X)^{-1} X^\top \cdot \sigma^2 I \cdot X (X^\top X)^{-1} \\
                               &= \sigma^2 (X^\top X)^{-1} X^\top X (X^\top X)^{-1} \\
                               &= \sigma^2 (X^\top X)^{-1}.
   \end{aligned}
   $$
3. Diagonalize $X^\top X = V \Lambda V^\top$ with eigenvalues $\lambda_1 \ge \dots \ge \lambda_D \ge 0$. Then
   $$
   (X^\top X)^{-1} = V \Lambda^{-1} V^\top, \quad \operatorname{Var}(\hat{w}) = \sigma^2 V \Lambda^{-1} V^\top.
   $$
   Along the eigenvector $v_i$, the variance is $\sigma^2 / \lambda_i$.
4. Near-collinearity makes some $\lambda_i \to 0^+$, so $\sigma^2 / \lambda_i \to \infty$. A tiny change in the training set
   can flip a coefficient from large positive to large negative.

**Consequence.** The model is unstable and uninterpretable, even though $X^\top X$ is technically
invertible.

#### Flaw 3 — No feature selection: no sparsity

**Claim.** OLS has no mechanism to set coefficients exactly to zero.

**Proof / reasoning.**

1. The OLS objective is smooth and unconstrained. Its solution is a deterministic function of
   $X^\top y$, and generically all coordinates are nonzero: the set of data for which some $\hat{w}_j = 0$
   exactly has measure zero.
2. There is no penalty on the magnitude or number of nonzero coefficients, so irrelevant
   features are used like any other.
3. Consequences: (a) overfitting, because noise in junk features is fitted too; (b) poor
   interpretability, because there is no notion of "which features matter".

> These three flaws motivate **regularization**: Ridge (L2) to stabilize the inverse, Lasso (L1)
> to produce exact zeros, and ElasticNet to combine both.

---

## 2. Regularization

Add a penalty on $w$ to the objective:

$$
\min_w \frac{1}{2N} \|y - Xw\|_2^2 + \operatorname{penalty}(w)
$$

The $\frac{1}{2N}$ scaling is the scikit-learn convention; it keeps the math below clean and matches
the code.

### 2.1 Ridge (L2) — fixes the singular inverse

$$
\text{Ridge:} \quad \min_w \|y - Xw\|_2^2 + \lambda \|w\|_2^2 \quad (\lambda > 0)
$$

**Derivation.**

$$
\begin{aligned}
\nabla_w &= -2X^\top y + 2X^\top Xw + 2\lambda w = 0 \\
\implies &2(X^\top X + \lambda I) w = 2X^\top y \\
\implies &(X^\top X + \lambda I) w = X^\top y \\
\implies &\hat{w}_{\text{ridge}} = (X^\top X + \lambda I)^{-1} X^\top y
\end{aligned}
$$

Why it fixes Flaw 1: if $X^\top X = V \Lambda V^\top$, then $X^\top X + \lambda I = V(\Lambda + \lambda I)V^\top$, whose eigenvalues are
$\lambda_i + \lambda$. With $\lambda > 0$ all are strictly positive, so the matrix is positive definite and
invertible even when $X^\top X$ is singular.

Why it fixes Flaw 2: the variance becomes $\sigma^2(X^\top X + \lambda I)^{-1}$, whose largest value is $\sigma^2 / \lambda$ —
bounded.

**Limitation.** Ridge shrinks coefficients but generically never makes them exactly zero. It does
not select features.

### 2.2 Lasso (L1) — produces sparsity

$$
\text{Lasso:} \quad \min_w \|y - Xw\|_2^2 + \lambda \|w\|_1, \quad \|w\|_1 = \sum_j |w_j|
$$

No closed-form solution. It does select features: many $\hat{w}_j$ become exactly zero.

### 2.3 ElasticNet — both

$$
\text{ElasticNet:} \quad f(w) = \frac{1}{2N}\|y - Xw\|_2^2 + \lambda_1\|w\|_1 + \frac{\lambda_2}{2}\|w\|_2^2
$$

with $\lambda_1 = \alpha \rho$ and $\lambda_2 = \alpha (1 - \rho)$, where $\alpha \ge 0$ is the total penalty strength and
$\rho \in [0, 1]$ is the L1 ratio (scikit-learn's `l1_ratio`).

- $\rho = 1 \implies$ Lasso.
- $\rho = 0 \implies$ Ridge.
- $\lambda_2 > 0$ makes the objective **strongly convex**, so the minimizer is **unique** (plain Lasso
  can have multiple minimizers under collinearity). $\lambda_1$ keeps the sparsity mechanism.

---

## 3. Why L1 produces exact zeros

This is the conceptual core. There are two equivalent views.

### 3.1 Geometric view: the L1 ball has vertices on the axes

The penalized problem is equivalent (Lagrangian duality) to a constrained one:

$$
\begin{aligned}
\text{Lasso:} \quad &\min_w \|y - Xw\|_2^2 \quad \text{s.t.} \quad \|w\|_1 \le t \\
\text{Ridge:} \quad &\min_w \|y - Xw\|_2^2 \quad \text{s.t.} \quad \|w\|_2^2 \le t
\end{aligned}
$$

Picture the level sets of $\|y - Xw\|_2^2$ as ellipses growing around the unconstrained OLS
solution, and the constraint set as a region around the origin. The solution is the first point
where an ellipse touches the region.

1. The L2 ball is a **smooth sphere**: the touching point is generically *not* on an axis, so no
   coordinate is exactly zero.
2. The L1 ball is a **polytope with sharp vertices on the axes** (a diamond in 2D). The first
   contact very often happens at a vertex — and at a vertex at least one coordinate is zero.

This is why L1 selects features and L2 only shrinks them. It is geometry, not magic.

### 3.2 Analytic view: the subdifferential at zero

The function $|w|$ is not differentiable at $0$. Its subdifferential is the set of slopes that
support the function at that point:

$$
\partial |w| = \begin{cases} \{ \operatorname{sign}(w) \} & \text{if } w \ne 0 \\ [-1, 1] & \text{if } w = 0 \end{cases}
$$

At $w = 0$ the derivative can be *any* value in $[-1, 1]$. That interval is exactly the "room"
that lets a coordinate stay at zero: the optimum condition at zero is an **inequality**, not an
equation.

Concretely, for one coordinate the optimality condition is

$$
v_j + \lambda_1 s = 0, \quad s \in \partial |w_j|
$$

- If $w_j \ne 0$, then $s = \operatorname{sign}(w_j)$ and $v_j$ must equal $\mp \lambda_1$ exactly.
- If $w_j = 0$, then any $s \in [-1, 1]$ works, i.e. $|v_j| \le \lambda_1$.

So a coordinate is allowed to sit at zero whenever the data's "pull" $v_j$ is smaller than the
threshold $\lambda_1$. That is the analytic form of the vertex argument above.

---

## 4. The optimization problem and KKT

### 4.1 The objective

Write the objective with the smooth and non-smooth parts separated:

$$
f(w) = \underbrace{\frac{1}{2N}\|y - Xw - b\mathbf{1}\|_2^2 + \frac{\lambda_2}{2}\|w\|_2^2}_{\text{smooth, differentiable}} + \underbrace{\lambda_1 \sum_j |w_j|}_{\text{non-smooth}}
$$

$f$ is convex but not differentiable wherever some $w_j = 0$. A minimizer $w^*$ is characterized
by the KKT condition

$$
0 \in \nabla_{\text{smooth}} f(w^*) + \lambda_1 \partial \|w^*\|_1
$$

We now compute the smooth gradient explicitly.

### 4.2 Expanding the smooth part

Let $e = y - Xw - b\mathbf{1}$. Then

$$
\|e\|_2^2 = e^\top e = (y - Xw - b\mathbf{1})^\top(y - Xw - b\mathbf{1})
$$

Multiply the four blocks term by term:

$$
\begin{aligned}
\|e\|_2^2 &= y^\top y - y^\top Xw - (Xw)^\top y - b y^\top \mathbf{1} - b \mathbf{1}^\top y \\
           &\quad + (Xw)^\top(Xw) + b(Xw)^\top \mathbf{1} + b\mathbf{1}^\top(Xw) + b^2 \mathbf{1}^\top \mathbf{1}
\end{aligned}
$$

Now simplify each group:

1. $y^\top Xw$ and $(Xw)^\top y$ are equal scalars, so they sum to $2 w^\top X^\top y$.
2. $b y^\top \mathbf{1}$ and $b \mathbf{1}^\top y$ are equal, so they sum to $2b \mathbf{1}^\top y$.
3. $b (Xw)^\top \mathbf{1}$ and $b \mathbf{1}^\top (Xw)$ are equal, so they sum to $2b \mathbf{1}^\top Xw$.
4. $\mathbf{1}^\top \mathbf{1} = N$, so the last term is $N b^2$.

Therefore

$$
\|e\|_2^2 = y^\top y - 2 w^\top X^\top y - 2b \mathbf{1}^\top y + w^\top X^\top Xw + 2b \mathbf{1}^\top Xw + N b^2
$$

and the smooth objective is

$$
S(w) = \frac{1}{2N} \left[ y^\top y - 2 w^\top X^\top y - 2b \mathbf{1}^\top y + w^\top X^\top Xw + 2b \mathbf{1}^\top Xw + N b^2 \right] + \frac{\lambda_2}{2} w^\top w
$$

### 4.3 The gradient, coordinate by coordinate

Differentiate $S$ with respect to $w_j$. Terms that do not contain $w_j$ vanish:

$$
\begin{aligned}
\frac{\partial}{\partial w_j} (w^\top X^\top y)   &= (X^\top y)_j \\
\frac{\partial}{\partial w_j} (w^\top X^\top Xw)  &= 2 (X^\top Xw)_j \\
\frac{\partial}{\partial w_j} (\mathbf{1}^\top Xw)    &= (X^\top \mathbf{1})_j \\
\frac{\partial}{\partial w_j} (w^\top w)     &= 2 w_j
\end{aligned}
$$

Substitute:

$$
\begin{aligned}
\frac{\partial S}{\partial w_j} &= \frac{1}{2N} \left[ -2(X^\top y)_j + 2(X^\top Xw)_j + 2b(X^\top \mathbf{1})_j \right] + \frac{\lambda_2}{2} \cdot 2w_j \\
                                  &= \frac{1}{N} \left[ -(X^\top y)_j + (X^\top Xw)_j + b(X^\top \mathbf{1})_j \right] + \lambda_2 w_j
\end{aligned}
$$

Factor the bracket:

$$
\begin{aligned}
-(X^\top y)_j + (X^\top Xw)_j + b(X^\top \mathbf{1})_j &= [X^\top(Xw + b\mathbf{1} - y)]_j \\
                                                          &= -[X^\top r]_j \quad \text{with } r = y - Xw - b\mathbf{1}.
\end{aligned}
$$

Hence

$$
v_j := \frac{\partial S}{\partial w_j} = -\frac{1}{N}(X^\top r)_j + \lambda_2 w_j
$$

For **centered data** the intercept is handled separately and $\mathbf{1}^\top X = 0$, so the $b$ term
disappears from the $w$-gradient; the formula above is still the one used with the centered
residual.

### 4.4 Subdifferential and KKT

The non-smooth part is $\lambda_1 \sum_j |w_j|$, which separates over coordinates:

$$
\frac{\partial}{\partial w_j} (\lambda_1 |w_j|) = \lambda_1 \partial |w_j|
$$

with

$$
\partial |w_j| = \begin{cases} \{ \operatorname{sign}(w_j) \} & \text{if } w_j \ne 0 \\ [-1, 1] & \text{if } w_j = 0 \end{cases}
$$

The KKT condition $0 \in \nabla_{\text{smooth}} f + \lambda_1 \partial \|w\|_1$ is therefore, coordinate by coordinate,

$$
0 \in v_j + \lambda_1 \partial |w_j|
$$

### 4.5 Reading the three cases

**Case $w_j > 0$.** Then $\partial |w_j| = \{+1\}$, so

$$
0 = v_j + \lambda_1 \cdot 1 \implies v_j = -\lambda_1
$$

**Case $w_j < 0$.** Then $\partial |w_j| = \{-1\}$, so

$$
0 = v_j + \lambda_1 \cdot (-1) \implies v_j = +\lambda_1
$$

**Case $w_j = 0$.** Then $\partial |w_j| = [-1, 1]$, so there exists $s \in [-1, 1]$ with

$$
0 = v_j + \lambda_1 s \implies v_j = -\lambda_1 s \implies |v_j| \le \lambda_1
$$

Combining:

$$
\begin{aligned}
w_j > 0 &: \quad v_j = -\lambda_1 \\
w_j < 0 &: \quad v_j = +\lambda_1 \\
w_j = 0 &: \quad |v_j| \le \lambda_1
\end{aligned}
$$

**Interpretation.** The quantity $v_j$ is the "pull" of the data on coordinate $j$. If that pull
is smaller in magnitude than $\lambda_1$, the coordinate is allowed to stay at exactly zero. This is the
selection mechanism, now derived rather than asserted.

---

## 5. Soft-thresholding

Fix all coordinates except $w_j$. We solve the one-dimensional subproblem exactly. This is the
building block of coordinate descent.

### 5.1 The coordinate subproblem

Let $r_{-j}$ be the residual with the current $w_j$ removed:

$$
r_{-j} = y - \sum_{k \ne j} x_k w_k - b\mathbf{1}
$$

Then the full residual is $y - Xw - b\mathbf{1} = r_{-j} - x_j w_j$, so the objective as a function of the
single variable $w_j$ is

$$
g(w_j) = \frac{1}{2N} \|r_{-j} - x_j w_j\|_2^2 + \lambda_1 |w_j| + \frac{\lambda_2}{2} w_j^2 + \text{const}
$$

### 5.2 Expanding the squared term

Expand the norm:

$$
\begin{aligned}
\|r_{-j} - x_j w_j\|_2^2 &= (r_{-j} - x_j w_j)^\top(r_{-j} - x_j w_j) \\
                        &= r_{-j}^\top r_{-j} - 2 w_j (x_j^\top r_{-j}) + w_j^2 (x_j^\top x_j) \\
                        &= \|r_{-j}\|_2^2 - 2 w_j (x_j^\top r_{-j}) + w_j^2 (x_j^\top x_j)
\end{aligned}
$$

Define

$$
\begin{aligned}
z_j &= \|x_j\|_2^2 = x_j^\top x_j \\
\rho_j &= x_j^\top r_{-j}
\end{aligned}
$$

Then

$$
\|r_{-j} - x_j w_j\|_2^2 = \|r_{-j}\|_2^2 - 2 \rho_j w_j + z_j w_j^2
$$

and, dropping the constant $\|r_{-j}\|_2^2$,

$$
g(w_j) = \frac{1}{2N} (z_j w_j^2 - 2 \rho_j w_j) + \lambda_1 |w_j| + \frac{\lambda_2}{2} w_j^2 + \text{const}
$$

### 5.3 Case $w_j > 0$

On this region $|w_j| = w_j$, so $g$ is differentiable. Differentiate:

$$
\begin{aligned}
g'(w_j) &= \frac{1}{2N} (2 z_j w_j - 2 \rho_j) + \lambda_1 + \lambda_2 w_j \\
        &= \frac{1}{N} (z_j w_j - \rho_j) + \lambda_1 + \lambda_2 w_j
\end{aligned}
$$

Set to zero:

$$
\frac{1}{N}(z_j w_j - \rho_j) + \lambda_1 + \lambda_2 w_j = 0
$$

Multiply the whole equation by $N$:

$$
z_j w_j - \rho_j + N\lambda_1 + N\lambda_2 w_j = 0
$$

Group the $w_j$ terms:

$$
(z_j + N\lambda_2) w_j = \rho_j - N\lambda_1
$$

Therefore

$$
w_j = \frac{\rho_j - N\lambda_1}{z_j + N\lambda_2}
$$

This case is self-consistent only when the result is positive, i.e. $\rho_j > N\lambda_1$.

### 5.4 Case $w_j < 0$

On this region $|w_j| = -w_j$, so the derivative of the penalty is $-\lambda_1$:

$$
g'(w_j) = \frac{1}{N}(z_j w_j - \rho_j) - \lambda_1 + \lambda_2 w_j
$$

Set to zero and multiply by $N$:

$$
\begin{aligned}
&z_j w_j - \rho_j - N\lambda_1 + N\lambda_2 w_j = 0 \\
\implies &(z_j + N\lambda_2) w_j = \rho_j + N\lambda_1 \\
\implies &w_j = \frac{\rho_j + N\lambda_1}{z_j + N\lambda_2}
\end{aligned}
$$

This case is self-consistent only when the result is negative, i.e. $\rho_j < -N\lambda_1$.

### 5.5 Case $w_j = 0$

At $w_j = 0$ the smooth part of $g$ has derivative

$$
g'_{\text{smooth}}(0) = \frac{1}{N}(z_j \cdot 0 - \rho_j) = -\frac{\rho_j}{N}
$$

and the penalty contributes the interval $\lambda_1[-1, 1]$. The condition for zero to be optimal is

$$
0 \in -\frac{\rho_j}{N} + \lambda_1[-1, 1]
$$

which means there exists $s \in [-1, 1]$ with

$$
0 = -\frac{\rho_j}{N} + \lambda_1 s \implies \frac{\rho_j}{N} = \lambda_1 s \implies \frac{|\rho_j|}{N} \le \lambda_1 \implies |\rho_j| \le N\lambda_1
$$

So the zero solution is admissible exactly when $|\rho_j| \le N\lambda_1$.

### 5.6 Assembling the operator

Combining the three cases:

$$
w_j = \begin{cases}
\dfrac{\rho_j - N\lambda_1}{z_j + N\lambda_2} & \text{if } \rho_j > N\lambda_1 \\[8pt]
\dfrac{\rho_j + N\lambda_1}{z_j + N\lambda_2} & \text{if } \rho_j < -N\lambda_1 \\[8pt]
0 & \text{if } |\rho_j| \le N\lambda_1
\end{cases}
$$

Define the **soft-thresholding operator**

$$
S(z, \gamma) = \operatorname{sign}(z) \cdot \max(|z| - \gamma, 0)
$$

Its piecewise form is

$$
S(z, \gamma) = \begin{cases}
z - \gamma & \text{if } z > \gamma \\
z + \gamma & \text{if } z < -\gamma \\
0          & \text{if } |z| \le \gamma
\end{cases}
$$

Comparing the two piecewise definitions gives the compact update

$$
w_j^* = \frac{S(\rho_j, N\lambda_1)}{z_j + N\lambda_2}
$$

**Sanity check.** At $\rho_j = N\lambda_1$ the first branch gives $0$, and at $\rho_j = -N\lambda_1$ the second
branch gives $0$. The function is continuous at both boundaries.

### 5.7 Properties

- **Shrinkage:** every nonzero coefficient is pulled toward zero by exactly $N\lambda_1$ in the
  numerator.
- **Selection:** any $|\rho_j| \le N\lambda_1$ becomes exactly zero, not merely small.
- **Odd symmetry:** $S(-z, \gamma) = -S(z, \gamma)$.
- **No closed form for the whole vector:** this only solves one coordinate at a time, which is
  precisely why we iterate.

---

## 6. Coordinate descent

### 6.1 Why one coordinate at a time

The full problem has no closed-form solution because of the non-smooth L1 term. But §5 solved
the **one-coordinate** problem exactly. Coordinate descent turns the hard vector problem into a
sequence of easy scalar problems:

> Freeze every coordinate except $w_j$; solve that one exactly; move to the next; repeat.

### 6.2 The partial-residual identity

We want to avoid recomputing $\rho_j = x_j^\top r_{-j}$ from scratch every time. Relate $r_{-j}$ to the full
residual $r$:

$$
\begin{aligned}
r &= y - Xw - b\mathbf{1} \\
  &= y - \sum_k x_k w_k - b\mathbf{1} \\
  &= y - \sum_{k \ne j} x_k w_k - x_j w_j - b\mathbf{1} \\
  &= r_{-j} - x_j w_j
\end{aligned}
$$

Rearrange:

$$
r_{-j} = r + x_j w_j
$$

Therefore

$$
\begin{aligned}
\rho_j &= x_j^\top r_{-j} \\
        &= x_j^\top(r + x_j w_j) \\
        &= x_j^\top r + w_j (x_j^\top x_j) \\
        &= x_j^\top r + w_j z_j
\end{aligned}
$$

So a single sparse dot product $x_j^\top r$ plus a scalar correction gives $\rho_j$.

### 6.3 The update

Putting §5.6 and §6.2 together:

$$
\begin{aligned}
z_j &= \|x_j\|_2^2 \\
\rho_j &= x_j^\top r + w_j z_j \\
w_j &\leftarrow \frac{S(\rho_j, N\lambda_1)}{z_j + N\lambda_2}
\end{aligned}
$$

### 6.4 Residual maintenance

After updating $w_j$ by $\Delta = w_j^{\text{new}} - w_j$, update the residual instead of recomputing it.

**Proof.**

$$
\begin{aligned}
r_{\text{new}} &= y - Xw_{\text{new}} - b\mathbf{1} \\
               &= y - X(w + \Delta e_j) - b\mathbf{1} \\
               &= y - Xw - b\mathbf{1} - \Delta x_j \\
               &= r - \Delta x_j
\end{aligned}
$$

$\blacksquare$

For a CSC matrix, $x_j$ has $\operatorname{nnz}_j$ nonzeros, so this update costs $\mathcal{O}(\operatorname{nnz}_j)$.

### 6.5 Intercept and sparse centering

With an intercept, the model is $\hat{y} = Xw + b$. The intercept is not penalized. At the optimum,
for centered data,

$$
b = \bar{y} - \bar{x}^\top w
$$

where $\bar{x} = \frac{1}{N}X^\top \mathbf{1}$ and $\bar{y} = \frac{1}{N}\mathbf{1}^\top y$.

Naively centering $X$ destroys sparsity. Instead, keep the **uncentered residual**
$r_{\text{raw}} = y - Xw$ and the scalar intercept $b$ separately. We now derive the update used in the
code.

**Step 1 — relation between centered and raw residual.** With $X_c = X - \mathbf{1}\bar{x}^\top$ and
$y_c = y - \bar{y}\mathbf{1}$,

$$
\begin{aligned}
r_c &= y_c - X_c w \\
    &= (y - \bar{y}\mathbf{1}) - (X - \mathbf{1}\bar{x}^\top)w \\
    &= y - Xw - (\bar{y} - \bar{x}^\top w)\mathbf{1} \\
    &= r_{\text{raw}} - b\mathbf{1}
\end{aligned}
$$

so the centered residual is the raw residual shifted by the scalar intercept.

**Step 2 — centered column norm.** For column $j$:

$$
\begin{aligned}
z_j^c &= \|x_j - \bar{x}_j \mathbf{1}\|_2^2 \\
      &= (x_j - \bar{x}_j \mathbf{1})^\top(x_j - \bar{x}_j \mathbf{1}) \\
      &= x_j^\top x_j - 2\bar{x}_j(x_j^\top \mathbf{1}) + \bar{x}_j^2(\mathbf{1}^\top \mathbf{1}) \\
      &= \|x_j\|_2^2 - 2\bar{x}_j(N \bar{x}_j) + N \bar{x}_j^2 \\
      &= \|x_j\|_2^2 - N \bar{x}_j^2
\end{aligned}
$$

**Step 3 — centered partial residual.** We need
$\rho_j^c = x_{c,j}^\top r_{c,-j}$, where $r_{c,-j} = r_c + x_{c,j} w_j$ (same argument as §6.2).
Thus

$$
\begin{aligned}
\rho_j^c &= x_{c,j}^\top r_c + w_j \|x_{c,j}\|_2^2 \\
         &= x_{c,j}^\top r_c + w_j z_j^c
\end{aligned}
$$

Now expand the first term:

$$
\begin{aligned}
x_{c,j}^\top r_c &= (x_j - \bar{x}_j \mathbf{1})^\top(r_{\text{raw}} - b\mathbf{1}) \\
                 &= x_j^\top r_{\text{raw}} - b(x_j^\top \mathbf{1}) - \bar{x}_j(\mathbf{1}^\top r_{\text{raw}}) + b \bar{x}_j(\mathbf{1}^\top \mathbf{1}) \\
                 &= x_j^\top r_{\text{raw}} - b N \bar{x}_j - \bar{x}_j \sum_i (r_{\text{raw}})_i + b N \bar{x}_j
\end{aligned}
$$

The last two terms cancel if $\sum_i (r_{\text{raw}})_i = N b$. Check:

$$
\sum_i (r_{\text{raw}})_i = \mathbf{1}^\top(y - Xw) = \mathbf{1}^\top y - \mathbf{1}^\top Xw = N\bar{y} - N \bar{x}^\top w = N(\bar{y} - \bar{x}^\top w) = N b
$$

So indeed

$$
x_{c,j}^\top r_c = x_j^\top r_{\text{raw}} - b N \bar{x}_j
$$

and therefore

$$
\rho_j^c = x_j^\top r_{\text{raw}} - b N \bar{x}_j + w_j z_j^c
$$

**Step 4 — intercept update.** Since $b = \bar{y} - \bar{x}^\top w$, changing $w_j$ by $\Delta$ changes $b$ by
$-\bar{x}_j \Delta$:

$$
b \leftarrow b - \bar{x}_j \Delta
$$

**Step 5 — initialization.** Pick $w = w_0$ and set

$$
\begin{aligned}
b &= \bar{y} - \bar{x}^\top w_0 \\
r_{\text{raw}} &= y - Xw_0
\end{aligned}
$$

so that $\sum_i (r_{\text{raw}} - b)_i = 0$, i.e. the centered residual really is centered.

This gives the code's sparse update, with every quantity computable in $\mathcal{O}(\operatorname{nnz}_j)$.

### 6.6 Convergence argument

- Each coordinate step computes the **exact** minimizer along that coordinate, so $f$ never
  increases.
- $f$ is convex and bounded below, so the sequence $f(w^k)$ converges.
- At any limit point, no coordinate can be improved, which is exactly the KKT condition of §4.
  Hence every limit point is a global minimizer.
- If $\lambda_2 > 0$, $f$ is strongly convex and the minimizer is unique, so the whole sequence
  converges to it.

(A rigorous statement is the classical result for coordinate descent on convex, coordinate-wise
minimizable objectives; see Tseng, 2001.)

### 6.7 Complexity

- One coordinate: $\mathcal{O}(\operatorname{nnz}_j)$ for $x_j^\top r$ plus $\mathcal{O}(\operatorname{nnz}_j)$ for $r \leftarrow r - \Delta x_j$.
- One sweep: $\mathcal{O}(\operatorname{nnz}(X))$.
- Memory: $\mathcal{O}(\operatorname{nnz}(X))$ for CSC plus $\mathcal{O}(N + D)$.

---

## 7. Stopping criteria

Never stop because "$w$ changed little between sweeps" — that does not measure distance to the
optimum. Use the KKT conditions.

### 7.1 KKT violation

From §4.3, the smooth gradient is

$$
v_j = -\frac{1}{N}(X^\top r)_j + \lambda_2 w_j
$$

with $r$ the current centered residual. Since the centered residual satisfies $\sum_i r_i = 0$,
we have $x_{c,j}^\top r = x_j^\top r - \bar{x}_j \sum_i r_i = x_j^\top r$, so the raw matrix-vector product is enough.

The per-coordinate violation follows directly from §4.5:

$$
\operatorname{viol}_j = \begin{cases}
|v_j + \lambda_1| & \text{if } w_j > 0 \\
|v_j - \lambda_1| & \text{if } w_j < 0 \\
\max(|v_j| - \lambda_1, 0) & \text{if } w_j = 0
\end{cases}
$$

Stop when $\max_j \operatorname{viol}_j \le \text{tol}$. At the optimum every $\operatorname{viol}_j$ is zero.

### 7.2 Duality: Fenchel setup

scikit-learn uses a relative duality gap, so a fair comparison must be aware of the difference.
We derive the gap for the centered problem.

Write the problem as $\min_w g(Xw) + h(w)$ with

$$
g(u) = \frac{1}{2N}\|y - u\|_2^2, \quad h(w) = \lambda_1 \|w\|_1 + \frac{\lambda_2}{2}\|w\|_2^2
$$

The Fenchel dual is

$$
\max_\theta -g^*(\theta) - h^*(-X^\top \theta)
$$

where $g^*$ and $h^*$ are the convex conjugates.

### 7.3 Conjugate of $g$

By definition,

$$
g^*(\theta) = \sup_u \left[ \theta^\top u - \frac{1}{2N}\|y - u\|_2^2 \right]
$$

Substitute $u = y - v$ (so $v = y - u$):

$$
\begin{aligned}
g^*(\theta) &= \sup_v \left[ \theta^\top(y - v) - \frac{1}{2N}\|v\|_2^2 \right] \\
             &= \theta^\top y + \sup_v \left[ -\theta^\top v - \frac{1}{2N}\|v\|_2^2 \right]
\end{aligned}
$$

The inner supremum is a concave quadratic in $v$. Its gradient is $-\theta - v/N$; setting it to zero
gives $v = -N\theta$. The value there is

$$
-\theta^\top(-N\theta) - \frac{1}{2N}\|-N\theta\|_2^2 = N\|\theta\|_2^2 - \frac{N^2}{2N}\|\theta\|_2^2 = \frac{N}{2}\|\theta\|_2^2
$$

Therefore

$$
g^*(\theta) = \theta^\top y + \frac{N}{2}\|\theta\|_2^2
$$

### 7.4 Conjugate of $h$

$h$ separates over coordinates:

$$
h^*(z) = \sum_j \sup_{w_j} \left[ z_j w_j - \lambda_1|w_j| - \frac{\lambda_2}{2}w_j^2 \right]
$$

For one coordinate, assume $\lambda_2 > 0$. Consider $w_j > 0$, so $|w_j| = w_j$. The inner function is

$$
z_j w_j - \lambda_1 w_j - \frac{\lambda_2}{2}w_j^2
$$

Its derivative is $z_j - \lambda_1 - \lambda_2 w_j$; setting it to zero gives

$$
w_j = \frac{z_j - \lambda_1}{\lambda_2}
$$

which is positive exactly when $z_j > \lambda_1$. Substituting back:

$$
\begin{aligned}
z_j w_j - \lambda_1 w_j - \frac{\lambda_2}{2}w_j^2 &= (z_j - \lambda_1)w_j - \frac{\lambda_2}{2}w_j^2 \\
                                                    &= \frac{(z_j - \lambda_1)^2}{\lambda_2} - \frac{(z_j - \lambda_1)^2}{2\lambda_2} \\
                                                    &= \frac{(z_j - \lambda_1)^2}{2\lambda_2}
\end{aligned}
$$

The case $w_j < 0$ is symmetric and gives $\frac{(|z_j| - \lambda_1)^2}{2\lambda_2}$ when $z_j < -\lambda_1$. When
$|z_j| \le \lambda_1$, the optimum is $w_j = 0$ and the value is $0$. Combining:

$$
h^*(z) = \sum_j \frac{1}{2\lambda_2} (|z_j| - \lambda_1)_+^2, \quad (a)_+ = \max(a, 0)
$$

For $\lambda_2 = 0$ (plain Lasso), the inner supremum is finite only if $|z_j| \le \lambda_1$, so

$$
h^*(z) = \begin{cases} 0 & \text{if } \|z\|_\infty \le \lambda_1 \\ +\infty & \text{otherwise} \end{cases}
$$

### 7.5 Evaluating the gap at $\theta = -r/N$

The natural dual point is the one induced by the primal residual. Since
$\nabla g(u) = (u - y)/N$, at $u = Xw$ we get

$$
\theta = \frac{Xw - y}{N} = -\frac{r}{N}
$$

Now compute each piece:

$$
\begin{aligned}
\theta^\top y &= -\frac{r^\top y}{N} \\
\frac{N}{2}\|\theta\|_2^2 &= \frac{N}{2}\frac{\|r\|_2^2}{N^2} = \frac{\|r\|_2^2}{2N} \\
-X^\top \theta &= \frac{X^\top r}{N} =: z
\end{aligned}
$$

For $\lambda_2 > 0$, the dual value is

$$
\begin{aligned}
D(\theta) &= -g^*(\theta) - h^*(-X^\top \theta) \\
          &= -\theta^\top y - \frac{N}{2}\|\theta\|_2^2 - h^*(z) \\
          &= \frac{r^\top y}{N} - \frac{\|r\|_2^2}{2N} - h^*(z)
\end{aligned}
$$

and the primal value is

$$
P(w) = \frac{\|r\|_2^2}{2N} + \lambda_1\|w\|_1 + \frac{\lambda_2}{2}\|w\|_2^2
$$

Subtract:

$$
\begin{aligned}
\text{gap} &= P(w) - D(\theta) \\
           &= \frac{\|r\|_2^2}{2N} + \lambda_1\|w\|_1 + \frac{\lambda_2}{2}\|w\|_2^2 - \frac{r^\top y}{N} + \frac{\|r\|_2^2}{2N} + h^*(z) \\
           &= \frac{\|r\|_2^2}{N} - \frac{r^\top y}{N} + \lambda_1\|w\|_1 + \frac{\lambda_2}{2}\|w\|_2^2 + h^*(z)
\end{aligned}
$$

This is the quantity the code computes. Weak duality guarantees $\text{gap} \ge 0$; it reaches $0$ at the
optimum.

### 7.6 Lasso special case ($\lambda_2 = 0$)

When $\lambda_2 = 0$, $h^*$ is an indicator function, so $\theta$ must be dual feasible:

$$
\|X^\top \theta\|_\infty \le \lambda_1
$$

At a non-optimal iterate this may fail. To keep a valid bound, rescale

$$
\theta \leftarrow \theta \cdot \min\left(1, \frac{\lambda_1}{\|X^\top \theta\|_\infty}\right)
$$

and evaluate $D(\theta)$ with the rescaled $\theta$. This is what the code does.

### 7.7 Why tolerance semantics matter

`tol` is not a universal quantity:

- Our `tol` = maximum KKT violation.
- scikit-learn's `tol` = relative duality gap.

Two solvers "with `tol = 1e-9`" can therefore stop at different accuracies. Any benchmark must
either report the final KKT violation for both (as `bench/bench.py` does) or compare
time-to-the-same-criterion. Otherwise the timing comparison is meaningless.

---

## 8. Regularization path and warm start

In practice we solve not one $\alpha$ but a **sequence** $\alpha_1 > \alpha_2 > \dots$, the regularization path.

### 8.1 The largest useful alpha

At $w = 0$ the residual is $r = y_c$ (centered $y$). From §4.3,

$$
v_j = -\frac{1}{N}(X^\top y_c)_j
$$

KKT at $w_j = 0$ requires $|v_j| \le \lambda_1$ for every $j$:

$$
\frac{1}{N}|(X^\top y_c)_j| \le \lambda_1 \quad \text{for all } j
$$

The smallest $\lambda_1$ that satisfies all $D$ inequalities is the largest left-hand side:

$$
\lambda_1^{\max} = \frac{\max_j |x_j^\top y_c|}{N}
$$

Because the centered $y_c$ sums to zero, $x_{c,j}^\top y_c = x_j^\top y_c$, so the raw dot product is
correct. Finally, since $\lambda_1 = \alpha \rho$,

$$
\alpha_{\max} = \frac{\lambda_1^{\max}}{\rho}
$$

For $\rho = 0$ (Ridge) the solution is never exactly zero, so $\alpha_{\max} = \infty$ and the path must start
from a chosen finite $\alpha$ instead.

### 8.2 The path

Choose a descending geometric grid:

$$
\alpha_k = \alpha_{\max} \cdot \varepsilon^{k / (n_{\text{alphas}} - 1)}, \quad k = 0, \dots, n_{\text{alphas}} - 1
$$

with $\varepsilon$ a small fraction (e.g. $10^{-3}$). For each $\alpha_k$, solve with the previous solution as
the starting point.

### 8.3 Why warm start helps

As $\alpha$ decreases continuously, the minimizer moves continuously (strict convexity when
$\lambda_2 > 0$). Starting from the previous minimizer therefore places the solver very close to the new
one, so far fewer sweeps are needed. Warm start is often a larger speed-up than low-level
optimizations, and it is the foundation for active-set / strong-rule screening:

- **Strong rules** (Tibshirani et al., 2012) use the previous solution to predict which
  coefficients can be safely set to zero at the next $\alpha$, shrinking the working set.
- **Active set** methods solve on a small subset of coordinates and expand it only when the KKT
  conditions are violated.

Both are natural extensions of the loop in §6; neither changes the underlying math.

---

## 9. From theory to code

| Concept | Function / component | File |
|---|---|---|
| Objective $f(w)$ | `objective` | `reference.py` |
| Soft-threshold $S(z, \gamma)$ | `soft_threshold` | `reference.py` |
| Column statistics and centering | `_column_stats` | `reference.py` |
| Coordinate descent | `coordinate_descent`, `_cd_dense`, `_cd_sparse` | `reference.py` |
| KKT violation | `kkt_violation` | `reference.py` |
| Duality gap | `duality_gap` | `reference.py` |
| $\alpha_{\max}$ | `alpha_max` | `reference.py` |
| Warm-started path | `fit_path` | `reference.py` |
| C++ loop over CSC | `coordinate_descent_csc` | `_core.cpp` |
| Python $\to$ C++ wrapper | `coordinate_descent_cpp` | `backend.py` |

The NumPy implementation is deliberately readable and serves as the correctness oracle. The C++
implementation mirrors it exactly; the test suite asserts they agree to machine precision.

---

## 10. Extension: the GLM family

The update in §6 does not actually depend on the loss being squared error. It only needs the
gradient and curvature of the loss. We now derive the general form.

### 10.1 Second-order expansion

Let the loss be $\ell(\hat{y}_i, y_i)$ with $\hat{y} = Xw + b$. Define

$$
g_i = \frac{\partial \ell(\hat{y}_i, y_i)}{\partial \hat{y}_i}, \quad h_i = \frac{\partial^2 \ell(\hat{y}_i, y_i)}{\partial \hat{y}_i^2}
$$

Around the current prediction, the second-order Taylor expansion is

$$
\ell(\hat{y}_i + \delta_i, y_i) \approx \ell(\hat{y}_i, y_i) + g_i \delta_i + \frac{1}{2} h_i \delta_i^2
$$

### 10.2 Coordinate subproblem

Change only $w_j$ by $\Delta = u - w_j$, so $\delta_i = x_{ij} \Delta$. Dropping constants:

$$
\frac{1}{N} \sum_i \left[ g_i x_{ij} (u - w_j) + \frac{1}{2} h_i x_{ij}^2 (u - w_j)^2 \right] + \lambda_1 |u| + \frac{\lambda_2}{2} u^2
$$

Define the aggregated curvature and gradient:

$$
A_j = \frac{1}{N} \sum_i h_i x_{ij}^2, \quad B_j = \frac{1}{N} \sum_i g_i x_{ij}
$$

Expand the quadratic in $u$:

$$
\begin{aligned}
\frac{1}{N} \sum_i g_i x_{ij} (u - w_j) &= B_j u - B_j w_j \\
\frac{1}{N} \sum_i \frac{1}{2} h_i x_{ij}^2 (u - w_j)^2 &= \frac{1}{2}A_j (u^2 - 2w_j u + w_j^2) \\
                                                         &= \frac{1}{2}A_j u^2 - A_j w_j u + \text{const}
\end{aligned}
$$

Adding the L2 term:

$$
\frac{1}{2}(A_j + \lambda_2) u^2 - (A_j w_j - B_j) u + \lambda_1 |u| + \text{const}
$$

This has the same form as §5.3–5.5, with

$$
c = A_j + \lambda_2, \quad d = A_j w_j - B_j
$$

and its minimizer is

$$
u = \frac{S(d, \lambda_1)}{c} = \frac{S(A_j w_j - B_j, \lambda_1)}{A_j + \lambda_2}
$$

### 10.3 Working residual form

Define $r_i = -g_i$ (the negative gradient, a "working residual"). Then
$-B_j = \frac{1}{N} \sum_i r_i x_{ij}$, so

$$
\begin{aligned}
\rho_j &:= A_j w_j + \frac{1}{N} \sum_i r_i x_{ij} \\
u &= \frac{S(\rho_j, \lambda_1)}{A_j + \lambda_2}
\end{aligned}
$$

### 10.4 Recovering the Gaussian case

For squared error $\ell(\hat{y}, y) = \frac{1}{2}(\hat{y} - y)^2$:

$$
\begin{aligned}
g_i &= \hat{y}_i - y_i, \quad h_i = 1, \quad r_i = y_i - \hat{y}_i \\
A_j &= \frac{1}{N}\|x_j\|_2^2, \quad \rho_j = \frac{1}{N}x_j^\top r + w_j \frac{1}{N}\|x_j\|_2^2
\end{aligned}
$$

Substituting and multiplying numerator and denominator by $N$:

$$
u = \frac{S(x_j^\top r + w_j\|x_j\|_2^2, N\lambda_1)}{\|x_j\|_2^2 + N\lambda_2}
$$

which is exactly §6.3. The general form therefore covers Logistic, Poisson, Gamma, and other
GLMs by changing only $g_i$ and $h_i$.

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
