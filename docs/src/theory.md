# Theory

## Basis expansion of the Schrödinger equation

We are going to solve the Schrödinger equation

```math
\hat{H}|\psi\rangle = \epsilon|\psi\rangle,
```

where $\hat{H}$ is the Hamiltonian of a quantum few-body system, $|\psi\rangle$ and $\epsilon$ are the eigenfunction and the eigenvalue to be found.

We shall expand the wave-function $|\psi\rangle$ in terms of a set of basis functions $|i\rangle$ for $i = 1 \ldots n$,

```math
|\psi\rangle = \sum_{i=1}^{n} c_i |i\rangle.
```

Inserting the expansion into the Schrödinger equation and multiplying from the left with $\langle k|$ for $1 \leq k \leq n$ gives

```math
\sum_{i=1}^{n} \langle k|\hat{H}|i\rangle c_i = \epsilon \sum_{i=1}^{n} \langle k|i\rangle c_i.
```

Or, in the matrix notation

```math
Hc = \epsilon Nc,
```

where $H$ and $N$ are correspondingly the Hamiltonian and the overlap matrices with the matrix elements

```math
H_{ki} = \langle k|\hat{H}|i\rangle, \quad N_{ki} = \langle k|i\rangle
```

## Gaussians as basis functions

We shall use the so-called Correlated Gaussians (or Explicitly Correlated Gaussians) as the basis functions. For a system of $N$ particles with coordinates $\vec{r}_i$, $i = 1 \ldots N$, the Correlated Gaussian is defined as

```math
g(\vec{r}_1, \ldots, \vec{r}_N) = \exp \left( - \sum_{i,j=1}^{N} A_{ij}\vec{r}_i \cdot \vec{r}_j - \sum_{i=1}^{N} \vec{s}_i \cdot \vec{r}_i \right),
```

where $\vec{r}_i \cdot \vec{r}_j$ denotes the dot-product of the two vectors; and where $A$, a symmetric positive-defined matrix, and $\vec{s}_i$, $i=1,\ldots,N$, the shift-vectors, are (cleverly chosen) parameters of the Gaussian.

In matrix notation,

```math
g(\vec{r}) = \exp \left( -\vec{r}^T A \vec{r} + \vec{s}^T \vec{r} \right),
```

where $\vec{r}$ is the column of the coordinates $\vec{r}_i$ and $\vec{s}$ is the column of the shift-vectors $\vec{s}_i$,

```math
\vec{r} =
\begin{pmatrix}
\vec{r}_1 \\
\vdots \\
\vec{r}_N
\end{pmatrix}, \quad
\vec{s} =
\begin{pmatrix}
\vec{s}_1 \\
\vdots \\
\vec{s}_N
\end{pmatrix},
```

and

```math
\vec{r}^T A \vec{r} + \vec{s}^T \vec{r} = \sum_{i,j} \vec{r}_i \cdot A_{ij}\vec{r}_j + \sum_i \vec{s}_i \cdot \vec{r}_i.
```

## Stochastic basis construction

The simplest strategy for choosing the Gaussian parameters is **stochastic
greedy search** (`solve_ECG`): candidate basis functions are generated from
quasi-random sequences (Halton, Sobol) and accepted one at a time if they
reduce the lowest eigenvalue.  This is fast and robust, but the quality of the
final basis depends on the sampling distribution and the number of accepted
functions.

## Variational parameter optimisation

A more systematic approach is to treat all parameters of all basis functions
simultaneously as a continuous optimisation problem (`solve_ECG_variational`).

### The variational principle

By the Rayleigh-Ritz variational principle, the lowest generalised eigenvalue
$\lambda_{\min}$ of $Hc = \lambda S c$ satisfies

```math
\lambda_{\min} \;\geq\; E_0,
```

where $E_0$ is the exact ground-state energy.  Equality holds when the
parameter space is large enough to contain the true ground state.  Therefore
minimising $\lambda_{\min}$ over the Gaussian parameters is a rigorous
upper-bound approach: the optimum is approached monotonically from above.

### Parameterisation

Every $n_{\text{dim}} \times n_{\text{dim}}$ positive-definite matrix $A$ is
written as $A = L L^T$ where $L$ is lower-triangular with positive diagonal.
Rather than optimising $L$ directly (which requires inequality constraints),
the diagonal entries are reparameterised as $L_{ii} = \exp(\theta_{ii})$, so
that the full parameter vector $\theta$ is unconstrained:

```math
L_{ij}(\theta) =
\begin{cases}
e^{\theta_{ij}} & i = j, \\
\theta_{ij} & i > j.
\end{cases}
```

The shift vector $s \in \mathbb{R}^{n_{\text{dim}}}$ is also included in
$\theta$ without any transformation.  The complete parameter vector for a basis
of $n$ Gaussians therefore has dimension

```math
|\theta| = n \times \left(\frac{n_{\text{dim}}(n_{\text{dim}}+1)}{2} + n_{\text{dim}}\right).
```

### Gradient computation (Hellmann-Feynman)

Computing the gradient $\nabla_\theta \lambda_{\min}$ by automatic
differentiation through the eigensolver is expensive and numerically fragile.
Instead, the **Hellmann-Feynman theorem** is used: if $c$ is the eigenvector
corresponding to $\lambda_{\min}$, then

```math
\frac{\partial \lambda_{\min}}{\partial \theta_k}
= c^T \!\left(\frac{\partial H}{\partial \theta_k} - \lambda_{\min}\,\frac{\partial S}{\partial \theta_k}\right) c.
```

The eigenproblem is solved in `Float64` to obtain $\lambda_{\min}$ and $c$.
ForwardDiff then differentiates only through the matrix-build steps
($H(\theta)$ and $S(\theta)$), which avoids differentiating through any
eigenvalue decomposition.  The chunk size of ForwardDiff is tuned so that each
Gaussian contributes approximately five dual-number columns per pass.

### Optimisation

The combined value-and-gradient function is passed to the L-BFGS
implementation provided by [OptimKit.jl](https://github.com/Jutho/OptimKit.jl).
Regions where the overlap matrix $S$ is near-singular (degenerate Gaussians)
are handled by returning $(+\infty, \mathbf{0})$ from the objective, which
acts as an infinite-cost barrier that the line search naturally avoids.

### Trace loss (warm-start refinement)

An alternative surrogate loss is $\operatorname{Tr}(S^{-1}H)$, the sum of
**all** generalised eigenvalues.  This is differentiable everywhere without
requiring an eigensolver, but is only useful when the initial basis is already
close to the physical ground state (e.g. after a stochastic run), because for
random initialisations the upper eigenvalues are large and the optimizer may
find a trivial degenerate minimum.
