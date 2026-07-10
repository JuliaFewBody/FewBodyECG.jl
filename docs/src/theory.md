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
g(\vec{r}_1, \ldots, \vec{r}_N) = \exp \left( - \sum_{i,j=1}^{N} A_{ij}\vec{r}_i \cdot \vec{r}_j + \sum_{i=1}^{N} \vec{s}_i \cdot \vec{r}_i \right),
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

In the package, `A` has size `N x N`, while the shift supervector `s` has size `N x 3`: each row is the three-dimensional shift of one coordinate. Passing an `N`-element shift vector remains supported as a compatibility form for a shift along the fixed `z` axis.

## Spin-dependent Gaussian interactions

`SpinGaussian(orbital, spin)` combines an orbital Gaussian with a direct product of spin-1/2 projection states. The tensor and spin-orbit interactions use this representation and retain complex matrix elements. The available closed forms are Gaussian central, oscillator, many-body Gaussian, tensor, and spin-orbit terms. The tensor form can remove its central spin-spin part with `traceless = true`.

The short-range Yukawa and derivative-generated form factors are intentionally not included yet. A linear combination of Gaussian form factors is the fully analytic alternative used here.
