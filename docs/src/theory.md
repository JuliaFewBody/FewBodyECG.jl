# Theory

## Basis expansion

`FewBodyECG.jl` solves the Schrödinger equation reformulated as an eigenvalue equation

```math
\hat{H}|\psi\rangle = \varepsilon |\psi\rangle,
```
where ``\hat{H}`` is the Hamiltonian of the few-body system with a discrete spectrum of eigenvalues ``\varepsilon`` and corresponding wave function ``|\psi\rangle``. We expand the wave function in terms of basis function and corresponding coefficients

```math
|\psi\rangle = \sum_{i=1}^N c_i |\phi_i \rangle.
```
We insert this expansion into the Schrödinger equation and multiply by ``\langle \phi_k|_{1\leq k \leq N}`` from the left

```math
\sum_{i=1}^N \langle \phi_k | \hat{H} | \phi_i \rangle = \varepsilon \sum_{i=1}^N \langle \phi_k | \phi_i \rangle c_i.
```
Writting this expression in matrix notation, the generalized eigenvalue problem now looks like
```math
   \mathcal{H}c = \varepsilon \mathcal{N}c,
```
where 
```math
   \mathcal{H} = \langle \phi_k | \hat{H} | \phi_i \rangle, \quad \mathcal{N} = \langle \phi_k | \phi_i \rangle, \quad c = (c_1, ..., c_N)^\top.
```
The overlap matrix, ``\mathcal{N}``, would be equal to a delta function if we picked an orthonormal basis, so we must pick a non-orthonormal basis for the variation method to work. The general idea is to use the Rayleigh-Ritz method to approximate the energy states and the wave functions

```math
E_0 \leq \frac{\langle \phi | \hat{H} | \phi \rangle }{\langle \phi | \phi \rangle }.
```
From this theorem we see that in order to approximate the ground state, ``E_0``, we have to minimize the functional wrt. the variational parameter, ``c``. The functional then becomes
```math
   E[\phi] = \frac{c^\dagger \mathcal{H} c}{c^\dagger \mathcal{N}c},
```
and by minimization we get the following condition for the generalized eigenvalue equation

```math
   \frac{2}{c^\dagger \mathcal{N}c}(\mathcal{H}c-E[\phi]\mathcal{N}c)^\dagger = 0
```

## Basis functions

We use a non-orthonormal basis expansion called explicitly correlated Gaussians given by

```math
   \phi_g(\mathbf{r}_1, \dots, \mathbf{r}_N) = \exp\left( - \sum_{i,j=1}^N  A_{ij}\mathbf{r}_i \cdot \mathbf{r}_j + \sum_{i=1}^N \mathbf{s}_i \cdot \mathbf{r}_i \right),
```
for a system of ``N`` total particles each with coordinates ``\mathbf{r}_i`` for ``i=1,...,N``. The bold faces denote vectors. Writing this in matrix notation we get

```math
   g(\mathbf{r}) = \exp(-\mathbf{r}^\top A \mathbf{r} + \mathbf{s}^\top \mathbf{r}),
```
where ``A`` is a positive definite and symmetric matrix and ``\mathbf{s}`` are the shift vectors of the Gaussians. 

Returning to the basis expansion, we can now represent the few-body wave function as a linear combination of explicitly correlated Gaussians
```math
   \Psi(\mathbf{r}_1,...,\mathbf{r}_N) = \sum_{i=1}^{N_{\text{G}}} c_i g_i(\mathbf{r}_1,...,\mathbf{r}_N), 
```
where ``N_{\text{G}}`` is the number of Gaussians you want in your expansion. 

## Matrix elements

We are now ready to show-case one of the advantages of this numerical method. All of the matrix elements are analytical and easy to calculate. We simply insert the operators and calculate the matrix element.

### Overlap of two Gaussians
First, we consider the overlap of two Gaussians, ``g'`` and ``g```

```math
   \langle g' | g \rangle = \int \text{d}^3 \mathbf{r}_1 ... \text{d}^3 \, \mathbf{r}_N \exp(-\mathbf{r}^\top (A'+A)\mathbf{r}+(\mathbf{s}'+\mathbf{s})^\top \mathbf{r}).
```
We define ``B=A'+A`` and ``\mathbf{v}=s'+s`` and make an orthogonal transformation from which we get

```math
   \langle g' | g \rangle = \text{e}^{\frac{1}{4}\mathbf{v}^\top B^{-1}\mathbf{v}} \left( \frac{\pi^N}{\text{det}(B)} \right)
```

### Kinetic operator

Following the same approach we calculate the matrix elements for the kinetic operator given by

```math
   \hat{T} = - \sum_{ij=1}^N \frac{\partial }{\partial \mathbf{r}} \Lambda \frac{\partial }{\partial \mathbf{r}^\top},
```
where ``\Lambda = \frac{\hbar^2}{2m_i}\delta_{ij}. Calculating the matrix elements yields

```math
   \langle g' | \hat{T} | g \rangle = (6 \text{Tr}(A'\Lambda A B^{-1})+2(A' \mathbf{u}-\mathbf{s'})^\top \Lambda(2A \mathbf{u}-\mathbf{s})) \text{e}^{\frac{1}{2}\mathbf{v^\top \mathbf{u}}} \left( \frac{\pi^N}{\text{det}(B)} \right)^{3/2},
```
where ``B=A'+A`` and ``\mathbf{u} = \frac{1}{2}B^{-1}\mathbf{v}`` and ``\text{Tr}`` denotes the trace.

## References

1. D. V. Fedorov, A. F. Teilmann, M. C. Østerlund, and T. L. Norrbohm,
   "Explicitly Correlated Gaussians with Tensor Pre-factors: Analytic Matrix
   Elements," *Few-Body Systems* **65**, 75 (2024).
   [doi:10.1007/s00601-024-01945-x](https://doi.org/10.1007/s00601-024-01945-x).
2. Y. Suzuki and K. Varga, *Stochastic Variational Approach to
   Quantum-Mechanical Few-Body Problems*, Springer, 1998.
3. Antique.jl, analytical solutions for solvable quantum-mechanical models:
   [github.com/ohno/Antique.jl](https://github.com/ohno/Antique.jl).
