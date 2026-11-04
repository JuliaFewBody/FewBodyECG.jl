#set heading(numbering: none)
#text(font: "Avenir Next", size: 17pt, weight: "bold")[Paper direction for FewBodyECG.jl]

This project note is an assessment based on the literature review and a read-only inspection of the current working tree. It proposes experiments; it reports no new numerical results and makes no priority claim.

= Recommended contribution

*Working title:* Automatic generation and differentiation of explicitly correlated Gaussian matrix elements.

*Research question:* Can one implementation of a shifted scalar operator kernel reliably supply mixed-rank matrix elements and further derivatives, at a practical cost, while reducing the independent formulas that must be implemented and checked?

A defensible claim would be: “We implement ECG generating-function matrix elements using automatic differentiation, with composable operator kernels and nested parameter derivatives, and quantify their accuracy, stability and computational cost.” The contribution becomes stronger if it enables a useful calculation that would otherwise require several independent derivative implementations. Calling the generating-function identity itself new, or presenting AD energy optimization as the first such use in ECGs, would conflict with the reviewed work.

= What the repository already supplies

The inspected #raw("src/polarized_elements.jl") implements rank-1/rank-2 and cross-rank matrix elements through mixed shift derivatives of rank-0 elements. Nested ForwardDiff tags also account for outer user derivatives. This is the most distinctive starting point for the proposed implementation study.

#raw("src/gradient.jl") uses log-Cholesky width parameters and a first-order generalized Hellmann-Feynman contraction. These are useful components, with close precedents. Its present GVM entry points explicitly require rank-0 basis functions: the current code should not be described as already optimizing a full polarized/tensor basis. The numerical radial kernels and spin-dependent operators provide potential extensions to validate, rather than evidence that every nested derivative combination currently works.

No package tests, benchmark runs or performance measurements were performed for this literature review. Existing staged source changes were left untouched. Claims about code behavior above refer to inspection, not new validation.

= Minimum convincing evidence

1. *Matrix-element accuracy.* Compare generated rank-0/1/2 and mixed-rank values against independent analytic formulas. Add degree-3/4 cases or angularly coupled combinations to establish reach. Include zero and nonzero shifts, several Jacobi dimensions and near-redundant bases. Report absolute and relative errors; relative error alone is misleading near a selection-rule zero.

2. *Derivative accuracy.* Check outer derivatives with respect to widths, source/center parameters and interaction parameters. Use analytic results or higher-precision checks where available; finite differences are a secondary consistency check. Test smooth limits of Coulomb kernels at zero displacement and the convergence of numerical integration separately.

3. *Cost.* Measure compilation separately from warmed execution, then record memory, allocations and runtime against analytic or recurrence implementations as derivative order, particle number, basis size and operator complexity increase. Generic nested forward-mode AD can accumulate substantial derivative-order overhead; do not assume a speedup. Implementation size can support the case if counted reproducibly, but does not substitute for a physical calculation.

4. *Optimization comparison.* At matched initial bases and accuracy targets, compare SVM/refinement, analytic-gradient optimization where available, AD optimization and hybrids. Use multiple seeds and report time to a target accuracy and failure rates. Compare implementations that handle near-dependent bases consistently. Include full matrix construction and eigensolver costs.

5. *One demanding observable.* Prefer a three-body problem with angular/tensor coupling and a physical sensitivity or response quantity. A simpler Coulombic system can provide an independent reference before this step. Hydrogen p/d energies are useful reproduction checks, but repeat the 2024 tensor paper's validation and are insufficient as the main result.

= Two stronger extensions

*Operator coverage:* demonstrate practical optimization including a tensor/spin-dependent or higher-derivative interaction, then compare a basis optimized with and without it. Detmold et al.'s excluded terms motivate the problem, but reproducing their finite-volume EFT setup is a separate, larger undertaking. A controlled few-body Gaussian model is a reasonable first test if its physical scope is stated.

*Relaxed response:* compute an interaction-parameter sensitivity or a static response, with a clear definition of fixed-basis versus reoptimized-basis response. First derivatives of a stationary energy can use an envelope/Hellmann-Feynman argument. Second derivatives need eigenvector and nonlinear-basis relaxation, obtained from response equations or justified implicit differentiation.

= Numerical issues that determine credibility

Source differentiation is with respect to an auxiliary Gaussian source, not a numerical approximation to a spatial differential operator. Operator evaluation and AD accuracy are limited by floating-point arithmetic and any kernel approximations. A Cartesian prefactor's degree is also not automatically its total angular momentum: harmonic projection/coupling and permutation symmetry remain part of the method.

Replacing an overlap matrix with a regularized one changes the eigenproblem. Document the choice and evaluate the final physical Rayleigh quotient with the original matrices; do not assume all regularized eigenvalues retain the desired variational meaning. If an eigenvalue cutoff changes the retained subspace, the effective optimization map can become nonsmooth. Near degeneracies require a state/subspace treatment instead of blindly differentiating a single ordered eigenvalue.

For radial quadrature, differentiate a numerically controlled integral representation. Verify convergence as both integration tolerance and derivative order vary, and distinguish differentiation of the mathematical integral from differentiation of an adaptive algorithm's branch choices. AD alone does not correct the Gaussian basis's cusp or long-distance limitations.

= First experiment and decision rule

Start with one Coulomb or radial-potential kernel, rank-1/rank-2 cross elements and an outer width derivative. Verify it against independent formulas at zero and nonzero shifts, then measure its warmed runtime and compilation cost. This is a compact test of the distinctive composition already suggested by the code.

Proceed to a methods paper if that composition remains accurate and practical and supports a broader operator family, useful response or substantially easier verified implementation. If it only reproduces the known low-rank formulas with large overhead, present it as a software contribution and choose a narrower application result. The literature supports a research opportunity, but does not yet establish that the proposed implementation clears that threshold.
