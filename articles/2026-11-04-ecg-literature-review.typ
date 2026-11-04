#let title = [Explicitly correlated and shifted Gaussians:\ Literature and opportunities for automatic differentiation]
#let review_date = [4 November 2026]
#let short_title = [ECG literature and automatic differentiation]
#let serif = "Libertinus Serif"
#let sans = ("Avenir Next", serif)
#let ink = rgb("24282b")
#let muted = rgb("53616b")
#let accent = rgb("245b65")
#let tint = rgb("f2f6f6")
#let rule = rgb("cbd4d7")
#set document(title: title)
#set page(paper: "a4", margin: (x: 18mm, y: 18mm), footer: context {
  set text(font: sans, size: 8pt, fill: muted)
  grid(columns: (1fr, auto), gutter: 12pt, short_title, counter(page).display())
})
#set text(font: serif, size: 10pt, fill: ink)
#set par(justify: true, leading: 0.55em, spacing: 0.65em)
#set heading(numbering: "1.1")
#show heading: set text(font: sans, weight: "bold")
#show heading.where(level: 1): set text(size: 13pt)
#show heading.where(level: 2): set text(size: 11pt)
#show heading.where(level: 1): set block(above: 1.3em, below: 0.55em)
#show heading.where(level: 2): set block(above: 1em, below: 0.45em)
#show link: set text(fill: accent)
#set figure(gap: 6pt)
#show figure.caption: set text(size: 9pt, fill: muted)
#let report_table(columns, headers, rows) = {
  assert(headers.len() == columns.len())
  for row in rows { assert(row.len() == columns.len()) }
  set text(size: 9pt)
  set par(justify: false, leading: 0.45em)
  table(columns: columns, align: left + top, inset: (x: 5pt, y: 5pt),
    stroke: (x, y) => (bottom: if y == 0 { 0.8pt + accent } else { 0.4pt + rule }),
    fill: (x, y) => if y == 0 { tint },
    table.header(repeat: true, ..headers.map(h => text(weight: "bold", h))),
    ..rows.flatten())
}
#block(breakable: false, below: 10pt)[
  #text(font: sans, size: 18pt, weight: "bold", title)
  #v(5pt)
  #text(font: sans, size: 9pt, fill: muted, review_date)
]

= Overview

Explicitly correlated Gaussians (ECGs) represent a few-body wavefunction as a linear combination of Gaussians containing interparticle correlations. Shifted or floating variants add movable centers; polynomial and global-vector prefactors supply angular structure. Their appeal is the combination of flexible correlation and tractable integrals. This review maps the literature for a developer of FewBodyECG.jl considering a methods paper, with particular attention to derivative computation and the distinction between a mathematical construction and its software implementation. @mitroy2013

The principal finding is that automatic differentiation (AD) has already been applied to shifted correlated-Gaussian optimization, while source derivatives have long generated angular matrix elements. The narrower opportunity is to make those derivative constructions reusable, composable and numerically reliable, then demonstrate a capability or measured advantage beyond existing implementations. Analytic integrals and AD are compatible: an analytic scalar integral can be the function that AD differentiates. @sun2022 @suzuki2008 @fedorov2024

== Search scope and publication volume

Discovery combined landscape searches, historical gradient work, cross-vocabulary searches for floating/global-vector Gaussians, and searches for differentiable programming. Sources were verified against author preprints and publisher records. Full-text scrutiny concentrated on Sun 2022, Detmold 2023, Recchia 2024, Fedorov 2017/2024, global-vector constructions and selected extensions. Some application papers were assessed from abstracts and publication records; their entries do not establish implementation details absent from those sources. This is a targeted review, not a systematic review or a proof that undiscovered prior art does not exist.

OpenAlex was queried on 4 November 2026 with the quoted phrase #raw("\"explicitly correlated gaussian\"") and a publication-date cutoff of 2026-11-04. The returned query description applies stemming, including singular/plural variants. The three scopes below are nested searches, not disjoint categories. @mitroy2013 provides a separate historical entry point to the field.

#report_table((1.65fr, auto, 2.25fr), ([Search scope], [Records], [Interpretation]), (
  ([Title only], [131], [The method is explicitly named in the title.]),
  ([Title or abstract], [377], [A more focused measure of the indexed literature.]),
  ([Broad search], [699], [Also searches available full text and keywords; may capture incidental mentions or references.]),
))

These are indexed records, without manual relevance screening or journal/preprint deduplication. Older papers using different terminology can be missed. A separate broad query for shifted/floating phrases returned 21 records; it is especially incomplete because terminology varies. The same ECG phrase combined with AD terms returned only six and missed known nuclear AD work, illustrating why it cannot measure AD adoption. Exact requests and yearly responses are saved in #raw(".knowledge/publication-search.json"). The API documentation explains the search scopes: #link("https://help.openalex.org/api/searching/")[OpenAlex search documentation]. The defensible magnitude is *hundreds of indexed ECG-related records*, with a substantially smaller selected set needed to evaluate this particular novelty claim.

= Key questions

== What are researchers trying to calculate?

Current work spans precision molecular operators and spectroscopy, few-fermion pairing, finite-volume nuclear EFT, response properties and time-dependent dynamics. The recurring aim is to obtain reliable observables in strongly correlated systems small enough for a flexible explicitly correlated representation. The examples below establish activity, not market shares of the literature.

#report_table((1.1fr, 2.25fr, 1.65fr), ([Theme], [Current research question], [Representative evidence]), (
  ([Precision atoms and molecules], [Relativistic and spin corrections, finite-mass effects and symmetry-resolved response.], [Rácsai et al. 2024; Jeszenszki et al. 2025; Mrozek and Strasburger 2026. @racsai2024 @jeszenszki2025 @mrozek2026]),
  ([Ultracold systems], [How pairing and collective behavior emerge with a few trapped particles.], [Laird et al. 2024/2026; Gao et al. 2026. @laird2024 @laird2026 @gao2026]),
  ([Nuclear EFT], [How finite-volume spectra constrain interactions and predict free-space observables.], [Weiss-Attia et al. 2024; Avraham and Bazak 2026. @weissattia2024 @avraham2026]),
  ([New representations], [How to handle anisotropy, complex wavepackets and periodic systems.], [Beutel et al. 2021; Schrader et al. 2025; Varga 2026. @beutel2021 @schrader2025 @varga2026]),
))

== Has the field already adopted automatic differentiation?

Yes, in identifiable methods and applications. That establishes prior art without establishing its prevalence across the entire field. It is necessary to distinguish integration, differentiation of an integral with respect to basis parameters, generation of prefactors by source differentiation, and differentiation of an optimized observable. A paper can use analytic integrals and AD simultaneously. The first ECG energy gradient is not a recent development: implementations using analytic gradients date at least to the 1990s. @kinghorn1997 @fedorov2017

== What remains difficult after derivatives become available?

An optimizer still encounters redundant basis vectors and poorly conditioned overlap matrices. Angular and permutation symmetry still require explicit construction. Higher responses require the response of the eigenvector and, when applicable, of the optimized nonlinear basis. The relevant question is whether a derivative implementation improves reliable calculations at a fixed accuracy and cost. Gradient stationarity alone is not convergence to an exact physical answer. @recchia2024 @muolo2018 @mitchell2024

= Technical approaches

== Analytic integrals with manually or symbolically derived gradients

Analytic integral evaluation is a principal strength of ECGs. Kinghorn and Poshusta implemented exponent-parameter gradients and compared several optimization methods on atomic and positronic systems in 1997. Fedorov's shifted-Gaussian work supplies analytic matrix elements and nonlinear-parameter derivatives; the journal citation is 2017, although online publication was in December 2016. These are suitable derivative baselines, rather than evidence that Gaussian optimization is exclusively stochastic. @kinghorn1997 @fedorov2017

Precision work continues to extend the operator algebra. Rácsai and collaborators make regularized relativistic corrections practical with floating ECGs by replacing one Coulomb factor with a Gaussian expansion. Jeszenszki and collaborators evaluate spin-dependent Breit-Pauli operators in molecular ECG computations. These papers show that integral and operator coverage remains an active computational problem. Their physical approximations and regularization choices remain relevant when a derivative backend changes. @racsai2024 @jeszenszki2025

The active groups represented here include the Budapest Mátyus group for precision floating-ECG operators, the Adamowicz/Bubin research lineage for high-accuracy all-particle calculations, and Fedorov's Aarhus work on shifted kernels. This grouping describes the cited work, rather than an exhaustive affiliation census. @mitroy2013 @racsai2024 @fedorov2024

== Source-generating kernels and angular prefactors

The global-vector literature constructs angular functions from a Gaussian containing an auxiliary source, using source derivatives and angular projection. Suzuki and collaborators cover double-global-vector angular structure and noncentral operators. Silvestre-Brac and Mathieu derive tensor and spin-orbit elements in a related generating-function framework. Joyce and Varga give an arbitrary-angular-momentum matrix-element algorithm; its publisher abstract was checked, but its full implementation mechanism was not examined in this review. @suzuki2008 @silvestrebrac2008 @joyce2016

Zaklama and collaborators explicitly generate coordinate monomials and their matrix elements by differentiating one-dimensional ECG source kernels. Fedorov and collaborators generate rank-1 and rank-2 prefactors by Taylor expansion in shifts, with hydrogen p/d tests. The Gaussian expansion method's infinitesimally shifted lobes provide a related construction using limits of shifted sums. These precedents rule out treating source differentiation, general angular structure or avoiding repeated angular integration as new mathematical ideas. @zaklama2020 @fedorov2024 @hiyama2018

For real Gaussians, collect the Jacobi Cartesian coordinates into a vector $x$ and write
$ g(A,s;x) = exp(-x^T A x + s^T x). $
An auxiliary displacement along a vector $a$ gives
$ frac(partial g(A,s+epsilon a;x), partial epsilon) = (a^T x) g(A,s+epsilon a;x). $
Applying the corresponding bra and ket source derivatives to the scalar operator kernel generates the required polynomial matrix element, when differentiation and integration may be interchanged. Complex basis functions additionally require consistent conjugation. This is the established construction to which AD can supply an evaluation backend. @suzuki2008 @fedorov2024

No verified source in this targeted search was found to use general-purpose AD specifically as an ECG prefactor generator with nested outer parameter differentiation. That is a bounded search result, not a first-use claim. A close cross-field precedent uses AD of Gaussian generating functions to obtain photon statistics; novelty should therefore be stated at the level of the ECG implementation and demonstrated capabilities. @fitzke2023

== Automatic differentiation for variational optimization

Sun, Detmold, Luo and Shanahan explicitly optimize shifted Gaussian widths, displacements and linear coefficients by AD of a variational energy, then combine optimized blocks through a separate generalized eigenproblem. Their nuclear EFT benchmarks extend to six particles. This is direct prior art for AD-based shifted-ECG optimization, although their reported gains do not establish universal superiority over stochastic optimization. @sun2022

Detmold, Romero-López and Shanahan extend the approach to NLO finite-volume two-nucleon spectra. Crucially, Section II.C.3 excludes the relativistic kinetic correction and s-d mixing term from the optimization objective because their gradients are demanding, while retaining them in the final eigenproblem. This supplies a concrete motivation for improving derivative coverage; reproducing that gap in a new implementation would require matching its operators, symmetry and physical regime. @detmold2023

Recchia and collaborators already compare fixed, growing and hybrid gradient variational methods with SVM for bosonic helium clusters. Their implementation uses PyTorch; the manuscript does not specify the exact autograd path sufficiently to establish differentiation through its eigensolver. Their algorithms and conditioning discussion are essential baselines for any GVM, DynamicGVM or SVM-to-GVM paper. @recchia2024

A public Julia implementation, #link("https://github.com/AlainChance/ECG_Julia")[ECG_Julia], also advertises ForwardDiff matrix-element derivatives followed by a generalized Hellmann-Feynman contraction. Its repository manuscript is dated June 2026; a peer-reviewed journal publication was not verified. This is relevant software prior art, rather than a reference-energy benchmark. A broad claim of the first Julia ECG solver with AD gradients is therefore unsuitable.

== Floating ECGs, numerical kernels and response

Floating Gaussians efficiently describe off-center molecular density, but a shifted function is not generally an angular-momentum eigenfunction. Muolo, Mátyus and Reiher apply rotation and parity projection in pre-Born-Oppenheimer calculations. An implementation that differentiates source coordinates must still supply the symmetry machinery required by its physical application. @muolo2018

Response is an active application rather than an empty area: Mrozek and Strasburger's October 2026 preprint evaluates static and dynamic excited-beryllium polarizabilities using Hylleraas-functional response wavefunctions and approximately 7,000-8,000 ECGs. In adjacent quantum chemistry, Mitchell, Turney and Schaefer apply AD to MP2-F12 derivatives and discuss difficulties with higher derivatives and degeneracies. MP2-F12 is explicitly correlated, but is not the few-body ECG ansatz. @mrozek2026 @mitchell2024

For a normalized, isolated generalized eigenpair, a first derivative can be assembled as
$ frac(partial E,partial theta) = c^dagger (frac(partial H,partial theta) - E frac(partial S,partial theta)) c, quad c^dagger S c = 1. $
This formula is established, including in Varga's 2026 Appendix I. Full second derivatives generally require eigenvector response; nonlinear-basis response is also needed for derivatives of a reoptimized variational model. Re-differentiating a contraction while freezing all coefficients does not provide that response. The distinction follows from differentiating the stationarity conditions. @varga2026

== Deformation, time dependence and periodicity

Beutel and collaborators extend ECGs to nonspherical environments, including cavity-QED Hamiltonians. Schrader, Pedersen and Kvaal propagate complex ECG wavepackets with Rothe's method, variable projection and analytic gradients, demonstrated on multidimensional model dynamics. These are complementary representation and propagation developments; shifting an otherwise isotropic Gaussian does not by itself reproduce their full capabilities. @beutel2021 @schrader2025

Gao and collaborators' June 2026 preprint uses time-dependent ECGs to describe collective expansion and molecule formation in up to six fermions. Varga's periodic ECG work is published as JCP 165, 044104 (2026), with analytic lattice-sum elements and a hydrogen-chain demonstration. The latter already includes Cholesky widths and generalized Hellmann-Feynman optimization formulas. These results preclude broad novelty claims based only on dynamics, periodicity or positive-definite parameterization. @gao2026 @varga2026

Laird and collaborators use ECG/SVM calculations to study pairing and the structure of mesoscopic two-dimensional Fermi gases. Their 2025 preprint on energetic and structural properties is now a 2026 SciPost Physics Core paper. Nuclear applications likewise remain active in NLO finite-volume extrapolation and finite-cutoff scattering calculations. These give demanding application targets, but a generic library need not reproduce every domain to support a focused paper. @laird2024 @laird2026 @weissattia2024 @avraham2026

= Open problems

The ranking below is an assessment of methodological opportunities for a small research team. Urgency describes relevance to a credible methods contribution, not a measured community consensus. Each proposed capability still needs numerical evidence.

#report_table((auto, 1.2fr, 1.75fr, 1fr, auto), ([No.], [Problem], [Why it matters], [Who can act], [Urgency]), (
  ([1], [Composable derivative generation], [Connect established source kernels to reusable AD and outer derivatives; demonstrate a practical advantage. @zaklama2020 @fedorov2024], [ECG and AD developers], [High]),
  ([2], [Operator-complete optimization], [Include useful noncentral or higher-derivative terms that prior DP work treated only after optimization. @detmold2023], [Few-body methods teams], [High]),
  ([3], [Stable relaxed response], [Control overlap conditioning, state tracking and basis response for trustworthy sensitivities. @recchia2024 @mitchell2024], [Numerical analysis and physics teams], [High]),
  ([4], [Differentiable numerical radial kernels], [Keep integration and derivative errors controlled when closed forms are unavailable or inconvenient. @fedorov2017 @racsai2024], [Integral-library developers], [High]),
  ([5], [Derivative-order scaling], [Measure compilation, memory and execution costs against specialized formulas before generality becomes impractical. @fedorov2024 @fitzke2023], [AD and ECG developers], [High]),
  ([6], [Application-level validation], [Show improved observables or lower computational cost beyond low-order hydrogen formula checks. @mrozek2026 @laird2026 @gao2026], [Domain collaborators], [Medium]),
))

#bibliography("2026-11-04-ecg-literature-review.bib", style: "ieee")

#pagebreak()
#include "2026-11-04-ecg-paper-direction.typ"
