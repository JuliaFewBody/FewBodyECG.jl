# ECG literature and automatic differentiation

Reviewed 2026-11-04. Balanced, targeted literature review for FewBodyECG.jl. This is a selected evidence base, not a systematic census. Full-text examination focused on the closest methodological sources; some application entries were screened at abstract/publisher-record level. The report distinguishes those evidence levels. Publication-count queries are preserved in publication-search.json.

Six PDF text caches were rendered with the available pdftotext fallback; they are searchable text, not reliable equation transcriptions. Source-level equation checks used the original papers / author HTML. Other entries record metadata rather than cached full text. Metadata acquisition used Crossref and primary arXiv citation tags after the Semantic Scholar batch endpoint returned HTTP 429.

## Field landscape

- Established analytic integrals and basis-parameter gradients: [@mitroy2013], [@kinghorn1997], [@fedorov2017].
- Shift/source generating functions, angular momentum and tensor operators: [@suzuki2008], [@silvestrebrac2008], [@joyce2016], [@hiyama2018], [@zaklama2020], [@fedorov2024]. The source-derivative identities predate general-purpose AD implementations.
- Differentiable Gaussian optimization: [@sun2022], [@detmold2023], [@recchia2024]. Sun explicitly uses shifted Gaussians and AD. Recchia establishes GVM/DynamicGVM/hybrid algorithms; its manuscript does not establish the exact autograd path. Public ECG_Julia software also uses ForwardDiff and generalized Hellmann-Feynman contraction (repository link in the report).
- Precision floating ECGs: pre-Born-Oppenheimer projection [@muolo2018], regularized relativistic operators [@racsai2024], spin-dependent molecular operators [@jeszenszki2025], excited-state polarizabilities [@mrozek2026].
- Ultracold and nuclear applications: [@laird2024], [@laird2026], [@weissattia2024], [@avraham2026]. Publications address pairing, density matrices, finite-volume EFT matching and scattering.
- Extensions: deformed/cavity ECGs [@beutel2021], complex wavepacket propagation [@schrader2025], six-fermion dynamics [@gao2026], periodic electronic systems [@varga2026].
- Adjacent rather than ECG-equivalent AD precedents: explicitly correlated MP2-F12 derivatives [@mitchell2024] and Gaussian generating-function AD in quantum optics [@fitzke2023].

## Key open problems

- Demonstrate a reusable AD implementation of source-generated matrix elements with useful nested parameter derivatives, against the established generating-function and analytic-gradient baselines [@suzuki2008; @zaklama2020; @fedorov2024; @sun2022]. No first-use claim is established by this search.
- Include costly noncentral/higher-derivative operators in differentiable optimization; the relativistic and s-d mixing terms were excluded from the optimization objective in [@detmold2023], Section II.C.3.
- Compute relaxed response and parameter sensitivities with controlled overlap conditioning; first-order Hellmann-Feynman formulas do not alone supply full second-order response [@varga2026; @mitchell2024; @mrozek2026].
- Support realistic radial kernels with controlled differentiation and integration error, beyond the simplest Gaussian/Coulomb models [@racsai2024; @fedorov2017].

## Key bottlenecks

- Near-linear dependence of a nonorthogonal basis, state degeneracy, regularization and changes in retained overlap subspaces [@recchia2024; @sun2022].
- Derivative-order cost, compilation, memory and cancellation in generic nested AD. Measure these against analytic formulas and recurrences; no speed advantage is assumed [@fedorov2024; @fitzke2023].
- Symmetry projection and angular coupling remain necessary: a Cartesian polynomial degree or a shifted Gaussian does not by itself define total angular momentum [@suzuki2008; @muolo2018].
- Smooth Gaussians retain cusp and long-range representation limitations, irrespective of derivative backend [@mitroy2013; @racsai2024].
