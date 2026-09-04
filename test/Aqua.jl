using Aqua
using ExplicitImports

Aqua.test_all(FewBodyECG)

# These are documented extension APIs, but their packages do not yet mark them
# public for Julia's `Base.ispublic` metadata.
ExplicitImports.test_explicit_imports(
    FewBodyECG;
    all_qualified_accesses_are_public = (;
        ignore = (:Chunk, :GradientConfig, :gradient, :KineticTerm, :PotentialTerm, :require_one_based_indexing),
    ),
)
