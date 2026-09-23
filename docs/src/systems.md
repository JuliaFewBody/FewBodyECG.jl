# Building systems

The main system builder is `Operators`.  Give it particle masses and, when
convenient, charges:

```julia
ops = Operators([1.0e15, 1.0], [+1.0, -1.0])
ops += "Kinetic"
ops += "Coulomb"
```

`ops += term` builds a new `Operators` containing the extra term and rebinds
`ops` to it; `push!(ops, term)` adds the term to the existing object in place.
Any earlier reference to `ops` keeps its old term list after `+=`, but sees the
new term after `push!`.

Masses and charges are in atomic units.  A very large mass such as `1.0e15`
is the usual fixed-nucleus approximation.
