# Building systems

The main system builder is `Operators`.  Give it particle masses and, when
convenient, charges:

```julia
ops = Operators([1.0e15, 1.0], [+1.0, -1.0])
ops += "Kinetic"
ops += "Coulomb"
```

Masses and charges are in atomic units.  A very large mass such as `1.0e15`
is the usual fixed-nucleus approximation.
