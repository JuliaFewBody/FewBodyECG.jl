using FewBodyECG

helium = Operators([1.0e15, 1.0, 1.0], [+2.0, -1.0, -1.0])
helium += "Kinetic"
helium += "Coulomb"

he_ref = -2.9037
he = solve(helium, SVM(basis = 50, candidates = 25, scale = 1.0))

helium_num = Operators([1.0e15, 1.0, 1.0])
helium_num += "Kinetic"
f_nucleus(r) = -2.0 / r
f_electron(r) = 1.0 / r

helium_num += (f_nucleus, numerical, 1, 2)
helium_num += (f_nucleus, numerical, 1, 3)
helium_num += (f_electron, numerical, 2, 3)

he_num = solve(helium_num, SVM(basis = 50, candidates = 25, scale = 1.0))
