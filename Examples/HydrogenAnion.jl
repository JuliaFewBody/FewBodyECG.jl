using Revise 
using .FewBodyPhysics

masses = [1e15, 1.0, 1.0]
w_list = [ [1, -1, 0], [1, 0, -1], [0, 1, -1] ]

K = [0 0 0; 0 1/2 0; 0 0 1/2]
J, U = jacobi_transform(masses)
K_transformedformed = J * K * J'
w_transformedformedformed = [U' * w_list[i] for i in 1:length(w_list)]
Theortical_value = -0.527751016523

p, Energy, bases = run_simulation(50, :quasirandom, w_transformedformedformed, K_transformedformed)

