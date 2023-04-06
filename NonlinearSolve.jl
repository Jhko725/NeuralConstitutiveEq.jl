##
using Pkg
Pkg.add("NonlinearSolve")
##
using NonliearSolve, StaticArrays
f(u, p) = u .* u .- p
u0 = @Svector[1.0, 1.0]
p = 2.0
probN = NonlinearProblem(f, u0, p)
solver = solve(probN, NewtonRaphson(), reltol = 1e-9)
##