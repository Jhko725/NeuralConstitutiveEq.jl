##
using Pkg
Pkg.add("StaticArrays")

##
using NonlinearSolve, StaticArrays
f(u, p) = u .* u .- p
u0 = @SVector[1.0, 1.0]
p = 2.0
probN = NonlinearProblem(f, u0, p)
solver = solve(probN, NewtonRaphson(), reltol = 1e-9)
print(solver)
##
using NonlinearSolve
f(u, p) = u * u - 2.0
uspan = (1.0, 2.0)
probB = IntervalNonlinearProblem(f, uspan)
sol = solve(probB, Falsi())
##
f(u, p) = u .* u .- 2.0
u0 = 1.5
probB = NonlinearProblem(f, u0)
cache = init(probB, NewtonRaphson())
solcer = solve!(cache)

##
using Integrals
f(u, p) = sum(sin.(u))
prob = IntegeralProblem(f, ones(3), 3ones(3))
sol = solce(prob, HCubatureJL();, retol= 1e-3, abstol = 1e-3)