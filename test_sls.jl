##
using Roots
using IntervalRootFinding

E₁ = 0.01
E₂ = 10
η = 0.0025
t_s = 2e-3
t = 0.5
f(x) = E₁*(t-x)/(3*η) - exp(-(E₂*(t-x))/(3*η)) 
sol = roots(f, (-Inf..Inf))
print(sol)

let 
    f(x) = 10*(0.5-x)/(3*0.0025) - exp(-(10*(0.5-x)/(3*0.0025)))
    sol = roots(f, (-4..Inf))
    print(sol)
end