##
using Roots
using IntervalRootFinding
import Statistics: mean
using StaticArrays


# using Quadrature

E₁ = 100
E₂ = 1
η = 10
t = 0.3
tₘ = 0.2
t_max = tₘ
# f(t₁) = E₁*(t-t₁)/(3*η) - exp(-(E₂*(t-t₁))/(3*η)) + 1 
# sol = roots(f, (-Inf..Inf))
# print(sol)
##
# let 
#     f(x) = 10*(0.5-x)/(3*0.0025) - exp(-(10*(0.5-x)/(3*0.0025)))
#     sol = roots(f, (-4..Inf))
#     print(sol)
# end


##
t = 0.2:0.05:0.4
t₁_array = Vector{Float64}()
f(t₁) = E₁*(2*tₘ-t₁-t) + 3*η*(2*exp(-(E₂*(t-tₘ)/(3*η)))-exp(-(E₂*(t-t₁)/(3*η)))-1)
for i in t
    t = i
    push!(t₁_array, find_zero(f, [-Inf, Inf]))

end
##
print(t₁_array)


f(t₁) = E₁*(2*t_max-t₁-t) + 3*η*(2*exp(-(E₂*(t-t_max)/(3*η)))-exp(-(E₂*(t-t₁)/(3*η)))-1)
x = find_zero(f, [-Inf,Inf])
clamp(x, zero(Float64), Inf)
typeof(x)

##


sol = roots(f, (-Inf..Inf))
a = sol[1].interval

print(a)
setformat(:standard)

print(typeof(a))





sol = roots(f, (-Inf..Inf))
sol2 = find_zero(f,[-Inf, Inf])
print(sol[1].interval)
##
a = Interval(1.0,2.0)
a.mean