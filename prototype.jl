##
using Plots
using Integrals
include("./constitutive.jl")
include("./loading.jl")
##
function t₁_analytic(t, loading::Triangular, constit_eqn::PowerLawRheology)
    t₁ = t - 2^(1 / (1 - constit_eqn.γ)) * (t - loading.t_max)
    return clamp(t₁, 0, Inf)
end
##
loading = Triangular(10, 0.2)
plr = PowerLawRheology(572, 1.0, 0.42)
t = LinRange(0.2, 0.4, 100) |> collect
##
plot(t, map(x -> t₁_analytic(x, loading, plr), t))
##
function PLR_integrand(t_, t, loading::Triangular, constit_eqn::PowerLawRheology)
    v = loading.v
    E₀, t₀, γ = constit_eqn.E₀, constit_eqn.t₀, constit_eqn.γ
    return v * E₀ * ((t - t_) / t₀)^(-γ)
end

integrand(t_, params) = integrand(t_, params...)
##
IntegralProblem(integrand, 0.17696173488041395, 0.2, p=[])

##
# KWW model

kww = KWW(1.0, 2.0)
t = LinRange(0.2, 0.4, 100)

function KWW_integrand(t_, t, constit_eqn::KWW)
    τ, β = constit_eqn.τ, constit_eqn.β
    return @. exp((-(t-t_)/τ)^β)
end

