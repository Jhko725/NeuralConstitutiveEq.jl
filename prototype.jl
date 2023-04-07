##
using Plots
using Integrals
using IntegralsCubature
using NonlinearSolve
using Flux
include("./constitutive.jl")
include("./loading.jl")
##
function t₁_analytic(t, loading::Triangular, constit_eqn::PowerLawRheology)
    t₁ = @. t - 2^(1 / (1 - constit_eqn.γ)) * (t - loading.t_max)
    return clamp.(t₁, 0, Inf)
end
##
loading = Triangular(10, 0.2)
plr = PowerLawRheology(572, 1.0, 0.42)
t = LinRange(0.2, 0.4, 100)
##
plot(t, map(x -> t₁_analytic(x, loading, plr), t))
##
function PLR_integrand(t_::Float64, t::Float64, loading::Triangular, constit_eqn::PowerLawRheology)::Float64
    v = loading.v
    E₀, t₀, γ = constit_eqn.E₀, constit_eqn.t₀, constit_eqn.γ
    return v * E₀ * ((t - t_) / t₀)^(-γ)
end

PRL_integrand(t_, params) = PLR_integrand(t_, params...)
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

prob = IntegralProblem(integrand, 0.17696173488041395, 0.2, [0.21, loading, plr], batch=0)
solve(prob, CubatureJLp())
##
prob = IntegralProblem(integrand, 0.2, 0.21, [0.21, loading, plr])
solve(prob, CubatureJLh(), abstol=1e-10)
##
function ting_residual(t₁, t, loading::Triangular, constit_eqn::PowerLawRheology)
    t_max = loading.t_max |> Float64
    param = [t, loading, constit_eqn]
    app = IntegralProblem(integrand, Float64(t₁), t_max, param)
    ret = IntegralProblem(integrand, t_max, Float64(t), param)
    return solve(app, QuadGKJL()).u - solve(ret, QuadGKJL()).u
end

ting_residual(t₁, params) = ting_residual(t₁, params...)
##
ting_residual(0.17696173488041395, 0.21, loading, plr)
##
nonlinear_prob = NonlinearProblem(ting_residual, 0.2, [0.21, loading, plr])
##
solver = solve(nonlinear_prob, SimpleNewtonRaphson(autodiff=Val{false}()), reltol=1e-9)
##
model = Chain(Dense(1 => 32, tanh), Dense(32 => 1))
nn_integrand = IntegralProblem((x, p) -> model(x), [0.0], [1.0], batch=1)
solve(nn_integrand, GaussLegendre())
##
model([1.0])
