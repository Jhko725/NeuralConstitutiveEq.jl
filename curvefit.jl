##
import SpecialFunctions
using LsqFit
using Plots
include("./constitutive.jl")
include("./loading.jl")
include("./tipgeometry.jl")

function analytic_t₁(constit_eqn::PowerLawRheology{T}, loading::Triangular{T})::Function where {T<:AbstractFloat}
    γ = constit_eqn.γ
    t_max = loading.t_max
    coeff = T(2.0^(1.0 / (1.0 - γ)))

    t₁(t::T)::T = clamp(t - coeff * (t - t_max), zero(T), Inf)
    return t₁
end

function analytic_force(constit_eqn::PowerLawRheology{T}, loading::Triangular{T}, tip::AbstractTipGeometry)::Function where {T<:AbstractFloat}
    E₀, t₀, γ = constit_eqn.E₀, constit_eqn.t₀, constit_eqn.γ
    v, t_max = loading.v, loading.t_max
    a, b = α(tip), β(tip)
    coeff = E₀ * t₀^γ * a * b * v^b * SpecialFunctions.beta(b, 1 - γ)
    t₁ = analytic_t₁(constit_eqn, loading)

    F(t::T)::T = t <= t_max ? coeff * t^(b - γ) : coeff * t₁(t)^(b - γ)
    return F
end
##
loading = Triangular(10.0, 0.2) # 10um/s, 0.2
plr = PowerLawRheology(0.572, 1.0, 0.2) # 572Pa -> 0.572
tip = Conical(π / 10.0)
t₁ = analytic_t₁(plr, loading)
F = analytic_force(plr, loading, tip)
##
let
    t_s = 2e-3
    t_data = 0.0:t_s:2*loading.t_max
    f_true = F.(t_data)
    p = plot(t_data, f_true)
    noise = randn(Float64, size(f_true)) * 0.1
    scatter!(t_data, f_true + noise)
end
##
let
    t_s = 2e-3
    noise_amp = 0.1
    t_data = 0.0:t_s:2*loading.t_max
    f_true = F.(t_data)
    noise = randn(Float64, size(f_true)) * noise_amp
    f_data = f_true + noise

    function f_fit(t, params)
        force_func = analytic_force(PowerLawRheology(params[1], 1.0, params[2]), loading, tip)
        return force_func.(t)
    end


    fit = curve_fit(f_fit, t_data, f_data, [0.1, 0.5], lower=[0.0, 0.0], upper=[1.0, Inf])
    print(fit.param)
    print(standard_errors(fit))
end
##
function run_experiment(E₀, γ, noise_amp, t_s, app_only=true)
    tip = Conical(π / 10.0)
    loading = Triangular(10.0, 0.2) # 10um/s, 0.2
    plr = PowerLawRheology(E₀, 1.0, γ)
    F = analytic_force(plr, loading, tip)

    if app_only
        t_end = t_s
    else
        t_end = 2t_s
    end
    t_data = 0.0:t_s:2*loading.t_max
    f_true = F.(t_data)
    noise = randn(Float64, size(f_true)) * noise_amp
    f_data = f_true + noise

    function f_fit(t, params)
        force_func = analytic_force(PowerLawRheology(params[1], 1.0, params[2]), loading, tip)
        return force_func.(t)
    end

    fit = curve_fit(f_fit, t_data, f_data, [0.1, 0.5], lower=[0.0, 0.0], upper=[1.0, Inf])
    param = fit.param
    se = standard_errors(fit)
    return cat(param, se, dims=2)
end
##
let
    E₀_true, γ_true = 0.572, 0.8
    f_s = 2e-3
    noise_amps = 0:0.01:0.3
    result_app = map(x -> run_experiment(E₀_true, γ_true, x, f_s, true), noise_amps) |> x -> cat(x..., dims=3)
    result_total = map(x -> run_experiment(E₀_true, γ_true, x, f_s, false), noise_amps) |> x -> cat(x..., dims=3)
    plot_E = scatter(noise_amps, result_app[1, 1, :], yerror=result_app[1, 2, :], label="Approach only")
    scatter!(plot_E, noise_amps, result_total[1, 1, :], yerror=result_total[1, 2, :], label="Approach & Retract")
    hline!(plot_E, [E₀_true], label="Ground truth")
    plot_γ = scatter(noise_amps, result_app[2, 1, :], yerror=result_app[2, 2, :], label="Approach only")
    scatter!(plot_γ, noise_amps, result_total[2, 1, :], yerror=result_total[2, 2, :], label="Approach & Retract")
    hline!(plot_γ, [γ_true], label="Ground truth")
    plot(plot_E, plot_γ, layout=(1, 2))
end
##
let
    E₀_true, noise_amp = 0.572, 0.05
    f_s = 2e-3
    γ_trues = 0.0:0.05:0.995
    result_app = map(x -> run_experiment(E₀_true, x, noise_amp, f_s, true), γ_trues) |> x -> cat(x..., dims=3)
    result_total = map(x -> run_experiment(E₀_true, x, noise_amp, f_s, false), γ_trues) |> x -> cat(x..., dims=3)
    plot_E = scatter(γ_trues, result_app[1, 1, :], yerror=result_app[1, 2, :], label="Approach only")
    scatter!(plot_E, γ_trues, result_total[1, 1, :], yerror=result_total[1, 2, :], label="Approach & Retract")
    hline!(plot_E, [E₀_true], label="Ground truth")
    plot_γ = scatter(γ_trues, result_app[2, 1, :] - γ_trues, yerror=result_app[2, 2, :], label="Approach only")
    scatter!(plot_γ, γ_trues, result_total[2, 1, :] - γ_trues, yerror=result_total[2, 2, :], label="Approach & Retract")
    hline!(plot_γ, [0.0], label="Ground truth")
    plot(plot_E, plot_γ, layout=(1, 2))
end