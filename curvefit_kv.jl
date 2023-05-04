##
using LsqFit
using Plots
include("./constitutive.jl")
include("./indentation.jl")
include("./tipgeometry.jl")
include("force.jl")
##
loading = Triangular(10.0, 0.2) # 10um/s, 0.2
plr = PowerLawRheology(0.572, 1.0, 0.2) # 572Pa -> 0.572
kv = KelvinVoigt(1.614, 0.025) # 1614Pa -> 1.614, 25Pa.s -> 0.025 Pa.s
tip = Conical(π / 10.0)
F(t) = force(t, plr, loading, tip)
F_kv(t) = force_kv(t, kv, loading, tip)
##
let
    t_s = 2e-3
    t_data = 0.0:t_s:2*loading.t_max
    f_true = F_kv.(t_data)
    p = plot(t_data, f_true, label="KV_Analytic")
    noise = randn(Float64, size(f_true)) * 0.04
    scatter!(t_data, f_true+noise, label="KV_Simulation")
end
##
let
    import Statistics: mean
    t_s = 2e-3
    noise_amp = 0.08
    t_data = 0.0:t_s:2*loading.t_max
    f_true = F.(t_data)
    noise = randn(Float64, size(f_true)) * noise_amp
    f_data = f_true + noise

    function f_fit(t, params)
        _force(t) = force(t, PowerLawRheology(params[1], 1.0, params[2]), loading, tip)
        return _force.(t)
    end

    fit = curve_fit(f_fit, t_data, f_data, [0.1, 0.5], lower=[0.0, 0.0], upper=[1.0, Inf])
    print(fit.param)
    print(standard_errors(fit))
    print(fit.resid)
    mse = mean(fit.resid.^2)
    print(mse)
end
##
let
    t_s = 2e-3
    noise_amp = 0.08
    t_data = 0.0:t_s:2*loading.t_max
    f_true = F_kv.(t_data)
    noise = randn(Float64, size(f_true)) * noise_amp
    f_data = f_true + noise

    function f_fit_kv(t, params)
        _force(t) = force_kv(t, KelvinVoigt(params[1], params[2]), loading, tip)
        return _force.(t)
    end

    fit = curve_fit(f_fit_kv, t_data, f_data, [1.0, 0.1], lower=[0.0, 0.0], upper=[Inf, Inf])
    print(fit.param)
    print(standard_errors(fit))
end
##
function run_experiment_kv(E, η, noise_amp, t_s, app_only=true)
    tip = Conical(π / 10.0)
    loading = Triangular(10.0, 0.2) # 10um/s, 0.2
    kv = KelvinVoigt(E, η)
    F(t) = force_kv(t, kv, loading, tip)

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
        _force(t) = force_kv(t, KelvinVoigt(params[1], params[2]), loading, tip)
        return _force.(t)
    end

    fit = curve_fit(f_fit, t_data, f_data, [1.0, 0.07], lower=[0.0, 0.0], upper=[Inf, Inf])
    param = fit.param
    se = standard_errors(fit)
    return cat(param, se, dims=2)
end
##
let 
   run_experiment_kv(1.614, 0.08, 0.02, 2e-3)[2,1] 
end
##
let
E_true, η_true = 1.614, 0.025
    f_s = 2e-3
    noise_amps = 0:0.002:0.04
    result_app = map(x -> run_experiment_kv(E_true, η_true, x, f_s, true), noise_amps) |> x -> cat(x..., dims=3)
    result_total = map(x -> run_experiment_kv(E_true, η_true, x, f_s, false), noise_amps) |> x -> cat(x..., dims=3)
    plot_E = scatter(noise_amps, result_app[1, 1, :], yerror=result_app[1, 2, :], label="Approach only", ylabel="E")
    scatter!(plot_E, noise_amps, result_total[1, 1, :], yerror=result_total[1, 2, :], label="Approach & Retract")
    hline!(plot_E, [E_true], label="Ground truth")
    plot_η = scatter(noise_amps, result_app[2, 1, :], yerror=result_app[2, 2, :], label="Approach only", ylabel="η")
    scatter!(plot_η, noise_amps, result_total[2, 1, :], yerror=result_total[2, 2, :], label="Approach & Retract")
    hline!(plot_η, [η_true], label="Ground truth")
    plot(plot_E, plot_η, layout=(1, 2), xlabel="Noise amplitude")
end
##
let
    E_true, noise_amp = 1.614, 0.02
    f_s = 2e-3
    η_trues = 0.0:0.005:0.1
    result_app = map(x -> run_experiment_kv(E_true, x, noise_amp, f_s, true), η_trues) |> x -> cat(x..., dims=3)
    result_total = map(x -> run_experiment_kv(E_true, x, noise_amp, f_s, false), η_trues) |> x -> cat(x..., dims=3)
    plot_E = scatter(η_trues, result_app[1, 1, :], yerror=result_app[1, 2, :], label="Approach only", ylabel="E_true")
    scatter!(plot_E, η_trues, result_total[1, 1, :], yerror=result_total[1, 2, :], label="Approach & Retract")
    hline!(plot_E, [E_true], label="Ground truth")
    plot_η = scatter(η_trues, result_app[2, 1, :] - η_trues, yerror=result_app[2, 2, :], label="Approach only", ylabel="η_fit - η_true")
    scatter!(plot_η, η_trues, result_total[2, 1, :] - η_trues, yerror=result_total[2, 2, :], label="Approach & Retract")
    print(result_app[2, 1, :])
    hline!(plot_η, [0.0], label="Ground truth")
    plot(plot_E, plot_η, layout=(1, 2), xlabel="η")
end